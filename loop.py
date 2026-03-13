"""
Autonomous experiment loop runner using Claude Agent SDK.

Usage:
    uv run loop.py              # interactive (asks for duration)
    uv run loop.py --hours 6    # skip prompt, run 6 hours

Ctrl+C to stop cleanly. Ctrl+C twice to force quit.
"""

import argparse
import asyncio
import os
import re
import signal
import subprocess
import sys
import time

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

# ── ANSI ──────────────────────────────────────────────────────────────────────

DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
RESET = "\033[0m"
CLEAR_LINE = "\033[2K\r"

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(SCRIPT_DIR, "results.tsv")
LOG_PATH = os.path.join(SCRIPT_DIR, "run.log")
PROGRAM_PATH = os.path.join(SCRIPT_DIR, "program.md")

BAR_WIDTH = 25

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = open(PROGRAM_PATH).read()

FIRST_MESSAGE = """\
Start the experiment loop. The branch and baseline are already set up.
Read results.tsv for current state, then begin experimenting.
You are fully autonomous — run experiments continuously without stopping.
Each cycle: decide what to try, modify train.py, commit, run training, \
evaluate, update results.tsv, keep or discard, then start the next one.
Do NOT ask for permission or confirmation. Do NOT stop between experiments.\
"""

NEXT_MESSAGE = "Continue. Run the next experiment immediately."


# ── Helpers ───────────────────────────────────────────────────────────────────

def out(msg: str = "") -> None:
    print(msg, flush=True)


def banner(msg: str) -> None:
    out(f"\n{BOLD}{'─' * 60}{RESET}")
    out(f"{BOLD}  {msg}{RESET}")
    out(f"{BOLD}{'─' * 60}{RESET}\n")


def read_results_tsv() -> list[dict]:
    if not os.path.exists(RESULTS_PATH):
        return []
    rows = []
    with open(RESULTS_PATH) as f:
        headers = f.readline().strip().split("\t")
        for line in f:
            vals = line.strip().split("\t")
            if len(vals) == len(headers):
                rows.append(dict(zip(headers, vals)))
    return rows


# ── Interactive startup ──────────────────────────────────────────────────────

def interactive_startup(skip_hours: float | None = None) -> float:
    """Show current state, ask for duration, return hours."""
    results = read_results_tsv()
    branch = "?"
    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=SCRIPT_DIR, text=True,
        ).strip()
    except Exception:
        pass

    out(f"\n  {BOLD}autoresearch-mlx{RESET}")
    out(f"  {DIM}branch:{RESET} {branch}   "
        f"{DIM}experiments:{RESET} {len(results)}")

    if results:
        kept = [r for r in results if r.get("status") == "keep"]
        if kept:
            best = min(kept, key=lambda r: float(r.get("val_bpb", "999")))
            out(f"  {DIM}best:{RESET} {GREEN}{best.get('val_bpb', '?')}{RESET} bpb"
                f"  {DIM}({best.get('description', '?')}){RESET}")
    out("")

    if skip_hours is not None:
        out(f"  Duration: {skip_hours}h\n")
        return skip_hours

    try:
        raw = input(f"  Duration in hours {DIM}[6]{RESET}: ").strip().strip("\r")
        hours = float(raw) if raw else 6.0
    except (ValueError, EOFError):
        hours = 6.0

    out("")
    return hours


# ── Progress bar ──────────────────────────────────────────────────────────────

STEP_RE = re.compile(
    r"step\s+\d+\s+\((\d+\.\d+)%\)\s+\|\s+loss:\s+([\d.]+).*?"
    r"tok/sec:\s+([\d,]+).*?remaining:\s+(\d+)s"
)


def parse_train_line(line: str) -> dict | None:
    m = STEP_RE.search(line)
    if not m:
        return None
    return {
        "pct": float(m.group(1)),
        "loss": float(m.group(2)),
        "tps": m.group(3),
        "remaining": int(m.group(4)),
    }


def render_bar(pct: float, loss: float, tps: str, remaining: int) -> str:
    filled = int(BAR_WIDTH * pct / 100)
    empty = BAR_WIDTH - filled
    bar = f"{GREEN}{'█' * filled}{DIM}{'░' * empty}{RESET}"
    return (
        f"  {CYAN}▶{RESET} {pct:5.1f}% {bar} "
        f"{DIM}loss {loss:.3f}  {tps} tok/s  {remaining}s left{RESET}"
    )


async def train_progress_poller(stop_event: asyncio.Event) -> None:
    """Tail run.log and render a progress bar until stop_event is set."""
    initial_size = os.path.getsize(LOG_PATH) if os.path.exists(LOG_PATH) else 0

    # Wait up to 30s for log to start growing
    for _ in range(15):
        if stop_event.is_set():
            return
        if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > initial_size:
            break
        await asyncio.sleep(2)

    while not stop_event.is_set():
        try:
            if os.path.exists(LOG_PATH):
                with open(LOG_PATH, "rb") as f:
                    f.seek(max(0, f.seek(0, 2) - 4096))
                    tail = f.read().decode("utf-8", errors="ignore")

                for segment in reversed(tail.replace("\r", "\n").split("\n")):
                    parsed = parse_train_line(segment)
                    if parsed:
                        print(
                            f"{CLEAR_LINE}"
                            f"{render_bar(parsed['pct'], parsed['loss'], parsed['tps'], parsed['remaining'])}",
                            end="", flush=True,
                        )
                        break
        except Exception:
            pass

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass

    # Clear progress line
    print(CLEAR_LINE, end="", flush=True)


# ── Tool formatting ──────────────────────────────────────────────────────────

def is_training_cmd(cmd: str) -> bool:
    return "train.py" in cmd and ("uv run" in cmd or "python" in cmd) and ">" in cmd


def format_tool(name: str, inp: dict) -> str | None:
    """Format a tool call. Returns '__TRAINING__' sentinel for training runs."""
    if name == "Read":
        path = inp.get("file_path", "?")
        return f"{DIM}  read {os.path.basename(path)}{RESET}"

    if name == "Write":
        path = inp.get("file_path", "?")
        return f"{YELLOW}  write {os.path.basename(path)}{RESET}"

    if name == "Edit":
        path = inp.get("file_path", "?")
        old = inp.get("old_string", "")
        preview = old[:50].replace("\n", " ").strip()
        if len(old) > 50:
            preview += "..."
        return f"{YELLOW}  edit {os.path.basename(path)}{RESET} {DIM}{preview}{RESET}"

    if name == "Bash":
        cmd = inp.get("command", "")
        desc = inp.get("description", "")
        if is_training_cmd(cmd):
            return "__TRAINING__"
        if cmd.startswith("git "):
            return f"{DIM}  $ {cmd[:80]}{RESET}"
        if desc:
            return f"{DIM}  $ {desc}{RESET}"
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        return f"{DIM}  $ {cmd}{RESET}"

    if name in ("Glob", "Grep"):
        pattern = inp.get("pattern", "?")
        return f"{DIM}  {name.lower()} {pattern[:60]}{RESET}"

    return f"{DIM}  [{name}]{RESET}"


# ── Scoreboard ────────────────────────────────────────────────────────────────

def print_scoreboard() -> None:
    results = read_results_tsv()
    if not results:
        return

    kept = [r for r in results if r.get("status") == "keep"]

    out(f"\n{BOLD}  Scoreboard{RESET}")
    out(f"  {DIM}{'─' * 50}{RESET}")

    for r in results:
        status = r.get("status", "?")
        bpb = r.get("val_bpb", "?")
        desc = r.get("description", "?")
        mem = r.get("memory_gb", "?")

        if status == "keep":
            out(f"  {GREEN}✓ {bpb}{RESET}  {DIM}{mem}GB{RESET}  {desc}")
        elif status == "discard":
            out(f"  {DIM}✗ {bpb}  {mem}GB  {desc}{RESET}")
        else:
            out(f"  {RED}! crash{RESET}  {DIM}{desc}{RESET}")

    if kept:
        best = min(kept, key=lambda r: float(r.get("val_bpb", "999")))
        baseline = kept[0]
        base_bpb = float(baseline.get("val_bpb", "0"))
        best_bpb = float(best.get("val_bpb", "0"))
        if base_bpb > 0:
            improvement = (base_bpb - best_bpb) / base_bpb * 100
            out(f"\n  {BOLD}Best: {best_bpb:.6f}{RESET} "
                f"{GREEN}({improvement:+.1f}% vs baseline){RESET}")
    out("")


def print_final_summary(
    start_time: float, rounds: int, total_cost: float,
) -> None:
    elapsed_h = (time.time() - start_time) / 3600
    results = read_results_tsv()

    banner(f"Session complete — {rounds} rounds in {elapsed_h:.1f}h | ${total_cost:.2f}")

    if results:
        kept = [r for r in results if r.get("status") == "keep"]
        discarded = [r for r in results if r.get("status") == "discard"]
        crashed = [r for r in results if r.get("status") == "crash"]

        out(f"  {GREEN}Kept: {len(kept)}{RESET}  "
            f"{DIM}Discarded: {len(discarded)}  Crashed: {len(crashed)}{RESET}")

        if kept:
            best = min(kept, key=lambda r: float(r.get("val_bpb", "999")))
            baseline = kept[0]
            out(f"  Baseline: {baseline.get('val_bpb', '?')}")
            out(f"  {BOLD}Best:     {best.get('val_bpb', '?')}"
                f" — {best.get('description', '?')}{RESET}")
    else:
        out("  No results found.")
    out("")


# ── Experiment round ──────────────────────────────────────────────────────────

async def run_round(
    client: ClaudeSDKClient, round_num: int,
) -> tuple[float, float]:
    """Run one experiment round. Returns (cost_usd, duration_s)."""
    prompt = FIRST_MESSAGE if round_num == 1 else NEXT_MESSAGE

    active_training_id: str | None = None
    progress_task: asyncio.Task | None = None
    stop_event: asyncio.Event | None = None

    try:
        await client.query(prompt)

        async for message in client.receive_response():
            # ── Assistant messages: text + tool calls ──
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text = block.text.strip()
                        if text:
                            for line in text.split("\n"):
                                line = line.strip()
                                if line:
                                    out(f"  {line}")

                    elif isinstance(block, ToolUseBlock):
                        formatted = format_tool(block.name, block.input or {})

                        if formatted == "__TRAINING__":
                            out(f"  {CYAN}{BOLD}▶ Training{RESET} {DIM}(5 min){RESET}")
                            active_training_id = block.id
                            stop_event = asyncio.Event()
                            progress_task = asyncio.create_task(
                                train_progress_poller(stop_event)
                            )
                        else:
                            # New tool call = training must be done
                            if progress_task and not progress_task.done():
                                stop_event.set()
                                await progress_task
                                progress_task = None
                                active_training_id = None

                            if formatted:
                                out(formatted)

                    elif isinstance(block, ToolResultBlock):
                        if block.tool_use_id == active_training_id:
                            if stop_event:
                                stop_event.set()
                            if progress_task:
                                await progress_task
                            progress_task = None
                            active_training_id = None

            # ── User messages: contain tool results ──
            elif isinstance(message, UserMessage):
                if isinstance(message.content, list):
                    for block in message.content:
                        if (
                            isinstance(block, ToolResultBlock)
                            and block.tool_use_id == active_training_id
                        ):
                            if stop_event:
                                stop_event.set()
                            if progress_task:
                                await progress_task
                            progress_task = None
                            active_training_id = None

            # ── Result: end of round ──
            elif isinstance(message, ResultMessage):
                cost = message.total_cost_usd or 0.0
                duration = message.duration_ms / 1000.0
                return (cost, duration)

    except asyncio.CancelledError:
        out(f"\n{YELLOW}  Interrupted.{RESET}")
        raise
    except Exception as e:
        out(f"\n{RED}  ERROR: {type(e).__name__}: {e}{RESET}")
    finally:
        if stop_event and not stop_event.is_set():
            stop_event.set()
        if progress_task and not progress_task.done():
            try:
                await progress_task
            except Exception:
                pass

    return (0.0, 0.0)


# ── Main loop ─────────────────────────────────────────────────────────────────

async def run_loop(hours: float) -> None:
    start_time = time.time()
    deadline = start_time + hours * 3600
    total_cost = 0.0
    round_num = 0
    stop_requested = False

    loop = asyncio.get_running_loop()

    def on_signal():
        nonlocal stop_requested
        if stop_requested:
            out(f"\n{RED}  Force quit.{RESET}")
            raise SystemExit(1)
        stop_requested = True
        out(f"\n{YELLOW}  Ctrl+C — stopping after current round...{RESET}")

    loop.add_signal_handler(signal.SIGINT, on_signal)
    loop.add_signal_handler(signal.SIGTERM, on_signal)

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        permission_mode="bypassPermissions",
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        cwd=SCRIPT_DIR,
        model="opus",
    )

    banner(f"autoresearch loop — {hours}h | Ctrl+C to stop")

    client = None
    try:
        client = ClaudeSDKClient(options=options)
        await client.connect()

        while time.time() < deadline and not stop_requested:
            remaining_min = (deadline - time.time()) / 60
            if remaining_min < 8:
                banner("Less than 8 min remaining — stopping")
                break

            round_num += 1

            if round_num > 1:
                print_scoreboard()

            banner(f"Round #{round_num} | {remaining_min:.0f} min remaining"
                   + (f" | ${total_cost:.2f}" if total_cost > 0 else ""))

            cost, duration = await run_round(client, round_num)
            total_cost += cost

            out(f"\n  {DIM}round: ${cost:.2f}  {duration:.0f}s"
                f"  |  session: ${total_cost:.2f}{RESET}")

    except SystemExit:
        pass
    except Exception as e:
        banner(f"Fatal: {type(e).__name__}: {e}")
    finally:
        loop.remove_signal_handler(signal.SIGINT)
        loop.remove_signal_handler(signal.SIGTERM)
        if client:
            try:
                await client.disconnect()
            except Exception:
                pass
        print_scoreboard()
        print_final_summary(start_time, round_num, total_cost)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous experiment loop runner",
        epilog="Ctrl+C to stop cleanly. Ctrl+C twice to force quit.",
    )
    parser.add_argument(
        "--hours", type=float, default=None,
        help="Duration in hours (skips interactive prompt)",
    )
    args = parser.parse_args()

    hours = interactive_startup(skip_hours=args.hours)

    try:
        asyncio.run(run_loop(hours))
    except KeyboardInterrupt:
        out(f"\n{RED}  Force quit.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
