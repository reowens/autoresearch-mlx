"""
Autonomous experiment loop. Runs ONE experiment per turn.

    uv run loop.py              # asks for number of runs
    uv run loop.py 10           # run 10 experiments
    uv run loop.py --dry-run    # show state without starting
    MODEL=sonnet uv run loop.py # use sonnet instead of opus
"""

import asyncio
import logging
import os
import subprocess
import sys
import time

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)
from claude_agent_sdk.types import StreamEvent

DIR = os.path.dirname(os.path.abspath(__file__))

_prompt_path = os.path.join(DIR, "program.md")
if not os.path.exists(_prompt_path):
    sys.exit(f"  Error: {_prompt_path} not found")
with open(_prompt_path) as f:
    PROMPT = f.read()

DEFAULT_TIME_BUDGET = 5  # minutes
OVERHEAD_MIN = 2  # ~2 min overhead per run (compile, eval, git)
MINS_PER_RUN = DEFAULT_TIME_BUDGET + OVERHEAD_MIN

MSG_FIRST = (
    "Read results.tsv, train.py, and prepare.py. Run exactly ONE experiment: "
    "modify train.py, commit, train, evaluate, update results.tsv, keep or discard. "
    "IMPORTANT: Run only ONE training run. If it crashes or diverges, log it and STOP. "
    "Do NOT revert and try something else — that counts as a second experiment. "
    "Stop after this single experiment is complete."
)
MSG_NEXT = (
    "Run exactly ONE more experiment, then stop. "
    "If it crashes or diverges, log it and stop — do not start another."
)


# ── Reporter protocol ────────────────────────────────────────────────────────

class LoopReporter:
    """Override methods to customize how loop events are displayed."""
    async def on_start(self, num_runs, model): pass
    async def on_round_start(self, round_num, num_runs, elapsed_min, total_cost): pass
    async def on_text(self, text): pass
    async def on_text_delta(self, chunk): pass
    async def on_tool_start(self, name): pass
    async def on_tool_use(self, name, label): pass
    async def on_training_detected(self): pass
    async def on_round_done(self, round_cost, total_cost, usage=None): pass
    async def on_round_failed(self, error): pass
    async def on_finished(self, num_runs, elapsed_min, total_cost, total_usage=None): pass
    async def on_stderr(self, line): pass


class CLIReporter(LoopReporter):
    """Plain terminal output (default)."""
    async def on_start(self, num_runs, model):
        est = num_runs * MINS_PER_RUN
        print(f"\n  Loop started — {num_runs} runs (~{est} min). Ctrl+C to stop.\n")

    async def on_round_start(self, round_num, num_runs, elapsed_min, total_cost):
        cost_str = f" | ${total_cost:.2f}" if total_cost > 0 else ""
        print(f"  === Round {round_num}/{num_runs} | {elapsed_min:.0f}m elapsed{cost_str} ===\n")

    async def on_text(self, text):
        print(f"  {text[:200]}")

    async def on_text_delta(self, chunk):
        print(chunk, end="", flush=True)

    async def on_tool_start(self, name):
        print(f"  [{name}] ", end="", flush=True)

    async def on_tool_use(self, name, label):
        print(f"  [{name}] {label}")

    async def on_training_detected(self):
        print("  > training (~5 min)...")

    async def on_round_done(self, round_cost, total_cost, usage=None):
        cost_note = " (included)" if round_cost == 0 else ""
        tokens = ""
        if usage:
            inp = usage.get("input_tokens", 0)
            out = usage.get("output_tokens", 0)
            tokens = f" | {(inp + out) // 1000}k tokens"
        print(f"  --- round done (${round_cost:.2f}{cost_note}{tokens} | total ${total_cost:.2f}) ---\n")

    async def on_round_failed(self, error):
        print(f"  --- round failed: {error} ---\n")

    async def on_stderr(self, line):
        print(f"  [SDK] {line}", file=sys.stderr)

    async def on_finished(self, num_runs, elapsed_min, total_cost, total_usage=None):
        tokens = ""
        if total_usage:
            inp = total_usage.get("input_tokens", 0)
            out = total_usage.get("output_tokens", 0)
            tokens = f", {(inp + out) // 1000}k tokens"
        print(f"\n  Done — {num_runs} rounds, {elapsed_min:.0f}m, ${total_cost:.2f}{tokens}.")
        print_results()


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_results():
    path = os.path.join(DIR, "results.tsv")
    if not os.path.exists(path):
        print("  No results.tsv found.")
        return
    print()
    with open(path) as f:
        for line in f:
            print(f"  {line.rstrip()}")
    print()


def _tool_label(name, inp, cmd):
    """Extract a human-readable label from a tool use block."""
    if name == "Bash":
        return inp.get("description", "") or cmd[:80] or "bash"
    if name in ("Read", "Write", "Edit"):
        path = inp.get("file_path", "")
        return os.path.basename(path) if path else name
    if name in ("Glob", "Grep"):
        return inp.get("pattern", "")[:60] or name
    return name


# ── Core loop ─────────────────────────────────────────────────────────────────

log = logging.getLogger("loop")


async def run(num_runs, reporter=None, config=None):
    if reporter is None:
        reporter = CLIReporter()
    config = config or {}

    start = time.time()
    total_cost = 0.0
    total_usage = {"input_tokens": 0, "output_tokens": 0}

    model = config.get("model", os.environ.get("MODEL", "opus"))
    effort = config.get("effort", "medium")
    api_key = config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")

    time_budget_min = config.get("time_budget", DEFAULT_TIME_BUDGET)
    time_budget_sec = time_budget_min * 60
    mins_per_run = time_budget_min + OVERHEAD_MIN

    env = {"TIME_BUDGET": str(time_budget_sec)}
    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key

    sdk_log = logging.getLogger("claude_sdk_stderr")

    def _on_stderr(line):
        sdk_log.warning("SDK: %s", line.rstrip())
        # Note: can't await here since this is a sync callback
        # The dashboard reporter's on_stderr just logs, no mount() needed

    opts = ClaudeAgentOptions(
        system_prompt=PROMPT,
        permission_mode="bypassPermissions",
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        cwd=DIR,
        model=model,
        effort=effort,
        env=env,
        stderr=_on_stderr,
        include_partial_messages=True,
    )
    if api_key:
        try:
            opts.betas = ["context-1m-2025-08-07"]
        except Exception:
            pass

    await reporter.on_start(num_runs, model)
    log.info("Config: model=%s effort=%s time_budget=%dm", model, effort, time_budget_min)

    for round_num in range(1, num_runs + 1):
        elapsed = (time.time() - start) / 60
        await reporter.on_round_start(round_num, num_runs, elapsed, total_cost)

        prefix = f"[Round {round_num}/{num_runs}] "
        msg = prefix + (MSG_FIRST if round_num == 1 else MSG_NEXT)
        try:
            # Fresh client per round — prevents context accumulation
            async with ClaudeSDKClient(options=opts) as client:
                await client.query(msg)
                log.info("Round %d: query sent", round_num)

                in_tool = False

                async for m in client.receive_response():
                    # StreamEvent — real-time streaming chunks
                    if isinstance(m, StreamEvent):
                        event = m.event
                        etype = event.get("type", "")

                        if etype == "content_block_start":
                            cb = event.get("content_block", {})
                            if cb.get("type") == "tool_use":
                                in_tool = True
                                await reporter.on_tool_start(cb.get("name", ""))
                            else:
                                in_tool = False

                        elif etype == "content_block_delta":
                            delta = event.get("delta", {})
                            if delta.get("type") == "text_delta" and not in_tool:
                                chunk = delta.get("text", "")
                                if chunk:
                                    await reporter.on_text_delta(chunk)

                        elif etype == "content_block_stop":
                            in_tool = False

                    # AssistantMessage — complete message with tool calls
                    elif isinstance(m, AssistantMessage):
                        for b in m.content:
                            if isinstance(b, TextBlock) and b.text.strip():
                                await reporter.on_text(b.text.strip())
                            elif isinstance(b, ToolUseBlock):
                                inp = b.input or {}
                                cmd = inp.get("command", "") if b.name == "Bash" else ""
                                if b.name == "Bash" and cmd:
                                    log.debug("Bash: %s", cmd[:120])
                                if "uv run" in cmd and "train.py" in cmd and ">" in cmd:
                                    log.info("Training detected")
                                    await reporter.on_training_detected()
                                else:
                                    label = _tool_label(b.name, inp, cmd)
                                    await reporter.on_tool_use(b.name, label)

                    # ResultMessage — turn complete
                    elif isinstance(m, ResultMessage):
                        cost = m.total_cost_usd or 0
                        total_cost += cost
                        usage = m.usage or {}
                        if usage:
                            total_usage["input_tokens"] += usage.get("input_tokens", 0)
                            total_usage["output_tokens"] += usage.get("output_tokens", 0)
                            log.info("Round %d usage: %s", round_num, usage)
                        if hasattr(m, 'subtype') and m.subtype == "error_max_turns":
                            log.warning("Round %d hit max_turns limit", round_num)
                            await reporter.on_round_failed("hit turn limit")
                        else:
                            await reporter.on_round_done(cost, total_cost, usage)
                        break

        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.exception("Round %d failed", round_num)
            await reporter.on_round_failed(f"{type(e).__name__}: {e}")

    elapsed = (time.time() - start) / 60
    await reporter.on_finished(num_runs, elapsed, total_cost, total_usage)


# ── CLI entry point ───────────────────────────────────────────────────────────

def dry_run():
    branch = "?"
    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=DIR, text=True,
        ).strip()
    except Exception:
        pass
    model = os.environ.get("MODEL", "opus")
    print(f"\n  branch: {branch}  model: {model}")
    print_results()


def main():
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    if "--dry-run" in sys.argv:
        dry_run()
        return

    if args:
        try:
            num_runs = int(args[0])
            if num_runs <= 0:
                raise ValueError("must be positive")
        except ValueError as e:
            sys.exit(f"  Error: {e}")
    else:
        try:
            raw = input(f"  Number of runs [10] (~{10 * MINS_PER_RUN} min): ").strip()
            num_runs = int(raw) if raw else 10
        except (ValueError, EOFError, KeyboardInterrupt):
            num_runs = 10

    print()

    try:
        asyncio.run(run(num_runs))
    except KeyboardInterrupt:
        print("\n  Stopped.")
        print_results()


if __name__ == "__main__":
    main()
