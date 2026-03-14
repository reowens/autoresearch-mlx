"""
Autonomous experiment loop. Runs ONE experiment per turn.

    uv run loop.py              # asks for number of runs
    uv run loop.py 10           # run 10 experiments
    uv run loop.py --dry-run    # show state without starting
    MODEL=sonnet uv run loop.py # use sonnet instead of opus
"""

import asyncio
import os
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

MINS_PER_RUN = 7  # ~5 min training + ~2 min overhead

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
    def on_start(self, num_runs, model): pass
    def on_round_start(self, round_num, num_runs, elapsed_min, total_cost): pass
    def on_text(self, text): pass
    def on_text_delta(self, chunk): pass  # streaming text chunk
    def on_tool_start(self, name): pass   # tool call beginning
    def on_tool_use(self, name, label): pass
    def on_training_detected(self): pass
    def on_round_done(self, round_cost, total_cost): pass
    def on_round_failed(self, error): pass
    def on_finished(self, num_runs, elapsed_min, total_cost): pass
    def on_stderr(self, line): pass


class CLIReporter(LoopReporter):
    """Plain terminal output (default)."""
    def on_start(self, num_runs, model):
        est = num_runs * MINS_PER_RUN
        print(f"\n  Loop started — {num_runs} runs (~{est} min). Ctrl+C to stop.\n")

    def on_round_start(self, round_num, num_runs, elapsed_min, total_cost):
        cost_str = f" | ${total_cost:.2f}" if total_cost > 0 else ""
        print(f"  === Round {round_num}/{num_runs} | {elapsed_min:.0f}m elapsed{cost_str} ===\n")

    def on_text(self, text):
        print(f"  {text[:200]}")

    def on_tool_use(self, name, label):
        print(f"  [{name}] {label}")

    def on_text_delta(self, chunk):
        print(chunk, end="", flush=True)

    def on_tool_start(self, name):
        print(f"  [{name}] ", end="", flush=True)

    def on_training_detected(self):
        print("  > training (~5 min)...")

    def on_round_done(self, round_cost, total_cost):
        cost_note = " (included)" if round_cost == 0 else ""
        print(f"  --- round done (${round_cost:.2f}{cost_note} | total ${total_cost:.2f}) ---\n")

    def on_round_failed(self, error):
        print(f"  --- round failed: {error} ---\n")

    def on_stderr(self, line):
        print(f"  [SDK] {line}", file=sys.stderr)

    def on_finished(self, num_runs, elapsed_min, total_cost):
        print(f"\n  Done — {num_runs} rounds, {elapsed_min:.0f}m, ${total_cost:.2f}.")
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
    if name in ("Read", "Write"):
        path = inp.get("file_path", "")
        return os.path.basename(path) if path else name
    if name == "Edit":
        path = inp.get("file_path", "")
        return os.path.basename(path) if path else name
    if name in ("Glob", "Grep"):
        return inp.get("pattern", "")[:60] or name
    return name


# ── Core loop ─────────────────────────────────────────────────────────────────

async def run(num_runs, reporter=None, config=None):
    if reporter is None:
        reporter = CLIReporter()
    config = config or {}

    start = time.time()
    total_cost = 0.0

    model = config.get("model", os.environ.get("MODEL", "opus"))
    effort = config.get("effort", "high")
    api_key = config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")

    env = {}
    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key

    import logging
    sdk_log = logging.getLogger("claude_sdk_stderr")

    def _on_stderr(line):
        sdk_log.warning("SDK: %s", line.rstrip())
        reporter.on_stderr(line.rstrip())

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
    # Enable 1M context for API users only
    if api_key:
        try:
            opts.betas = ["context-1m-2025-08-07"]
        except Exception:
            pass

    reporter.on_start(num_runs, model)

    import logging
    loop_log = logging.getLogger("loop.run")

    async with ClaudeSDKClient(options=opts) as client:
        for round_num in range(1, num_runs + 1):
            elapsed = (time.time() - start) / 60
            reporter.on_round_start(round_num, num_runs, elapsed, total_cost)

            prefix = f"[Round {round_num}/{num_runs}] "
            msg = prefix + (MSG_FIRST if round_num == 1 else MSG_NEXT)
            try:
                await client.query(msg)
                loop_log.info("query sent, waiting for response")

                in_tool = False
                current_tool_name = None

                async for m in client.receive_response():
                    if isinstance(m, StreamEvent):
                        event = m.event if hasattr(m, 'event') else m
                        etype = event.get("type", "") if isinstance(event, dict) else ""

                        if etype == "content_block_start":
                            cb = event.get("content_block", {})
                            if cb.get("type") == "tool_use":
                                in_tool = True
                                current_tool_name = cb.get("name", "")
                                reporter.on_tool_start(current_tool_name)
                            else:
                                in_tool = False

                        elif etype == "content_block_delta":
                            delta = event.get("delta", {})
                            if delta.get("type") == "text_delta" and not in_tool:
                                chunk = delta.get("text", "")
                                if chunk:
                                    reporter.on_text_delta(chunk)

                        elif etype == "content_block_stop":
                            in_tool = False
                            current_tool_name = None

                    elif isinstance(m, AssistantMessage):
                        for b in m.content:
                            if isinstance(b, TextBlock) and b.text.strip():
                                reporter.on_text(b.text.strip())
                            elif isinstance(b, ToolUseBlock):
                                inp = b.input or {}
                                cmd = inp.get("command", "") if b.name == "Bash" else ""
                                if b.name == "Bash" and cmd:
                                    loop_log.debug("Bash cmd: %s", cmd[:120])
                                if "uv run" in cmd and "train.py" in cmd and ">" in cmd:
                                    loop_log.info("Training detected: %s", cmd[:120])
                                    reporter.on_training_detected()
                                else:
                                    label = _tool_label(b.name, inp, cmd)
                                    reporter.on_tool_use(b.name, label)

                    elif isinstance(m, ResultMessage):
                        cost = m.total_cost_usd or 0
                        total_cost += cost
                        reporter.on_round_done(cost, total_cost)
                        break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                reporter.on_round_failed(f"{type(e).__name__}: {e}")

    elapsed = (time.time() - start) / 60
    reporter.on_finished(num_runs, elapsed, total_cost)


# ── CLI entry point ───────────────────────────────────────────────────────────

def dry_run():
    """Show current state without starting the loop."""
    import subprocess
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
