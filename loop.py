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

DIR = os.path.dirname(os.path.abspath(__file__))

_prompt_path = os.path.join(DIR, "program.md")
if not os.path.exists(_prompt_path):
    sys.exit(f"  Error: {_prompt_path} not found")
with open(_prompt_path) as f:
    PROMPT = f.read()

MINS_PER_RUN = 7  # ~5 min training + ~2 min overhead

MSG_FIRST = (
    "Read results.tsv and train.py. Run exactly ONE experiment: "
    "modify train.py, commit, train, evaluate, update results.tsv, keep or discard. "
    "IMPORTANT: Run only ONE training run. If it crashes or diverges, log it and STOP. "
    "Do NOT revert and try something else — that counts as a second experiment. "
    "Stop after this single experiment is complete."
)
MSG_NEXT = (
    "Run exactly ONE more experiment, then stop. "
    "If it crashes or diverges, log it and stop — do not start another."
)


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


async def run(num_runs):
    start = time.time()
    total_cost = 0.0
    model = os.environ.get("MODEL", "opus")

    opts = ClaudeAgentOptions(
        system_prompt=PROMPT,
        permission_mode="bypassPermissions",
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        cwd=DIR,
        model=model,
    )

    est = num_runs * MINS_PER_RUN
    print(f"\n  Loop started — {num_runs} runs (~{est} min). Ctrl+C to stop.\n")

    async with ClaudeSDKClient(options=opts) as client:
        for round_num in range(1, num_runs + 1):
            elapsed = (time.time() - start) / 60
            cost_str = f" | ${total_cost:.2f}" if total_cost > 0 else ""
            print(f"  === Round {round_num}/{num_runs} | {elapsed:.0f}m elapsed{cost_str} ===\n")

            msg = MSG_FIRST if round_num == 1 else MSG_NEXT
            try:
                await client.query(msg)

                async for m in client.receive_response():
                    if isinstance(m, AssistantMessage):
                        for b in m.content:
                            if isinstance(b, TextBlock) and b.text.strip():
                                print(f"  {b.text.strip()[:200]}")
                            elif isinstance(b, ToolUseBlock):
                                inp = b.input or {}
                                cmd = inp.get("command", "") if b.name == "Bash" else ""
                                if "train.py" in cmd and ">" in cmd:
                                    print("  > training (~5 min)...")
                                else:
                                    label = inp.get("description", "") or cmd[:60] or b.name
                                    print(f"  [{b.name}] {label}")
                    elif isinstance(m, ResultMessage):
                        cost = m.total_cost_usd or 0
                        total_cost += cost
                        print(f"  --- round done (${cost:.2f} | total ${total_cost:.2f}) ---\n")
                        break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"  --- round failed: {type(e).__name__}: {e} ---\n")

    elapsed = (time.time() - start) / 60
    print(f"\n  Done — {num_runs} rounds, {elapsed:.0f}m, ${total_cost:.2f}.")
    print_results()


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
