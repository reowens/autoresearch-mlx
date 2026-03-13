"""
Autonomous experiment loop. Runs ONE experiment per turn, checks time between turns.

    uv run loop.py              # asks for duration
    uv run loop.py 15           # 15 minutes
    uv run loop.py 6            # 6 hours
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

MSG_FIRST = (
    "Read results.tsv and train.py. Run exactly ONE experiment: "
    "modify train.py, commit, train, evaluate, update results.tsv, keep or discard. "
    "Stop after this single experiment is complete."
)
MSG_NEXT = (
    "Run exactly ONE more experiment, then stop."
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


async def run(minutes):
    start = time.time()
    deadline = start + minutes * 60
    round_num = 0
    total_cost = 0.0
    model = os.environ.get("MODEL", "opus")

    opts = ClaudeAgentOptions(
        system_prompt=PROMPT,
        permission_mode="bypassPermissions",
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        cwd=DIR,
        model=model,
    )

    if minutes >= 60:
        budget = f"{minutes / 60:.4g}h"
    else:
        budget = f"{minutes:.0f}m"
    print(f"\n  Loop started — {budget} budget. Ctrl+C to stop.\n")

    async with ClaudeSDKClient(options=opts) as client:
        while True:
            round_num += 1
            elapsed = (time.time() - start) / 60
            remaining = max(0, (deadline - time.time()) / 60)
            cost_str = f" | ${total_cost:.2f}" if total_cost > 0 else ""
            print(f"  === Round {round_num} | {elapsed:.0f}m elapsed | {remaining:.0f}m left{cost_str} ===\n")

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

            if round_num > 1 and time.time() >= deadline:
                print("  Time's up.")
                break

    elapsed = (time.time() - start) / 60
    print(f"\n  Done — {round_num} rounds, {elapsed:.0f}m, ${total_cost:.2f}.")
    print_results()


def parse_duration(raw):
    """Parse duration string to minutes. >=10 means minutes, <10 means hours."""
    val = float(raw)
    if val <= 0:
        raise ValueError(f"duration must be positive, got {raw}")
    if val >= 10:
        return val  # minutes
    return val * 60  # hours → minutes


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
            minutes = parse_duration(args[0])
        except ValueError as e:
            sys.exit(f"  Error: {e}")
    else:
        try:
            raw = input("  Duration (single digit = hours, 10+ = minutes) [6h]: ").strip()
            minutes = parse_duration(raw) if raw else 360.0
        except (ValueError, EOFError, KeyboardInterrupt):
            minutes = 360.0

    print()

    try:
        asyncio.run(run(minutes))
    except KeyboardInterrupt:
        print("\n  Stopped.")
        print_results()


if __name__ == "__main__":
    main()
