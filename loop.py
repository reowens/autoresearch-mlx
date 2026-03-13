"""
Autonomous experiment loop. Runs ONE experiment per turn, checks time between turns.

    uv run loop.py          # asks for duration
    uv run loop.py 6        # 6 hours
    uv run loop.py 15       # 15 minutes
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
PROMPT = open(os.path.join(DIR, "program.md")).read()

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
    deadline = time.time() + minutes * 60
    round_num = 0

    opts = ClaudeAgentOptions(
        system_prompt=PROMPT,
        permission_mode="bypassPermissions",
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        cwd=DIR,
        model="opus",
    )

    if minutes >= 60:
        budget = f"{minutes / 60:.4g}h"
    else:
        budget = f"{minutes:.0f}m"
    print(f"\n  Loop started — {budget} budget. Ctrl+C to stop.\n")

    async with ClaudeSDKClient(options=opts) as client:
        while True:
            round_num += 1
            remaining = max(0, (deadline - time.time()) / 60)
            print(f"  === Round {round_num} | {remaining:.0f} min left ===\n")

            msg = MSG_FIRST if round_num == 1 else MSG_NEXT
            await client.query(msg)

            async for m in client.receive_response():
                if isinstance(m, AssistantMessage):
                    for b in m.content:
                        if isinstance(b, TextBlock) and b.text.strip():
                            print(f"  {b.text.strip()[:200]}")
                        elif isinstance(b, ToolUseBlock):
                            cmd = b.input.get("command", "") if b.name == "Bash" else ""
                            if "train.py" in cmd and ">" in cmd:
                                print("  > training (~5 min)...")
                            else:
                                label = b.input.get("description", "") or cmd[:60] or b.name
                                print(f"  [{b.name}] {label}")
                elif isinstance(m, ResultMessage):
                    cost = m.total_cost_usd or 0
                    print(f"  --- round done (${cost:.2f}) ---\n")
                    break

            if round_num > 1 and time.time() >= deadline:
                print("  Time's up.")
                break

    print(f"\n  Done — {round_num} rounds.")
    print_results()


def parse_duration(raw):
    """Parse duration string to minutes. >=10 means minutes, <10 means hours."""
    val = float(raw)
    if val >= 10:
        return val  # minutes
    return val * 60  # hours → minutes


def main():
    if len(sys.argv) > 1:
        minutes = parse_duration(sys.argv[1])
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
