"""
Autonomous experiment loop runner using Claude Agent SDK.

Usage:
    uv run loop.py              # 6 hours (default)
    uv run loop.py --hours 8    # 8 hours
    uv run loop.py --hours 1    # 1 hour (quick test)

Ctrl+C to stop cleanly between experiments.
"""

import argparse
import asyncio
import signal
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

SYSTEM_PROMPT = open("program.md").read()

FIRST_MESSAGE = """\
Start the experiment loop. The branch and baseline are already set up — \
read results.tsv to see the current state, then begin experimenting. \
Run experiments continuously without asking me anything.\
"""

NEXT_MESSAGE = """\
Continue the experiment loop. Check results.tsv for where we are, \
then run the next experiment. Do not ask me anything — just go.\
"""


def print_status(msg: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}\n")


async def run_loop(hours: float) -> None:
    deadline = time.time() + hours * 3600
    experiment = 0
    stop_requested = False

    def handle_signal(sig, frame):
        nonlocal stop_requested
        stop_requested = True
        print_status("Stop requested — finishing current experiment...")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        permission_mode="bypassPermissions",
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        cwd="/Users/reoiv/Development/beyond/tools/autoresearch-mlx",
    )

    print_status(f"Starting experiment loop — {hours}h budget ({hours * 3600:.0f}s)")

    async with ClaudeSDKClient(options=options) as client:
        while time.time() < deadline and not stop_requested:
            experiment += 1
            remaining = (deadline - time.time()) / 60
            print_status(
                f"Experiment #{experiment} | {remaining:.0f} min remaining"
            )

            prompt = FIRST_MESSAGE if experiment == 1 else NEXT_MESSAGE

            try:
                await client.query(prompt)

                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                # Print Claude's reasoning
                                for line in block.text.strip().split("\n"):
                                    print(f"  {line}")
                            elif isinstance(block, ToolUseBlock):
                                # Show tool use compactly
                                tool_input = str(block.input)
                                if len(tool_input) > 120:
                                    tool_input = tool_input[:117] + "..."
                                print(f"  [{block.name}] {tool_input}")
                    elif isinstance(message, ResultMessage):
                        break

            except Exception as e:
                print(f"\n  ERROR: {type(e).__name__}: {e}")
                print("  Continuing to next experiment...\n")
                continue

            # Check if we have time for another experiment (~8 min buffer)
            if deadline - time.time() < 480:
                print_status("Less than 8 min remaining — stopping")
                break

    elapsed = hours - (deadline - time.time()) / 3600
    print_status(
        f"Done — {experiment} experiments in {elapsed:.1f}h"
    )


def main():
    parser = argparse.ArgumentParser(description="Autonomous experiment loop")
    parser.add_argument(
        "--hours", type=float, default=6, help="Total time budget (default: 6)"
    )
    args = parser.parse_args()
    asyncio.run(run_loop(args.hours))


if __name__ == "__main__":
    main()
