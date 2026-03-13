"""
Interactive wizard for starting an autoresearch experiment run.

    uv run start.py
"""

import datetime
import os
import subprocess
import sys

DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.expanduser("~/.cache/autoresearch")
RESULTS_PATH = os.path.join(DIR, "results.tsv")


def run(cmd, **kwargs):
    return subprocess.run(cmd, cwd=DIR, text=True, capture_output=True, **kwargs)


def ask(prompt, default=""):
    """Prompt user with a default value. Returns stripped input or default."""
    suffix = f" [{default}]" if default else ""
    try:
        raw = input(f"  {prompt}{suffix}: ").strip()
        return raw if raw else default
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)


def confirm(prompt, default=True):
    """Yes/no prompt."""
    hint = "Y/n" if default else "y/N"
    raw = ask(prompt, hint)
    if raw in ("Y/n", "y/N"):
        return default
    return raw.lower() in ("y", "yes")


def step(label):
    print(f"\n  --- {label} ---\n")


def main():
    print("\n  autoresearch-mlx setup wizard\n")

    # ── 1. Check data ──
    step("Data")
    data_ok = (
        os.path.isdir(os.path.join(CACHE_DIR, "data"))
        and os.path.isfile(os.path.join(CACHE_DIR, "tokenizer", "tokenizer.pkl"))
    )
    if data_ok:
        shards = len([f for f in os.listdir(os.path.join(CACHE_DIR, "data")) if f.endswith(".parquet")])
        print(f"  Found {shards} data shards + tokenizer in {CACHE_DIR}")
    else:
        print(f"  No data found in {CACHE_DIR}")
        if confirm("Run prepare.py to download and process data?"):
            print("  Running prepare.py (this may take a few minutes)...\n")
            result = subprocess.run(
                ["uv", "run", "prepare.py"], cwd=DIR,
            )
            if result.returncode != 0:
                sys.exit("  Error: prepare.py failed")
            print()
        else:
            sys.exit("  Cannot continue without data.")

    # ── 2. Branch ──
    step("Branch")
    current = run(["git", "branch", "--show-current"]).stdout.strip()
    existing = [
        b.strip().removeprefix("autoresearch/")
        for b in run(["git", "branch", "--list", "autoresearch/*"]).stdout.splitlines()
        if b.strip()
    ]

    if current.startswith("autoresearch/"):
        tag = current.removeprefix("autoresearch/")
        print(f"  Already on branch: {current}")
        if not confirm("Continue on this branch?"):
            tag = None
        else:
            tag = current
    else:
        tag = None

    if tag is None:
        today = datetime.date.today().strftime("%b%d").lower()
        # Find an unused tag
        candidate = today
        i = 2
        while candidate in existing:
            candidate = f"{today}-{i}"
            i += 1

        tag = ask("Branch tag", candidate)
        branch = f"autoresearch/{tag}"

        if branch in [f"autoresearch/{e}" for e in existing]:
            print(f"  Branch {branch} already exists.")
            if confirm("Check it out?"):
                run(["git", "checkout", branch])
                print(f"  Switched to {branch}")
            else:
                sys.exit("  Aborted.")
        else:
            run(["git", "checkout", "-b", branch])
            print(f"  Created and switched to {branch}")

    # ── 3. Baseline ──
    step("Baseline")
    has_results = os.path.isfile(RESULTS_PATH) and os.path.getsize(RESULTS_PATH) > 50

    if has_results:
        with open(RESULTS_PATH) as f:
            lines = [l for l in f.readlines() if l.strip()]
        n = len(lines) - 1  # subtract header
        kept = sum(1 for l in lines[1:] if "\tkeep\t" in l)
        print(f"  results.tsv: {n} experiments ({kept} kept)")
    else:
        print("  No results.tsv found — need a baseline run.")
        print("  The first experiment establishes your baseline on this hardware.")
        print("  This will be handled automatically by the loop.\n")

    # ── 4. Number of runs ──
    step("Configuration")
    MINS_PER_RUN = 7
    model = os.environ.get("MODEL", "opus")
    default_runs = "10"
    num_runs = ask(f"Number of experiments (~{MINS_PER_RUN} min each)", default_runs)
    try:
        n = int(num_runs)
        est_min = n * MINS_PER_RUN
        if est_min >= 60:
            est = f"~{est_min / 60:.1f}h"
        else:
            est = f"~{est_min}m"
    except ValueError:
        est = "?"

    # ── 5. Summary + launch ──
    step("Ready")
    branch = run(["git", "branch", "--show-current"]).stdout.strip()
    print(f"  branch: {branch}")
    print(f"  model:  {model}")
    print(f"  runs:   {num_runs} ({est})")
    print()

    if not confirm("Start the experiment loop?"):
        print("  Aborted.")
        return

    print()
    os.environ["MODEL"] = model
    os.execvp("uv", ["uv", "run", "loop.py", num_runs])


if __name__ == "__main__":
    main()
