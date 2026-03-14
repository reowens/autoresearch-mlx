"""Launch the autoresearch TUI (wizard + dashboard).

    uv run start.py         # interactive (wizard or quick-launch)
    uv run start.py 10      # skip wizard, run 10 experiments
"""

from dashboard import main

if __name__ == "__main__":
    main()  # reads sys.argv directly
