# autoresearch-mlx task runner

root := justfile_directory()

# Interactive wizard: check data, pick branch, configure, then start
start:
    cd {{root}} && caffeinate -i uv run start.py

# Quick start with N runs (e.g. `just go 30`)
go RUNS="10":
    cd {{root}} && caffeinate -i uv run start.py {{RUNS}}

# Start autonomous experiment loop (e.g. `just loop 10` for 10 experiments)
loop *ARGS:
    cd {{root}} && caffeinate -i uv run loop.py {{ARGS}}

# Run a single 5-minute training experiment
train:
    cd {{root}} && uv run train.py

# Show current state (branch, model, results) without starting
status:
    cd {{root}} && uv run loop.py --dry-run

# Show results scoreboard
results:
    @cd {{root}} && cat results.tsv 2>/dev/null || echo "No results.tsv found"

# Tail dashboard logs live (run in a second terminal while TUI is running)
logs:
    @tail -f {{root}}/dashboard.log

# Prepare data + tokenizer (one-time setup)
prepare:
    cd {{root}} && uv run prepare.py

# Open analysis notebook
analyze:
    cd {{root}} && uv sync --extra analysis && uv run jupyter notebook analysis.ipynb
