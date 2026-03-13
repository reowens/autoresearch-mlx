# autoresearch-mlx task runner

set working-directory := justfile_directory()

# Start autonomous experiment loop (e.g. `just loop 15` for 15 min, `just loop 6` for 6h)
loop *ARGS:
    uv run loop.py {{ARGS}}

# Run a single 5-minute training experiment
train:
    uv run train.py

# Show results scoreboard
results:
    @cat results.tsv 2>/dev/null || echo "No results.tsv found"

# Prepare data + tokenizer (one-time setup)
prepare:
    uv run prepare.py

# Open analysis notebook
analyze:
    uv sync --extra analysis
    uv run jupyter notebook analysis.ipynb
