# autoresearch-mlx task runner

# Run a single 5-minute training experiment
train:
    uv run train.py

# Start autonomous experiment loop (default 6h, override with: just loop 8)
loop hours="6":
    cd {{justfile_directory()}} && uv run loop.py --hours {{hours}}

# Prepare data + tokenizer (one-time setup)
prepare:
    uv run prepare.py

# Open analysis notebook
analyze:
    uv sync --extra analysis
    uv run jupyter notebook analysis.ipynb

# Stop a running loop
stop:
    pkill -INT -f "loop.py"
