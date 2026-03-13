# autoresearch-mlx task runner

# Default loop duration
default_hours := "6"

# Run a single 5-minute training experiment
train:
    uv run train.py

# Start autonomous experiment loop (default 6h, override with: just loop 8)
loop hours=default_hours:
    cd {{justfile_directory()}} && timeout {{hours}}h claude -p program.md

# Start multi-agent experiment loop (requires hub)
loop-hub hours=default_hours:
    cd {{justfile_directory()}} && timeout {{hours}}h claude -p program_agenthub.md

# Prepare data + tokenizer (one-time setup)
prepare:
    uv run prepare.py

# Open analysis notebook
analyze:
    uv sync --extra analysis
    uv run jupyter notebook analysis.ipynb
