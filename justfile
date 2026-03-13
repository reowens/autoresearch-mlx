# autoresearch-mlx task runner

# Run a single 5-minute training experiment
train:
    uv run train.py

# Start autonomous experiment loop (overnight run)
loop:
    cd {{justfile_directory()}} && claude -p program.md

# Start multi-agent experiment loop (requires hub)
loop-hub:
    cd {{justfile_directory()}} && claude -p program_agenthub.md

# Prepare data + tokenizer (one-time setup)
prepare:
    uv run prepare.py

# Open analysis notebook
analyze:
    uv sync --extra analysis
    uv run jupyter notebook analysis.ipynb
