# autoresearch-mlx task runner

# Start autonomous experiment loop (interactive — asks for duration)
# caffeinate prevents macOS sleep while running
loop:
    cd {{justfile_directory()}} && caffeinate -dims uv run loop.py

# Run a single 5-minute training experiment
train:
    cd {{justfile_directory()}} && caffeinate -dims uv run train.py

# Prepare data + tokenizer (one-time setup)
prepare:
    uv run prepare.py

# Open analysis notebook
analyze:
    uv sync --extra analysis
    uv run jupyter notebook analysis.ipynb
