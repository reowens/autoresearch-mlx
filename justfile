# autoresearch-mlx task runner

# Start autonomous experiment loop (interactive — asks for duration)
# caffeinate runs in background to prevent macOS sleep without corrupting stdin
loop:
    cd {{justfile_directory()}} && caffeinate -dims -w $$ & uv run loop.py; kill %1 2>/dev/null

# Run a single 5-minute training experiment
train:
    cd {{justfile_directory()}} && caffeinate -dims -w $$ & uv run train.py; kill %1 2>/dev/null

# Prepare data + tokenizer (one-time setup)
prepare:
    uv run prepare.py

# Open analysis notebook
analyze:
    uv sync --extra analysis
    uv run jupyter notebook analysis.ipynb
