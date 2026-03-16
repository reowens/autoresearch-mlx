# autoresearch (Apple Silicon / MLX)

Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) ported to run natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). No PyTorch, no CUDA — just your Mac.

The core idea is unchanged: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5–10 minutes, checks if the result improved, keeps or discards, and repeats. You wake up to a log of experiments and (hopefully) a better model.

## Quick start

**Requirements:** Apple Silicon Mac (M1+), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run prepare.py        # one-time data + tokenizer prep
uv run train.py          # single training experiment
```

To run the autonomous experiment loop (requires Claude Code or API key):

```bash
uv run start.py          # TUI with settings wizard
uv run start.py 10       # skip wizard, run 10 experiments
```

## Results (114 experiments, M4 Pro 20-core GPU)

| Milestone | val_bpb | Key change |
|-----------|---------|------------|
| Baseline | 1.417 | MuonAdamW + mx.compile |
| + GQA | 1.283 | Half KV heads → faster steps |
| + No warmup | 1.277 | Zero-init makes warmup unnecessary |
| + Sigmoid gates | 1.309 | Bounded residual scaling (10 min budget) |
| + Weight decay 0.15 | 1.305 | Synergizes with sigmoid gates |
| **+ Learnable RMSNorm** | **1.304** | nn.RMSNorm with gain at 13 norm sites |

Best config: depth 6, GQA (n_kv_head=2), sigmoid skip gates, learnable RMSNorm, weight decay 0.15, Muon+AdamW with NorMuon variance reduction and cautious weight decay.

## Project structure

```
train.py              — model, optimizer, training loop (agent modifies this)
prepare.py            — data prep, tokenizer, evaluation (do not modify)
program.md            — agent instructions for the autonomous loop
suggestions.md        — prioritized experiment queue for the agent
results.tsv           — full experiment log (114 runs)
loop.py               — outer loop orchestrator (spawns one agent per round)
dashboard.py          — TUI dashboard (live progress, results table)
start.py              — entry point for TUI
justfile              — task runner (just start, just go 30, just logs)
```

## Key features

- **Golden tag system.** `git tag best` always points to the last kept config. The loop restores `train.py` from this tag before each round — no code drift.
- **Structural triage.** Computes effective rank (spectral entropy of weight SVDs) at 60 seconds. Kills degenerate experiments early instead of wasting the full budget.
- **Checkpoint before eval.** Saves model weights before final evaluation so training isn't lost if eval OOMs.
- **TUI dashboard.** Live training progress, thinking indicators, experiment results table, ETA, session cost tracking.
- **caffeinate integration.** `justfile` commands prevent macOS idle sleep during overnight runs.

## Differences from upstream

- **MLX instead of PyTorch/CUDA.** Native Apple Silicon training with unified memory.
- **MuonAdamW optimizer.** Full port including polar express orthogonalization, NorMuon variance reduction, cautious weight decay, and momentum/weight-decay schedules. Float32 Newton-Schulz (bf16 diverges on Apple Silicon).
- **mx.compile on forward+backward pass.** ~38% throughput gain via graph fusion.
- **Learnable RMSNorm.** nn.RMSNorm with gain at pre-attention, pre-MLP, and final norm sites. Bare norm kept for QK normalization and initial embedding norm.
- **Sigmoid skip gates.** `sigmoid(resid_lambda) * x` instead of raw linear scalars — bounded, better gradient flow.

## Analysis

```bash
uv sync --extra analysis
uv run jupyter notebook analysis.ipynb
```

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) — autoresearch and nanochat
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) — original MLX port this fork builds on
- [scasella/nanochat-mlx](https://github.com/scasella/nanochat-mlx) — MLX GPT and optimizer reference
- [awni/picochat](https://github.com/awni/picochat) — MLX training patterns
- [Apple MLX team](https://github.com/ml-explore/mlx)

## License

MIT
