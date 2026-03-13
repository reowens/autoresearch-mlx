# autoresearch (Apple Silicon / MLX)

Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) ported to run natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). No PyTorch, no CUDA — just your Mac.

The core idea is unchanged: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up to a log of experiments and (hopefully) a better model.

## Quick start

**Requirements:** Apple Silicon Mac (M1+), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run prepare.py        # one-time data + tokenizer prep
uv run train.py           # single 5-minute training experiment
```

Then point Claude Code (or another coding agent) at `program.md` and let it run the autonomous loop.

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions (the autonomous experiment protocol)
pyproject.toml  — dependencies (MLX, numpy, tiktoken, etc.)
```

## Performance (M4 Max, 5-minute budget)

| | AdamW (old) | MuonAdamW + mx.compile |
|--|-------------|------------------------|
| val_bpb | 1.497 | **1.417** |
| Throughput | ~100K tok/sec | ~138K tok/sec |
| Steps | 1,817 | 2,467 |

5.3% val_bpb improvement with 38% higher throughput, at the same memory footprint (19 GB).

## Differences from upstream

- **MLX instead of PyTorch/CUDA.** Native Apple Silicon training with unified memory.
- **MuonAdamW optimizer.** Full port of upstream's Muon + AdamW, including polar express orthogonalization, NorMuon variance reduction, cautious weight decay, and momentum/weight-decay schedules.
- **mx.compile on forward+backward pass.** ~38% throughput gain via graph fusion. Optimizer runs uncompiled (MLX can't compile custom optimizers with mutation patterns yet).
- **Tuned eval token budget.** 10x shards (~5.2M tokens) — middle ground between upstream's 40x and fast iteration.
- **~6-7 minutes per experiment.** 5 min training + compile/eval overhead.
- **MFU reporting is placeholder.** No Apple Silicon equivalent to the H100 FLOPs reference.

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) — autoresearch and nanochat
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) — original MLX port this fork builds on
- [scasella/nanochat-mlx](https://github.com/scasella/nanochat-mlx) — MLX GPT and optimizer reference
- [awni/picochat](https://github.com/awni/picochat) — MLX training patterns
- [Apple MLX team](https://github.com/ml-explore/mlx)

## License

MIT
