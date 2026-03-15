# Experiment Queue

Test **one per run**, in this order. Skip any that already appear in results.tsv.

## Priority 1 — Small changes, high signal (from modded-nanogpt speedrun research)

1. **Learned attention scale**: Replace the fixed `scale = 1.0 / math.sqrt(self.head_dim)` in `CausalSelfAttention.__call__` with a learnable per-head parameter. Add `self.attn_scale = mx.full((self.n_head,), 1.0 / math.sqrt(self.head_dim))` in `__init__`, then use `scale = self.attn_scale` reshaped for the attention call. The JAX modded-nanogpt port cut required steps nearly in half with this change. Zero compute cost.

2. **Sigmoid skip gates**: Replace linear `resid_lambdas` with sigmoid gates. In `__call__`, change `self.resid_lambdas[i] * x` to `mx.sigmoid(self.resid_lambdas[i]) * x`. Initialize `resid_lambdas` so sigmoid outputs ~0.82 (i.e. init value ~1.5 instead of 1.0). Better gradient flow than raw linear scalars. Proven in modded-nanogpt leaderboard records.

3. **Trapezoidal LR schedule**: Replace the current linear warmdown with a trapezoid: 0% to 5% warmup, 5% to 40% full LR, 40% to 100% linear decay to `FINAL_LR_FRAC`. Sharper transitions reportedly outperform smooth cosine/linear at short training budgets. Only modify `get_lr_multiplier`.

## Priority 2 — Medium complexity

4. **Sparse MoE on upper 2 layers**: Replace the MLP in layers 4 and 5 (the last two before final) with a top-2 routing over 8 small experts (each expert = current MLP width / 4). Add a simple linear router + load balancing aux loss (weight 0.01). Increases effective capacity without increasing active compute per token.

Do NOT change depth, batch size, or aspect ratio. Our depth 6 with GQA is validated as optimal for Apple Silicon.
