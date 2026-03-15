# Experiment Queue

## FIRST: Re-baseline (MANDATORY)

Do NOT skip this. Run `train.py` AS-IS with NO modifications. We need a fresh val_bpb number to confirm the 1.277 score is still valid. Commit as "baseline: re-verify best config", run training, log the result. Do NOT propose or analyze experiments until this run is complete.

## Then: Suggested experiments (from upstream discussion-43)

Test **one per run**, in this order:

1. **Init scale 0.68**: Multiply the transformer weight init scale by 0.68 (i.e. `scale = 3**0.5 * n_embd**-0.5 * 0.68`). Narrow optimum — 0.66 and 0.70 both tested worse upstream.
2. **x0_init 0.05**: Reduce x0 skip scalar init from 0.1 to 0.05 (in `init_weights`, change `mx.full(..., 0.1, ...)` to 0.05).
3. **Short window seq_len/8**: Change `short_window = long_window // 2` to `long_window // 8` (256 tokens instead of 1024). Less attention compute = faster steps = more tokens.
4. **Embedding weight decay**: Add weight decay to lm_head (0.01), wte embeddings (0.001), and value embeddings (0.003). Currently all 0.0. Modify the optimizer to pass per-group weight decay.
5. **RoPE base 200K**: Increase from 10K to 200K.

Do NOT change depth, batch size, or aspect ratio. Our depth 6 with GQA is validated as optimal for Apple Silicon.
