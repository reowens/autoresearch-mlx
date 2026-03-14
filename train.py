"""
Autoresearch pretraining script. Single-device, single-file.
Apple Silicon MLX port of karpathy/autoresearch.
Usage: uv run train.py
"""

import gc
import math
import os
import time
from dataclasses import dataclass
from functools import partial

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, evaluate_bpb, make_dataloader

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def create_additive_causal_mask(seq_len, dtype=mx.float32):
    indices = mx.arange(seq_len)
    blocked = indices[None, :] > indices[:, None]
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype), mx.array(0.0, dtype=dtype))


def create_sliding_window_mask(seq_len, window_size, dtype=mx.float32):
    indices = mx.arange(seq_len)
    causal = indices[None, :] > indices[:, None]
    too_far = (indices[:, None] - indices[None, :]) >= window_size
    blocked = causal | too_far
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype), mx.array(0.0, dtype=dtype))


def get_peak_memory_mb():
    return mx.get_peak_memory() / 1024 / 1024


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=10000)

    def __call__(self, x, ve, mask):
        batch_size, seq_len, _ = x.shape
        q = self.c_q(x).reshape(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)

        if ve is not None and self.ve_gate is not None:
            ve = ve.reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)
            gate = 2 * mx.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + mx.expand_dims(gate, axis=-1) * ve

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = norm(self.rope(q))
        k = norm(self.rope(k))

        scale = 1.0 / math.sqrt(self.head_dim)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = mx.maximum(x, 0) ** 2
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x, ve, mask):
        x = x + self.attn(norm(x), ve, mask)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = [Block(config, i) for i in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = mx.ones((config.n_layer,), dtype=mx.float32)
        self.x0_lambdas = mx.zeros((config.n_layer,), dtype=mx.float32)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = {
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer)
            if has_ve(i, config.n_layer)
        }
        self._mask_cache = {}

    def init_weights(self):
        n_embd = self.config.n_embd
        scale = 3**0.5 * n_embd**-0.5

        self.wte.weight = (mx.random.normal(self.wte.weight.shape) * 1.0).astype(mx.bfloat16)
        self.lm_head.weight = (mx.random.normal(self.lm_head.weight.shape) * 0.001).astype(mx.bfloat16)

        for block in self.blocks:
            block.attn.c_q.weight = mx.random.uniform(-scale, scale, block.attn.c_q.weight.shape).astype(mx.bfloat16)
            block.attn.c_k.weight = mx.random.uniform(-scale, scale, block.attn.c_k.weight.shape).astype(mx.bfloat16)
            block.attn.c_v.weight = mx.random.uniform(-scale, scale, block.attn.c_v.weight.shape).astype(mx.bfloat16)
            block.attn.c_proj.weight = mx.zeros_like(block.attn.c_proj.weight).astype(mx.bfloat16)
            block.mlp.c_fc.weight = mx.random.uniform(-scale, scale, block.mlp.c_fc.weight.shape).astype(mx.bfloat16)
            block.mlp.c_proj.weight = mx.zeros_like(block.mlp.c_proj.weight).astype(mx.bfloat16)
            if block.attn.ve_gate is not None:
                block.attn.ve_gate.weight = mx.zeros_like(block.attn.ve_gate.weight).astype(mx.bfloat16)

        self.resid_lambdas = mx.ones((self.config.n_layer,), dtype=mx.float32)
        self.x0_lambdas = mx.full((self.config.n_layer,), 0.1, dtype=mx.float32)

        for ve in self.value_embeds.values():
            ve.weight = mx.random.uniform(-scale, scale, ve.weight.shape).astype(mx.bfloat16)

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(char in "SL" for char in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": long_window, "S": short_window}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = long_window
        return window_sizes

    def _get_masks(self, seq_len):
        unique_windows = set(self.window_sizes)
        for window_size in unique_windows:
            key = (seq_len, window_size)
            if key not in self._mask_cache:
                if window_size >= seq_len:
                    self._mask_cache[key] = create_additive_causal_mask(seq_len)
                else:
                    self._mask_cache[key] = create_sliding_window_mask(seq_len, window_size)
        return [self._mask_cache[(seq_len, window_size)] for window_size in self.window_sizes]

    def __call__(self, idx, targets=None, reduction="mean"):
        _, seq_len = idx.shape
        masks = self._get_masks(seq_len)

        x = self.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, masks[i])
        x = norm(x)

        logits = self.lm_head(x).astype(mx.float32)
        logits = 15.0 * mx.tanh(logits / 15.0)

        if targets is None:
            return logits

        valid = targets != -1
        targets_safe = mx.where(valid, targets, mx.zeros_like(targets))
        ce = nn.losses.cross_entropy(logits, targets_safe, reduction="none")
        ce = ce * valid
        if reduction == "none":
            return ce
        denom = mx.maximum(mx.sum(valid), 1)
        return mx.sum(ce) / denom


# ---------------------------------------------------------------------------
# Optimizer: MuonAdamW (Muon for 2D matrix params, AdamW for everything else)
# Reference: miolini/autoresearch-macos (PyTorch MPS), ported to pure MLX.
# ---------------------------------------------------------------------------

# Newton iteration coefficients for polar decomposition (orthogonalization)
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


class MuonAdamW:
    """Combined optimizer: Muon for 2D matrix params in blocks, AdamW for others."""

    def __init__(self, model, unembedding_lr, embedding_lr, matrix_lr, weight_decay, adam_betas, scalar_lr):
        model_dim = model.config.n_embd
        dmodel_lr_scale = (model_dim / 768) ** -0.5

        self.adamw_params = {}   # path -> config
        self.muon_groups = {}    # shape -> {paths, lr, ...}
        self.adamw_state = {}    # path -> {m, v, t}
        self.muon_state = {}     # shape -> {momentum_buffer, second_momentum_buffer}

        flat_params = tree_flatten(model.parameters())
        for path, param in flat_params:
            if "blocks" in path and param.ndim == 2:
                # 2D matrix params in transformer blocks → Muon
                shape = param.shape
                if shape not in self.muon_groups:
                    self.muon_groups[shape] = {
                        "paths": [],
                        "lr": matrix_lr,
                        "momentum": 0.95,
                        "ns_steps": 5,
                        "beta2": 0.95,
                        "weight_decay": weight_decay,
                    }
                self.muon_groups[shape]["paths"].append(path)
            else:
                # Everything else → AdamW
                if "wte" in path or "value_embeds" in path:
                    lr, wd, betas = embedding_lr * dmodel_lr_scale, 0.0, adam_betas
                elif "lm_head" in path:
                    lr, wd, betas = unembedding_lr * dmodel_lr_scale, 0.0, adam_betas
                elif "resid_lambdas" in path:
                    lr, wd, betas = scalar_lr * 0.01, 0.0, adam_betas
                elif "x0_lambdas" in path:
                    lr, wd, betas = scalar_lr, 0.0, (0.96, 0.95)
                else:
                    lr, wd, betas = unembedding_lr * dmodel_lr_scale, 0.0, adam_betas

                self.adamw_params[path] = {
                    "lr": lr, "betas": betas, "eps": 1e-10, "weight_decay": wd,
                }

        self.initial_adamw_lrs = {p: c["lr"] for p, c in self.adamw_params.items()}
        self.initial_muon_lrs = {s: g["lr"] for s, g in self.muon_groups.items()}

    def _set_path_value(self, model, path, value):
        parts = path.split(".")
        obj = model
        for part in parts[:-1]:
            if isinstance(obj, list):
                obj = obj[int(part)]
            elif isinstance(obj, dict):
                obj = obj[part]
            else:
                obj = getattr(obj, part)
        last = parts[-1]
        if isinstance(obj, dict):
            obj[last] = value
        else:
            setattr(obj, last, value)

    def _step_adamw(self, path, grad, param, config):
        grad_f32 = grad.astype(mx.float32)
        param_f32 = param.astype(mx.float32)
        lr = config["lr"]
        beta1, beta2 = config["betas"]
        eps = config["eps"]
        wd = config["weight_decay"]

        if path not in self.adamw_state:
            self.adamw_state[path] = {
                "m": mx.zeros_like(grad_f32),
                "v": mx.zeros_like(grad_f32),
                "t": 0,
            }

        state = self.adamw_state[path]
        state["t"] += 1
        state["m"] = beta1 * state["m"] + (1 - beta1) * grad_f32
        state["v"] = beta2 * state["v"] + (1 - beta2) * (grad_f32 * grad_f32)

        bias1 = 1 - beta1 ** state["t"]
        bias2 = 1 - beta2 ** state["t"]
        denom = mx.sqrt(state["v"] / bias2) + eps
        step_size = lr / bias1

        param_f32 = param_f32 * (1 - lr * wd)
        param_f32 = param_f32 - step_size * (state["m"] / denom)
        return param_f32.astype(param.dtype)

    def _step_muon(self, shape, grp, flat_grads, flat_params, muon_momentum, muon_weight_decay):
        paths = grp["paths"]
        lr = grp["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5
        ns_steps = grp["ns_steps"]
        beta2 = grp["beta2"]
        num_params = len(paths)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Stack grads and params (float32 for optimizer math)
        stacked_grads = mx.stack([flat_grads[p].astype(mx.float32) for p in paths])
        stacked_params = mx.stack([flat_params[p].astype(mx.float32) for p in paths])

        # Initialize state if needed
        if shape not in self.muon_state:
            smb_shape = (
                (num_params, shape[-2], 1) if shape[-2] >= shape[-1]
                else (num_params, 1, shape[-1])
            )
            self.muon_state[shape] = {
                "momentum_buffer": mx.zeros((num_params, *shape), dtype=mx.float32),
                "second_momentum_buffer": mx.zeros(smb_shape, dtype=mx.float32),
            }
        state = self.muon_state[shape]

        # --- Nesterov momentum ---
        state["momentum_buffer"] = (
            muon_momentum * state["momentum_buffer"] + (1 - muon_momentum) * stacked_grads
        )
        g = (1 - muon_momentum) * stacked_grads + muon_momentum * state["momentum_buffer"]

        # --- Polar express orthogonalization (bfloat16 for efficiency) ---
        X = g.astype(mx.bfloat16)
        X_norm = mx.sqrt(mx.sum(X * X, axis=(-2, -1), keepdims=True))
        X = X / (X_norm * 1.02 + 1e-6)

        if shape[-2] > shape[-1]:
            for a, b, c in polar_express_coeffs[:ns_steps]:
                A = mx.swapaxes(X, -2, -1) @ X
                B = b * A + c * (A @ A)
                X = a * X + X @ B
        else:
            for a, b, c in polar_express_coeffs[:ns_steps]:
                A = X @ mx.swapaxes(X, -2, -1)
                B = b * A + c * (A @ A)
                X = a * X + B @ X
        g = X

        # --- NorMuon variance reduction (float32) ---
        g_f32 = g.astype(mx.float32)
        v_mean = mx.mean(g_f32 * g_f32, axis=red_dim, keepdims=True)
        red_dim_size = g.shape[red_dim]
        v_norm = mx.sqrt(mx.sum(v_mean, axis=(-2, -1), keepdims=True) * red_dim_size)

        state["second_momentum_buffer"] = (
            beta2 * state["second_momentum_buffer"] + (1 - beta2) * v_mean
        )

        step_size = mx.rsqrt(mx.maximum(state["second_momentum_buffer"], 1e-10))
        scaled_sq_sum = v_mean * red_dim_size * step_size ** 2
        v_norm_new = mx.sqrt(mx.sum(scaled_sq_sum, axis=(-2, -1), keepdims=True))
        final_scale = step_size * (v_norm / mx.maximum(v_norm_new, 1e-10))
        g = g_f32 * final_scale

        # --- Cautious weight decay + parameter update ---
        mask = (g * stacked_params) >= 0
        stacked_params = stacked_params - lr * g - lr * muon_weight_decay * stacked_params * mask

        return stacked_params

    def update(self, model, grads, muon_momentum=0.95, muon_weight_decay=0.0):
        flat_grads = dict(tree_flatten(grads))
        flat_params = dict(tree_flatten(model.parameters()))

        # AdamW params (embeddings, lm_head, scalars)
        for path, config in self.adamw_params.items():
            if path in flat_grads:
                new_param = self._step_adamw(path, flat_grads[path], flat_params[path], config)
                self._set_path_value(model, path, new_param)

        # Muon params (block matrices, grouped by shape for stacked update)
        for shape, grp in self.muon_groups.items():
            stacked_result = self._step_muon(
                shape, grp, flat_grads, flat_params, muon_momentum, muon_weight_decay,
            )
            for i, path in enumerate(grp["paths"]):
                self._set_path_value(model, path, stacked_result[i].astype(flat_params[path].dtype))

    def set_lr_multiplier(self, multiplier):
        for path, config in self.adamw_params.items():
            config["lr"] = self.initial_adamw_lrs[path] * multiplier
        for shape, grp in self.muon_groups.items():
            grp["lr"] = self.initial_muon_lrs[shape] * multiplier

    @property
    def state(self):
        arrays = []
        for s in self.adamw_state.values():
            arrays.extend([s["m"], s["v"]])
        for s in self.muon_state.values():
            arrays.extend([s["momentum_buffer"], s["second_momentum_buffer"]])
        return arrays


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64
HEAD_DIM = 64
WINDOW_PATTERN = "SL"

# Optimization (MuonAdamW: Muon for block matrices, AdamW for embeddings/scalars)
TOTAL_BATCH_SIZE = 2**14
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.05
WARMDOWN_RATIO = 0.3
FINAL_LR_FRAC = 0.0

# Model size
DEPTH = 4
DEVICE_BATCH_SIZE = 8
FINAL_EVAL_BATCH_SIZE = 256
STARTUP_EXCLUDE_STEPS = 10


def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


def get_muon_momentum(step):
    """Momentum warmup: 0.85 → 0.95 over 300 steps."""
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress):
    """Weight decay decays to 0 over training."""
    return WEIGHT_DECAY * (1 - progress)


t_start = time.time()
mx.random.seed(42)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)
t_data = time.time()
print(f"Data/tokenizer loaded in {t_data - t_start:.1f}s")

model_dim = ((DEPTH * ASPECT_RATIO + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
config = GPTConfig(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    n_layer=DEPTH,
    n_head=model_dim // HEAD_DIM,
    n_kv_head=model_dim // HEAD_DIM,
    n_embd=model_dim,
    window_pattern=WINDOW_PATTERN,
)

model = GPT(config)
model.init_weights()
mx.eval(model.parameters())
num_params = sum(param.size for _, param in tree_flatten(model.parameters()))

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = MuonAdamW(
    model,
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
    adam_betas=ADAM_BETAS,
    scalar_lr=SCALAR_LR,
)

loss_grad_fn = nn.value_and_grad(model, lambda model, inputs, targets: model(inputs, targets=targets))

# Compile forward+backward pass for fused ops (MLX 0.31+)
compile_state = [model.state]

@partial(mx.compile, inputs=compile_state, outputs=compile_state)
def compiled_fwd_bwd(x, y):
    return loss_grad_fn(model, x, y)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

smooth_train_loss = 0.0
total_training_time = 0.0
step = 0
t_compiled = None

while True:
    t0 = time.time()
    accum_grads = None
    train_loss = None

    for _ in range(grad_accum_steps):
        loss, grads = compiled_fwd_bwd(x, y)
        mx.eval(loss, grads)
        if t_compiled is None:
            t_compiled = time.time()
            print(f"Model compiled in {t_compiled - t_data:.1f}s")
        train_loss = loss
        if accum_grads is None:
            accum_grads = grads
        else:
            accum_grads = tree_map(lambda lhs, rhs: lhs + rhs, accum_grads, grads)
        x, y, epoch = next(train_loader)

    if grad_accum_steps > 1:
        accum_grads = tree_map(lambda grad: grad * (1.0 / grad_accum_steps), accum_grads)

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    optimizer.set_lr_multiplier(lrm)
    optimizer.update(model, accum_grads, muon_momentum=muon_momentum, muon_weight_decay=muon_weight_decay)
    mx.eval(model.parameters(), *optimizer.state)
    compile_state[0] = model.state  # refresh for next compiled call

    train_loss_f = float(train_loss.item())

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        raise SystemExit(1)

    dt = time.time() - t0
    if step >= STARTUP_EXCLUDE_STEPS:
        total_training_time += dt

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt) if dt > 0 else 0
    remaining = max(0.0, TIME_BUDGET - total_training_time)

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | "
        f"lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
        f"epoch: {epoch} | remaining: {remaining:.0f}s    ",
        end="",
        flush=True,
    )

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1
    if step >= STARTUP_EXCLUDE_STEPS and total_training_time >= TIME_BUDGET:
        break

print()
t_train = time.time()
print(f"Training completed in {t_train - t_compiled:.1f}s")

total_tokens = step * TOTAL_BATCH_SIZE
print("Starting final eval...")
print(f"Final eval batch size: {FINAL_EVAL_BATCH_SIZE}")
val_bpb = evaluate_bpb(model, tokenizer, FINAL_EVAL_BATCH_SIZE)
t_eval = time.time()
print(f"Final eval completed in {t_eval - t_train:.1f}s")

steady_state_mfu = 0.0
peak_vram_mb = get_peak_memory_mb()

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_eval - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
