import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import jax
import jax.numpy as jnp
from jax import jit, random
import flax.linen as fnn


class BenchmarkConfig:
    def __init__(
        self,
        vocab_size=10000,
        embed_dim=768,
        num_layers=12,
        num_query_heads=12,
        num_kv_heads=3,
        window_size=128,
        batch_size=4,
        seq_len=512,
        warmup_iters=3,
        timed_iters=10,
        compile_mode="max-autotune",
        precompute_masks=False,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.warmup_iters = warmup_iters
        self.timed_iters = timed_iters
        self.compile_mode = compile_mode
        self.precompute_masks = precompute_masks


class BenchmarkResult:
    def __init__(self, backend, device, avg_forward_ms, tokens_per_second, total_params_m):
        self.backend = backend
        self.device = device
        self.avg_forward_ms = avg_forward_ms
        self.tokens_per_second = tokens_per_second
        self.total_params_m = total_params_m

    def pretty(self):
        return (
            f"[{self.backend}] device={self.device} | "
            f"avg={self.avg_forward_ms:.2f} ms | "
            f"throughput={self.tokens_per_second:.0f} tok/s | "
            f"params={self.total_params_m:.1f}M"
        )
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / norm * self.weight


class LightningGQABlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_query_heads,
        num_kv_heads,
        window_size,
        layer_idx,
        cfg,
        dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_query_heads
        self.num_query_groups = num_query_heads // num_kv_heads
        self.use_banded = (layer_idx % 2) == 1
        self.window_size = window_size
        self.cfg = cfg

        self.qkv_proj = nn.Linear(
            embed_dim,
            embed_dim + 2 * self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self._precomputed_mask = None

    def _build_banded_mask(self, seq_len, device):
        positions = torch.arange(seq_len, device=device)
        delta = positions[:, None] - positions[None, :]
        allowed = (delta >= 0) & (delta < self.window_size)
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return mask.masked_fill(allowed, 0.0).clone()

    def set_precomputed_mask(self, mask):
        self._precomputed_mask = mask

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(
            qkv,
            [
                self.embed_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ],
            dim=-1,
        )

        batch_size, seq_len = x.size(0), x.size(1)
        q = q.view(batch_size, seq_len, self.num_query_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.num_query_groups > 1:
            k = k.view(batch_size, self.num_kv_heads, 1, seq_len, self.head_dim)
            k = k.expand(-1, -1, self.num_query_groups, -1, -1)
            k = k.reshape(batch_size, self.num_query_heads, seq_len, self.head_dim).contiguous()

            v = v.view(batch_size, self.num_kv_heads, 1, seq_len, self.head_dim)
            v = v.expand(-1, -1, self.num_query_groups, -1, -1)
            v = v.reshape(batch_size, self.num_query_heads, seq_len, self.head_dim).contiguous()

        attn_mask = None
        if self.use_banded:
            if self.cfg.precompute_masks and self._precomputed_mask is not None:
                attn_mask = self._precomputed_mask
            else:
                attn_mask = self._build_banded_mask(seq_len, x.device)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=not self.use_banded,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.embed_dim)
        x = residual + self.dropout(self.out_proj(attn_output))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class LightningOptimizedGQA_Transformer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(dict(vars(cfg)))
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.blocks = nn.ModuleList(
            LightningGQABlock(
                embed_dim=cfg.embed_dim,
                num_query_heads=cfg.num_query_heads,
                num_kv_heads=cfg.num_kv_heads,
                window_size=cfg.window_size,
                layer_idx=i,
                cfg=cfg,
            )
            for i in range(cfg.num_layers)
        )
        self.ln_f = RMSNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # Apply torch.compile if configured
        if cfg.compile_mode and hasattr(torch, "compile"):
            self.forward = torch.compile(self.forward, mode=cfg.compile_mode)

    def precompute_masks(self, seq_len, device):
        """Precompute masks for all banded layers"""
        for block in self.blocks:
            if block.use_banded:
                mask = block._build_banded_mask(seq_len, device)
                block.set_precomputed_mask(mask)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))



def benchmark_pytorch(cfg, name="pytorch"):
    device = torch.device("cuda")
    model = LightningOptimizedGQA_Transformer(cfg).to(device).eval()

    if cfg.precompute_masks:
        model.precompute_masks(cfg.seq_len, device)

    input_ids = torch.randint(
        0,
        cfg.vocab_size,
        (cfg.batch_size, cfg.seq_len),
        device=device,
    )

    with torch.no_grad():
        for _ in range(cfg.warmup_iters):
            model(input_ids)
        torch.cuda.synchronize()

        start_time = time.perf_counter()
        for _ in range(cfg.timed_iters):
            model(input_ids)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

    avg_time = elapsed / cfg.timed_iters
    total_params = sum(p.numel() for p in model.parameters())
    return BenchmarkResult(
        backend=name,
        device=str(device),
        avg_forward_ms=avg_time * 1_000,
        tokens_per_second=(cfg.batch_size * cfg.seq_len) / avg_time,
        total_params_m=total_params / 1e6,
    )


class JaxRMSNorm(fnn.Module):
    dim: int
    eps: float = 1e-6

    def setup(self):
        self.weight = self.param('weight', lambda rng, shape: jnp.ones(shape), (self.dim,))

    def __call__(self, x):
        norm = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return (x / norm) * self.weight


class JaxGroupedQueryAttention(fnn.Module):
    embed_dim: int
    num_query_heads: int
    num_kv_heads: int
    window_size: int
    use_banded: bool
    dropout_rate: float = 0.1

    def setup(self):
        self.head_dim = self.embed_dim // self.num_query_heads
        self.num_query_groups = self.num_query_heads // self.num_kv_heads
        self.scale = 1.0 / jnp.sqrt(self.head_dim)
        self.qkv_proj = fnn.Dense(
            self.embed_dim + 2 * self.num_kv_heads * self.head_dim,
            use_bias=False
        )
        self.out_proj = fnn.Dense(self.embed_dim, use_bias=False)
        self.attn_dropout = fnn.Dropout(self.dropout_rate)
        self.out_dropout = fnn.Dropout(self.dropout_rate)

    def __call__(self, x, deterministic):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)

        q_size = self.embed_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = jnp.split(qkv, [q_size, q_size + kv_size], axis=-1)

        q = q.reshape(batch_size, seq_len, self.num_query_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q, k, v = [t.transpose(0, 2, 1, 3) for t in (q, k, v)]
        if self.num_query_groups > 1:
            k = jnp.reshape(k, (batch_size, self.num_kv_heads, 1, seq_len, self.head_dim))
            k = jnp.broadcast_to(
                k,
                (batch_size, self.num_kv_heads, self.num_query_groups, seq_len, self.head_dim),
            )
            k = jnp.reshape(k, (batch_size, self.num_query_heads, seq_len, self.head_dim))

            v = jnp.reshape(v, (batch_size, self.num_kv_heads, 1, seq_len, self.head_dim))
            v = jnp.broadcast_to(
                v,
                (batch_size, self.num_kv_heads, self.num_query_groups, seq_len, self.head_dim),
            )
            v = jnp.reshape(v, (batch_size, self.num_query_heads, seq_len, self.head_dim))

        attn_weights = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * self.scale
        positions = jnp.arange(seq_len)
        delta = positions[:, None] - positions[None, :]
        if self.use_banded:
            allowed = (delta >= 0) & (delta < self.window_size)
        else:
            allowed = delta >= 0
        attn_weights = jnp.where(allowed[None, None, :, :], attn_weights, -1e9)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, deterministic=deterministic)
        out = jnp.matmul(attn_weights, v).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)
        return self.out_dropout(out, deterministic=deterministic)


class JaxTransformerBlock(fnn.Module):
    embed_dim: int
    num_query_heads: int
    num_kv_heads: int
    window_size: int
    layer_idx: int
    dropout_rate: float = 0.1

    def setup(self):
        self.use_banded = (self.layer_idx % 2) == 1
        self.attention = JaxGroupedQueryAttention(
            embed_dim=self.embed_dim,
            num_query_heads=self.num_query_heads,
            num_kv_heads=self.num_kv_heads,
            window_size=self.window_size,
            use_banded=self.use_banded,
            dropout_rate=self.dropout_rate,
        )
        self.norm1 = JaxRMSNorm(dim=self.embed_dim)
        self.norm2 = JaxRMSNorm(dim=self.embed_dim)
        self.ffn_dense1 = fnn.Dense(4 * self.embed_dim)
        self.ffn_dense2 = fnn.Dense(self.embed_dim)
        self.ffn_dropout = fnn.Dropout(self.dropout_rate)

    def __call__(self, x, deterministic):
        residual = x
        x = self.norm1(x)
        attn_out = self.attention(x, deterministic)
        x = residual + attn_out
        ffn_input = self.norm2(x)
        ffn_hidden = self.ffn_dense1(ffn_input)
        ffn_hidden = fnn.gelu(ffn_hidden)
        ffn_hidden = self.ffn_dense2(ffn_hidden)
        ffn_hidden = self.ffn_dropout(ffn_hidden, deterministic=deterministic)
        x = x + ffn_hidden
        return x


class JaxOptimizedGQA_Transformer(fnn.Module):
    vocab_size: int
    embed_dim: int
    num_layers: int
    num_query_heads: int
    num_kv_heads: int
    window_size: int
    dropout_rate: float = 0.1

    def setup(self):
        self.token_embedding = fnn.Embed(self.vocab_size, self.embed_dim)
        self.blocks = [
            JaxTransformerBlock(
                embed_dim=self.embed_dim,
                num_query_heads=self.num_query_heads,
                num_kv_heads=self.num_kv_heads,
                window_size=self.window_size,
                layer_idx=i,
                dropout_rate=self.dropout_rate,
            )
            for i in range(self.num_layers)
        ]
        self.ln_f = JaxRMSNorm(dim=self.embed_dim)
        self.lm_head = fnn.Dense(self.vocab_size, use_bias=False)

    def __call__(self, input_ids, deterministic=True):
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x, deterministic=deterministic)
        return self.lm_head(self.ln_f(x))


def benchmark_jax(cfg):
    model = JaxOptimizedGQA_Transformer(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_layers,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        window_size=cfg.window_size,
    )

    key = random.PRNGKey(0)
    dummy_input = jnp.ones((cfg.batch_size, cfg.seq_len), dtype=jnp.int32)
    params = model.init(key, dummy_input)["params"]
    forward = jit(lambda p, x: model.apply({"params": p}, x, deterministic=True))

    input_ids = random.randint(key, dummy_input.shape, 0, cfg.vocab_size)

    for _ in range(cfg.warmup_iters):
        forward(params, input_ids).block_until_ready()

    start_time = time.perf_counter()
    for _ in range(cfg.timed_iters):
        forward(params, input_ids).block_until_ready()
    elapsed = time.perf_counter() - start_time

    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    avg_time = elapsed / cfg.timed_iters
    return BenchmarkResult(
        backend="jax",
        device=str(jax.devices()[0]),
        avg_forward_ms=avg_time * 1_000,
        tokens_per_second=(cfg.batch_size * cfg.seq_len) / avg_time,
        total_params_m=total_params / 1e6,
    )


def main():
    base_cfg = BenchmarkConfig()

    print("Benchmarking Grouped-Query Attention: PyTorch vs JAX")
    print(f"Config: {base_cfg.num_layers} layers, {base_cfg.embed_dim} dim, "
          f"{base_cfg.num_query_heads} query heads, {base_cfg.num_kv_heads} KV heads")
    print(f"Batch size: {base_cfg.batch_size}, Sequence length: {base_cfg.seq_len}\n")

    presets = [
        ("reduce-overhead", BenchmarkConfig(compile_mode="reduce-overhead")),
        ("max-autotune", BenchmarkConfig(compile_mode="max-autotune")),
        ("max-autotune + precompute", BenchmarkConfig(compile_mode="max-autotune", precompute_masks=True)),
        ("no-compile", BenchmarkConfig(compile_mode=None)),
    ]

    results = []

    for name, cfg in presets:
        print(f"Running PyTorch ({name})...")
        results.append(benchmark_pytorch(cfg, name=f"pytorch-{name}"))

    print("Running JAX baseline...")
    jax_result = benchmark_jax(base_cfg)
    results.append(jax_result)

    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)
    for result in results:
        print(result.pretty())
    print("=" * 100)

    best_pytorch = min([r for r in results if r.backend.startswith("pytorch")], key=lambda r: r.avg_forward_ms)
    speedup = best_pytorch.avg_forward_ms / jax_result.avg_forward_ms
    print(f"\nBest PyTorch: {best_pytorch.backend} ({best_pytorch.avg_forward_ms:.2f} ms)")
    print(f"JAX: {jax_result.avg_forward_ms:.2f} ms")


if __name__ == "__main__":
    main()
