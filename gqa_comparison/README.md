# GQA Comparison
Benchmark for Grouped-Query Attention (GQA) blocks implemented using PyTorch Lightning and JAX.

Both implementations use identical architectures for fair comparison:
- Same normalization (RMSNorm)
- Same projection style (fused QKV)
- No positional encodings

## Grouped-Query Attention
In grouped-query attention, query heads are bucketed into groups that share key/value projections.
```math
\begin{aligned}
\mathrm{Queries:} & \;\; \mathrm{Q} \in \mathbb{R}^{B \times H_q \times L \times d} \\
\mathrm{Keys/Values:} & \;\; \mathrm{K}, \mathrm{V} \in \mathbb{R}^{B \times H_k \times L \times d} \\
\mathrm{Grouping\;factor:} & \;\; g = \tfrac{H_q}{H_k}
\end{aligned}
```

For each group \( i \):
```math
\mathrm{Attn}_i(Q, K, V) = \mathrm{softmax}\!\left(\tfrac{Q_i K_i^\top}{\sqrt{d}} + M \right)\, V_i
```
where:
* \( Q_i \) are the \( g \) query heads mapped to key/value head \( i \)
* \( M \) encodes the attention mask (causal for even layers, banded for odd layers)

```math
\mathrm{GQA}(Q, K, V) = \mathrm{Proj}\!\left(\mathrm{concat}_i \, \mathrm{Attn}_i(Q, K, V)\right)
```

## Benchmark Results

| Backend | Configuration | Forward Pass | Throughput | Parameters | Memory |
|---------|--------------|-------------|------------|------------|--------|
| **PyTorch** (Best Config) | max-autotune + precomp | 22.58 ms | 90,715 tok/s | 82.1M | 0.74 GB |
| **JAX** | JIT compiled | 7.96 ms | 257,385 tok/s | 89.7M | - |

**Test Configuration:**
- Model: 12 layers, 768 embed dim, 12 query heads, 3 KV heads
- Batch size: 4, Sequence length: 512
- Warmup: 3 iterations, Timed: 10 iterations