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
## Test Configuration
- Model: 12 layers, 768 embed dim, 12 query heads, 3 KV heads
- Batch size: 4, Sequence length: 512
- Warmup: 3 iterations, Timed: 10 iterations
- Window size: 128

## Benchmark Results

| Backend | Device | Configuration | Avg (ms) | Throughput (tok/s) | Parameters |
|---------|--------|---------------|----------|--------------------|------------|
| PyTorch | cuda   | reduce-overhead              | 22.97 | 89142  | 82.1M |
| PyTorch | cuda   | max-autotune                 | 22.97 | 89170  | 82.1M |
| PyTorch | cuda   | max-autotune + precompute    | 27.11 | 75547  | 82.1M |
| PyTorch | cuda   | no-compile                   | 25.16 | 81412  | 82.1M |
| JAX     | cuda   | JIT compiled                 | 8.95  | 228875 | 82.1M |

The latest run shows a `2.57x` PyTorch-to-JAX speedup (best PyTorch config vs. JAX baseline)
