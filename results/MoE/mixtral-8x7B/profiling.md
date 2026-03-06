# Mixtral-8x7B-20L Per-Phase Kernel Timing

## Environment

- Model: Mixtral-8x7B-20L, single H100 80GB
- Execution: CUDA graph (no compile), piecewise per-layer graphs
- Timing: CUDA events (GPU stream elapsed time, zero observer overhead)
- 20 profiled steps, 5 warmup, averaged across steps and layers
- All values are **per-layer averages** in microseconds (us)

## Computation Phases

| Phase | Decode | Prefill | Contents |
|-------|--------|---------|----------|
| stage1 | yes | yes | RMSNorm + QKV GEMM + RoPE + KV cache write |
| stage2 | yes | - | FlashInfer paged decode attention |
| stage3 | - | yes | Flash Attention v3 causal prefill |
| stage4a | yes | yes | O proj + residual + RMSNorm + router GEMM + softmax + top-k |
| stage4b | yes | yes | fused_moe (Triton) + SiLU + residual add |

## Decode (per-layer avg, us)

1 decode token at the given sequence position.

| Seq Pos | stage1 | stage2 | stage4a | stage4b | Total |
|--------:|-------:|-------:|--------:|--------:|------:|
| 128 | 74.9 | 17.7 | 53.0 | 327.1 | 472.6 |
| 256 | 75.1 | 17.9 | 53.0 | 327.6 | 473.6 |
| 512 | 75.2 | 18.0 | 53.2 | 327.5 | 474.0 |
| 1,024 | 75.0 | 18.4 | 53.4 | 327.5 | 474.3 |
| 2,048 | 75.2 | 20.0 | 53.4 | 327.6 | 476.2 |
| 4,096 | 75.1 | 20.9 | 53.8 | 327.6 | 477.4 |
| 8,192 | 75.2 | 27.6 | 53.8 | 327.7 | 484.4 |
| 16,384 | 75.3 | 41.2 | 54.0 | 327.9 | 498.3 |
| 32,768 | 75.1 | 65.1 | 54.2 | 327.9 | 522.2 |
| 65,536 | 75.2 | 114.6 | 53.9 | 327.7 | 571.4 |
| 131,072 | 75.1 | 208.4 | 53.8 | 327.8 | 665.1 |
| 262,144 | 75.2 | 389.5 | 53.9 | 327.5 | 846.0 |

## Prefill (per-layer avg, us)

1 prefill sequence of the given length, no concurrent decodes.

| Seq Len | stage1 | stage3 | stage4a | stage4b | Total |
|--------:|-------:|-------:|--------:|--------:|------:|
| 128 | 88.9 | 20.0 | 61.3 | 1,126 | 1,296 |
| 256 | 102.4 | 21.7 | 70.8 | 1,252 | 1,447 |
| 512 | 139.8 | 26.4 | 79.5 | 1,252 | 1,498 |
| 1,024 | 252.7 | 54.8 | 116.5 | 1,736 | 2,160 |
| 2,048 | 474.8 | 125.9 | 196.6 | 2,919 | 3,716 |

## Reference: Expert PCIe Transfer

| Metric | Value |
|--------|------:|
| Expert size (BF16) | 336 MB |
| H2D transfer time | 6,314 us |
| PCIe bandwidth | 52.0 GB/s |
