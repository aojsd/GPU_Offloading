# GPU Trace Replay — Results

## Environment

- 2x H100 80GB, PCIe 5.0, ~56 GB/s H2D per GPU (no contention: 1.00x with concurrent transfers)
- Full Mixtral-8x7B 32L × 8E (256 total experts, 336 MB each, ~86 GB total)
- System RAM: ~1 TB, no cgroup limit
- Bench script: `scripts/batched_replay.py`, real-batch warmup, full trace replay (single trial)
- Each config tests 4 cache policies × 3 prefetch policies = 12 combinations

## Trace Pipeline

- **Phase 1**: 200 ShareGPT conversations, per-conversation GPU traces with fixed 256-token prefill chunks
- **Phase 2**: Continuous batching simulator, `max_graph_size=512`, no preemption (full-sequence page pre-allocation)
- **Phase 3**: Policy simulation (4 cache × 3 prefetch = 12 policies per cache%)
- **Phase 4**: GPU replay with CUDA event I/O timing (in progress)

## Offloading Replay Results (Single GPU)

Workload: 200 ShareGPT conversations, continuous batching with no preemption
(full-sequence page pre-allocation). Fixed 256-token prefill chunks. 50/60% have
identical step counts (KV budget exceeds peak demand); 70% is slightly constrained;
80% and 85% have reduced KV budgets → smaller batches → more steps.

| Cache% | Cached Experts  | Steps | KV Budget (pages) | Avg Batch Size | Peak Batch Size |
|--------|-----|-------|-------------------|----------------|-----------------|
| 50%    | 128 | 4246  | 16,644            | 21.4           | 173             |
| 60%    | 153 | 4246  | 12,444            | 21.4           | 173             |
| 70%    | 179 | 4246  | 8,076             | 21.4           | 163             |
| 80%    | 204 | 4742  | 3,876             | 19.1           | 81              |
| 85%    | 217 | 5956  | 1,692             | 15.2           | 44              |

### GPU Replay: Wall-Clock Timing (All Policies)

_Pending re-run with fixed batched replay (B1-B6 fixes applied). Previous results
used BS=1 replay with eager routing — invalid for I/O-to-compute ratio analysis._

### Transfer Counts (from policy simulation)

Format: **demand / prefetch / total**

| Cache% | LRU-None | LRU-Oracle | LRU-O(1) | LFU-None | LFU-Oracle | LFU-O(1) |
|--------|----------|------------|----------|----------|------------|----------|
| 50%    | 623,771 / 0 / 623,771 | 0 / 623,771 / 623,771 | 491,807 / 131,964 / 623,771 | 452,983 / 0 / 452,983 | 0 / 458,237 / 458,237 | 328,355 / 126,347 / 454,702 |
| 60%    | 521,264 / 0 / 521,264 | 0 / 521,264 / 521,264 | 393,028 / 128,236 / 521,264 | 405,079 / 0 / 405,079 | 0 / 409,671 / 409,671 | 288,671 / 117,998 / 406,669 |
| 70%    | 487,949 / 0 / 487,949 | 0 / 487,949 / 487,949 | 369,453 / 118,496 / 487,949 | 368,191 / 0 / 368,191 | 0 / 371,424 / 371,424 | 263,187 / 106,308 / 369,495 |
| 80%    | 578,375 / 0 / 578,375 | 0 / 578,375 / 578,375 | 461,170 / 117,205 / 578,375 | 458,048 / 0 / 458,048 | 0 / 460,942 / 460,942 | 355,268 / 103,930 / 459,198 |
| 85%    | 1,071,932 / 0 / 1,071,932 | 0 / 1,071,932 / 1,071,932 | 910,130 / 161,802 / 1,071,932 | 284,968 / 0 / 284,968 | 0 / 296,618 / 296,618 | 153,450 / 134,214 / 287,664 |

| Cache% | Belady-None | Belady-Oracle | Belady-O(1) | SF-None | SF-Oracle | SF-O(1) |
|--------|-------------|---------------|-------------|---------|-----------|---------|
| 50%    | 258,340 / 0 / 258,340 | 0 / 265,290 / 265,290 | 155,574 / 104,884 / 260,458 | 330,384 / 0 / 330,384 | 0 / 342,082 / 342,082 | 207,985 / 125,155 / 333,140 |
| 60%    | 195,762 / 0 / 195,762 | 0 / 200,895 / 200,895 | 106,377 / 91,171 / 197,548 | 261,660 / 0 / 261,660 | 0 / 270,914 / 270,914 | 147,232 / 117,000 / 264,232 |
| 70%    | 141,089 / 0 / 141,089 | 0 / 144,813 / 144,813 | 65,548 / 77,145 / 142,693 | 194,157 / 0 / 194,157 | 0 / 201,676 / 201,676 | 92,138 / 104,512 / 196,650 |
| 80%    | 120,322 / 0 / 120,322 | 0 / 123,691 / 123,691 | 43,261 / 78,968 / 122,229 | 158,768 / 0 / 158,768 | 0 / 163,897 / 163,897 | 60,661 / 99,143 / 159,804 |
| 85%    | 158,602 / 0 / 158,602 | 0 / 163,522 / 163,522 | 40,811 / 121,190 / 162,001 | 197,864 / 0 / 197,864 | 0 / 210,355 / 210,355 | 82,630 / 119,494 / 202,124 |

### Transfers Per Step (total transfers / steps)

| Cache% | Steps | LRU-None | LFU-None | Belady-None | SF-None | LFU-Oracle | Belady-O(1) |
|--------|-------|----------|----------|-------------|---------|------------|-------------|
| 50%    | 4246  | 146.9    | 106.7    | 60.8        | 77.8    | 107.9      | 61.3        |
| 60%    | 4246  | 122.7    | 95.4     | 46.1        | 61.6    | 96.5       | 46.5        |
| 70%    | 4246  | 114.9    | 86.7     | 33.2        | 45.7    | 87.5       | 33.6        |
| 80%    | 4742  | 122.0    | 96.6     | 25.4        | 33.5    | 97.2       | 25.8        |
| 85%    | 5956  | 179.9    | 47.8     | 26.6        | 33.2    | 49.8       | 27.2        |

### Key Observations

1. **Belady dominates**: At every cache%, Belady has the lowest total transfers by
   a large margin. At 80%, Belady-None needs 120K transfers vs LRU-None's 578K — 4.8x
   fewer. This gap widens at higher cache% where Belady can better exploit future knowledge.

2. **LRU is prefetch-invariant**: LRU-None/Oracle/O(1) always have identical total
   transfers (e.g., all 623,771 at 50%). Oracle only changes when transfers happen (async
   vs blocking), not how many. This is the expected behavior for LRU.

3. **LFU-Oracle slightly increases total transfers**: At 50%, LFU-Oracle has 458,237
   total vs LFU-None's 452,983 (1.2% more). Oracle prefetch adds eviction churn that
   slightly outweighs the demand-load savings. The effect is consistent across cache%.

4. **StaticFreq Oracle > StaticFreq None by 3-6%**: Oracle prefetch causes more total
   transfers than NoPrefetch with StaticFreq, because prefetch evicts high-frequency
   non-static experts that would otherwise persist across steps. This is a known property
   of the interaction between global-frequency eviction and prefetch-induced churn
   (validated by StaticScratchpad experiments in earlier trace format).

5. **LRU pathology at 85%**: LRU-None needs 1.07M transfers at 85% (217 cache slots)
   vs only 578K at 80% (204 slots). With more cache and fewer KV pages, requests are
   served more sequentially (fewer concurrent, longer-lived), causing LRU's recency
   tracking to thrash. LFU and Belady scale smoothly.

6. **Belady-O(1) ≈ Belady-None**: Throttling prefetch to 1 per layer produces nearly
   the same total transfers as no prefetch (within 1-3%). This suggests the optimal
   prefetch strategy with Belady eviction is minimal — most benefit comes from the
   eviction policy itself.

7. **Policy ranking** (total transfers at 50%): Belady-None (258K) < Belady-O(1) (260K)
   < SF-None (330K) < SF-O(1) (333K) < LFU-None (453K) < LFU-O(1) (455K) < LFU-Oracle (458K) < LRU (624K).

## Compute vs Offloading Comparison
GPU replay with new traces pending — will update throughput comparison once completed.
