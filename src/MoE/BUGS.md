# Known Bugs

Code review conducted 2026-03-07. Updated 2026-03-08 after comprehensive
per-directory subagent reviews and replay fidelity audit.

Scope: bugs are categorized relative to the experiment pipeline (scripts 01-03).

---

## Blocking — FIXED (2026-03-08)

All six blocking bugs fixed. Traces must be re-collected with `torch.compile=True`
and pipeline re-run (scripts 01-03) before results are valid.

### B1. `03_gpu_replay.sh` (was `04_gpu_replay.sh`) dispatched to `run_all_policies.py`: unfaithful BS=1 replay — FIXED

**File:** `scripts/run_all_policies.py:178-179`

GPU replay creates the engine with `max_seqs=1` and replays with a single
decode sequence, ignoring the multi-sequence batch scheduling metadata from
`trace_utils.py`. `03_gpu_replay.sh` dispatches to `run_all_policies.py`,
so the entire experiment pipeline produces unfaithful timing data.

Batch size drives (a) compute kernel duration (attention + MoE scale with BS),
(b) the overlap window available to hide async prefetches behind compute, and
(c) PCIe contention. At BS=1, decode completes in ~9ms, providing almost no
overlap window. Real batches of 30+ sequences take much longer, allowing
prefetches to be hidden entirely. The I/O-to-compute ratio — the key metric
for comparing caching/prefetching policies — is meaningless at BS=1. Expert
cache transfer counts are correct (from policy simulator), but all timing
(total_ms, ms/step, io_ms, compute%) is invalid. `batched_replay.py` performs
faithful multi-sequence replay but is not wired into `03_gpu_replay.sh`.

**Fix:** Wire `03_gpu_replay.sh` to dispatch to `batched_replay.py` instead of
`run_all_policies.py`, or integrate batched replay into `run_all_policies.py`.

### B2. Replay and collection use `use_torch_compile=False` — FIXED

**Files:** `scripts/batched_replay.py:352`, `trace_construction/collect_batched_traces.py`

Both replay and trace collection hardcode `use_torch_compile=False`. Without
compile, Triton fusions for RMSNorm + residual + RoPE are disabled, and each
forward pass has ~32 graph break overheads. This makes compute kernels
meaningfully slower than production (where compile would be enabled), inflating
per-step latency and distorting the I/O-to-compute ratio.

**Why this is safe to fix:** torch.compile + CUDA graphs is the tested production
config (test_split_stage4.py validates it). ReplayController is open-loop — it
uses pre-computed transfer schedules from the policy simulator, not live routing.
Routing divergence from inductor noise does not affect which experts are
loaded/evicted/prefetched. All monkey-patched vLLM ops (`moe_align_block_size`,
`moe_sum`, `topk_softmax`) are fully vectorized and compile-safe. FlashInfer
`plan()` is called outside compiled functions. Stream/event operations in
ReplayController are CPU-side.

**Collection nuance:** Compiled routing produces different expert access patterns
than eager (3.8% vs 80.4% match with vLLM). For faithful performance modeling,
both collection and replay MUST use compile.

**Fix:** Set `use_torch_compile=True` in `batched_replay.py` and
`capture_cuda_graphs`. For collection, decide whether traces should
reflect compiled or eager routing (either is valid, but must be consistent with
the research question).

### B3. Memory budget mismatch causes silent batch shrinkage — FIXED

**File:** `scripts/batched_replay.py:297-306 vs trace_construction/trace_utils.py`

`batched_replay.py` computes graph overhead as `sum(graph_sizes) * 100_000 * 96
* 1.2` ≈ **17 GB** for the compact set, plus 2.5 GB fixed = **19.6 GB** total.
`trace_utils.py` uses only **2.5 GB** fixed overhead. This ~17 GB discrepancy
causes `batched_replay.py` to compute a much lower `max_seqs` than
`trace_utils.py` used. When `max_seqs < trace_peak_seqs`, admit events are
silently skipped (line 151-153: `if not free_seq_ids: continue`), and the
`if rid not in request_to_slot: continue` guard (line 178) silently drops those
requests from each step. The batch becomes smaller than prescribed — fewer
tokens, shorter kernels, more prefetch overlap than intended.

Additionally, the `96` constant (line 301) assumes 32 layers but Mixtral-8x7B-20L
has 20 layers — should be `num_layers * 3`, not hardcoded.

**Fix:** (a) Reconcile memory budgets: either embed `max_seqs` and `kv_budget`
in the batched trace and reuse them, or make both files use the same formula.
(b) Error out when `max_seqs < trace_peak_seqs` instead of silently degrading.

### B4. Padding tokens run full MoE kernel, inflating timing — FIXED

**File:** `moe_engine.py:1766-1768, 2630-2654`

When N_actual < graph_N (padding), stage4a (router) and stage4b (fused_experts)
operate on all graph_N tokens including padding. Attention correctly slices to
N_actual and zeros padding output, but MoE does not. Padding tokens embed
token 0, produce real routing decisions, and route to cached experts — executing
full matrix multiplications that inflate per-step latency. Worst case: 65 real
tokens padded to graph_N=128 → 49% wasted MoE compute. With the compact set
`[1,2,4,8,16,32,64,128,192,256,384,512]`, the largest gap is 128 tokens
(257→384 or 385→512). Fix: zero `topk_weights_buf[N_actual:]` before stage4b
so padding tokens contribute zero and route nowhere, or add intermediate graph
sizes to reduce gaps.

### B5. No graph-coverage validation before replay — FIXED

**File:** `scripts/batched_replay.py:355-373`

Graph capture stops on first OOM (`break`), but the script only checks that
**at least one** graph was captured before starting replay. If OOM occurs at
size 128, all steps needing >=128 tokens will crash mid-replay with
`RuntimeError("No piecewise CUDA graph covers N tokens")`. Fix: after capture,
scan `step_scheduling` for max `total_tokens` and verify a graph covers it.

### B6. Warmup uses single-sequence dummy, not real batches — FIXED

**File:** `scripts/batched_replay.py:104-112`

Warmup runs N steps of `step([0], token, [], [])` — a single decode
sequence with dummy tokens. Real replay uses batches of 30-200+ sequences with
mixed decode/prefill/continuation. This doesn't warm the GPU's L2 cache,
instruction cache, or TLBs for the actual workload shapes, causing first-step
jitter that can spill into measurements. Fix: replay the first N steps of the
actual trace as warmup (untimed).

---

## Not in experiment pipeline (scripts 01-03)

Bugs below affect files outside the scripts 01-03 dependency chain
(`benchmarks/`, `profiling/`, `tests/`, `tests/vLLM_comparison/`), or affect dormant
code paths in pipeline files (unused CLI flags, unused functions).

### `bench_comprehensive.py`: All-resident baseline silently crashes

**File:** `benchmarks/bench_comprehensive.py:173,182`

`is_offloading = experts_per_layer < num_experts` determines whether to add 128
to piecewise graph sizes. But the engine sets `self.offloading = True` whenever
`experts_per_layer` is not None, even when `experts_per_layer == num_experts`.
In the all-resident case, `prefill_to_slot()` routes through `step()`
which needs a piecewise graph >= 128 tokens. Since only `[1, 16, 32]` are
captured, `_find_nearest_piecewise_graph(128)` returns None and `step`
crashes silently. Fix: always include 128 in `piecewise_sizes`.

### `moe_engine.py`: `prefill()` and `generate()` crash in PP mode

**File:** `moe_engine.py:752-769, 1454`

`prefill()` indexes `self.block_table[sid, pg]` with 2D tensor indexing. In PP
mode, `self.block_table` is a list of per-GPU tensors → `TypeError`. Scripts
01-03 use `step()` directly and are unaffected.

### `run_all_policies.py`: `--dual-gpu` merge reads nonexistent files

**File:** `scripts/run_all_policies.py:351-357`

The `--dual-gpu` code path reads `cache{pct}pct/replay_results.json` but
`run_gpu_replay()` saves to `cache{pct}pct-{name}.json`. Merge silently
produces `{}`. Not triggered in normal pipeline.

### `trace_utils.py`: `request_id` mismatch when using `--indices`

**File:** `trace_construction/trace_utils.py`

Admit events use local index as `request_id`; complete events use mapped index.
Mismatch only when `--indices` is passed. Normal pipeline does not use
`--indices`.

### `accuracy_analysis.py`: Cross-prompt logit aggregation misalignment

**File:** `tests/vLLM_comparison/accuracy_analysis.py:722-737`

Flat-list concatenation misaligns at prompt boundaries when engines produce
different token counts.

### `batch_replay.py`: Trace classification assumes single-step prefill

**File:** `tests/vLLM_comparison/batch_replay.py:214-221`

Only first prefill chunk seen with chunked prefill. Harmless for current prompt
lengths (64-256 tokens).

### `bench_offload_prefill_mixed.py`: Missing ctypes libstdc++ preload

**File:** `benchmarks/bench_offload_prefill_mixed.py:1-20`

Missing GCC 13.3 libstdc++ preload needed on RHEL 8.10.

### `test_windowed_reset` has no behavioral assertions

**File:** `tests/test_replay_policy.py:549-560`

Only checks `dm.validate()` passes, never asserts actual eviction behavior.

### Benchmark fairness: vLLM comparison timing asymmetries

**Files:** `tests/vLLM_comparison/microbenchmark.py`, `tests/vLLM_comparison/batch_replay.py`,
`tests/vLLM_comparison/accuracy_analysis.py`

- CUDA events vs `time.perf_counter()` (systematically makes vLLM look slower)
- Single vLLM measurement vs median-of-5 for custom engine
- vLLM `step()` timing includes scheduler/output CPU overhead
- Missing `enable_prefix_caching=False` in decode correctness path
- `KL(A || B)` label is actually `KL(B || A)` per PyTorch convention

### Documentation mismatches

- `tests/README.md`: stale test counts, missing 4 test files
- `scripts/README.md`: missing shell scripts, stale labels
- `profiling/README.md`: missing CLI args for `nsys_pp_decode.py`
- `gpu_replay_trace.py`, `expert_offload_engine.py`: router inputs documented
  as float16 but are float32
- `profiling/`, `benchmarks/` docstrings: stale `python tests/...` paths
- `profile_phases.py`: hardcoded "20 steps, 5 warmup" in generated markdown
- `trace_construction/README.md`: documents `max_seqs=128` but default is 256
- `collect_batched_traces.py`: `output_tokens` vs `output_token_ids` off-by-one
- `tests/vLLM_comparison/README.md`: scripts table omits `accuracy_analysis.py`
- `run_all_policies.py`: comment says "3 dirnames" but does 2
