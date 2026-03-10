# GPU Trace Replay — Results

## Environment

- 2x H100 80GB, PCIe 5.0, ~56 GB/s H2D per GPU (no contention: 1.00x with concurrent transfers)
- Full Mixtral-8x7B 32L × 8E (256 total experts, 336 MB each, ~86 GB total)
- System RAM: ~1 TB, no cgroup limit
- Bench script: `scripts/batched_replay.py`, real-batch warmup, full trace replay (single trial)
- Each config tests 4 cache policies × 3 prefetch policies = 12 combinations

## Trace Pipeline

- **01 — Collect** (`01_collect_traces.sh`): GPU-based batched trace collection with
  continuous batching (greedy admission, LIFO preemption with recompute, dynamic page
  allocation). 200 ShareGPT conversations, PP=N, one run per cache fraction (60/70/80%).
  Outputs `batched_trace.json` + per-conversation traces.
- **02 — Simulate** (`02_policy_simulate.sh`): CPU-only policy simulation
  (4 cache × 3 prefetch = 12 policies per cache%). Saves `GPUReplayTrace` files.
- **03 — Replay** (`03_gpu_replay.sh`): GPU replay with CUDA event I/O timing.
  Loads precomputed policy traces from step 02, dispatches jobs across GPUs.

## Offloading Replay Results (Single GPU)

Workload: 200 ShareGPT conversations (198 for 60% due to prompt filter), continuous
batching with greedy admission, LIFO preemption with recompute, dynamic page allocation.
`max_seqs=32`, `max_graph_size=512`, `max_output_tokens=4096`.

| Cache% | Cached Experts | Steps | KV Budget (pages) | Avg Batch Size | Peak Batch Size | Preemptions |
|--------|---------------|-------|--------------------|----------------|-----------------|-------------|
| 60%    | 153           | 4370  | 10,444             | 19.0           | 32              | 0           |
| 70%    | 179           | 4400  | 6,076              | 19.3           | 32              | 0           |
| 80%    | 204           | 4387  | 1,876              | 19.1           | 32              | 111         |

### GPU Replay: Wall-Clock Timing + Transfer Counts

<table>
<style>
  .cache-border td { border-top: 3px solid #333; }
  .policy-border td:nth-child(n+2) { border-top: 1px solid #999; }
</style>
<thead>
<tr>
  <th>Cache%</th><th>Cache Policy</th><th>Prefetch</th>
  <th>ms/step</th><th>Compute%</th>
  <th>Demands</th><th>Prefetches</th><th>Total I/O</th>
</tr>
</thead>
<tbody>
<!-- 60% cache (153 experts, 4370 steps) -->
<tr class="cache-border">
  <td rowspan="12"><b>60%</b></td>
  <td rowspan="3">Belady</td>
  <td>None</td><td>498.25</td><td>7.0%</td><td>318,094</td><td>0</td><td>318,094</td>
</tr>
<tr><td>Oracle</td><td>486.94</td><td>8.6%</td><td>0</td><td>328,219</td><td>328,219</td></tr>
<tr><td>Oracle(1)</td><td>494.69</td><td>8.1%</td><td>213,100</td><td>107,972</td><td>321,072</td></tr>
<tr class="policy-border">
  <td rowspan="3">StaticFreq</td>
  <td>None</td><td>546.92</td><td>6.4%</td><td>351,286</td><td>0</td><td>351,286</td>
</tr>
<tr><td>Oracle</td><td>536.48</td><td>9.5%</td><td>0</td><td>363,396</td><td>363,396</td></tr>
<tr><td>Oracle(1)</td><td>548.16</td><td>7.4%</td><td>233,280</td><td>121,231</td><td>354,511</td></tr>
<tr class="policy-border">
  <td rowspan="3">LFU</td>
  <td>None</td><td>1066.83</td><td>3.3%</td><td>709,105</td><td>0</td><td>709,105</td>
</tr>
<tr><td>Oracle</td><td>1043.43</td><td>6.6%</td><td>0</td><td>712,251</td><td>712,251</td></tr>
<tr><td>Oracle(1)</td><td>1064.19</td><td>3.8%</td><td>587,934</td><td>121,864</td><td>709,798</td></tr>
<tr class="policy-border">
  <td rowspan="3">LRU</td>
  <td>None</td><td></td><td></td><td>800,246</td><td>0</td><td>800,246</td>
</tr>
<tr><td>Oracle</td><td></td><td></td><td>0</td><td>800,246</td><td>800,246</td></tr>
<tr><td>Oracle(1)</td><td></td><td></td><td>674,972</td><td>125,274</td><td>800,246</td></tr>
<!-- 70% cache (179 experts, 4400 steps) -->
<tr class="cache-border">
  <td rowspan="12"><b>70%</b></td>
  <td rowspan="3">Belady</td>
  <td>None</td><td>377.56</td><td>9.3%</td><td>236,516</td><td>0</td><td>236,516</td>
</tr>
<tr><td>Oracle</td><td>363.49</td><td>10.7%</td><td>0</td><td>244,094</td><td>244,094</td></tr>
<tr><td>Oracle(1)</td><td>371.03</td><td>10.9%</td><td>137,258</td><td>102,159</td><td>239,417</td></tr>
<tr class="policy-border">
  <td rowspan="3">StaticFreq</td>
  <td>None</td><td>412.29</td><td>8.6%</td><td>260,320</td><td>0</td><td>260,320</td>
</tr>
<tr><td>Oracle</td><td>401.49</td><td>12.6%</td><td>0</td><td>272,438</td><td>272,438</td></tr>
<tr><td>Oracle(1)</td><td>409.05</td><td>10.0%</td><td>148,974</td><td>114,585</td><td>263,559</td></tr>
<tr class="policy-border">
  <td rowspan="3">LFU</td>
  <td>None</td><td>1023.07</td><td>3.5%</td><td>683,184</td><td>0</td><td>683,184</td>
</tr>
<tr><td>Oracle</td><td>999.02</td><td>6.7%</td><td>0</td><td>686,583</td><td>686,583</td></tr>
<tr><td>Oracle(1)</td><td>1019.40</td><td>4.0%</td><td>568,843</td><td>114,964</td><td>683,807</td></tr>
<tr class="policy-border">
  <td rowspan="3">LRU</td>
  <td>None</td><td></td><td></td><td>787,822</td><td>0</td><td>787,822</td>
</tr>
<tr><td>Oracle</td><td></td><td></td><td>0</td><td>787,822</td><td>787,822</td></tr>
<tr><td>Oracle(1)</td><td></td><td></td><td>669,004</td><td>118,818</td><td>787,822</td></tr>
<!-- 80% cache (204 experts, 4387 steps) -->
<tr class="cache-border">
  <td rowspan="12"><b>80%</b></td>
  <td rowspan="3">Belady</td>
  <td>None</td><td>268.19</td><td>14.2%</td><td>158,474</td><td>0</td><td>158,474</td>
</tr>
<tr><td>Oracle</td><td>251.09</td><td>16.2%</td><td>0</td><td>163,480</td><td>163,480</td></tr>
<tr><td>Oracle(1)</td><td>255.51</td><td>16.9%</td><td>63,684</td><td>97,584</td><td>161,268</td></tr>
<tr class="policy-border">
  <td rowspan="3">StaticFreq</td>
  <td>None</td><td>302.82</td><td>12.5%</td><td>182,442</td><td>0</td><td>182,442</td>
</tr>
<tr><td>Oracle</td><td>291.62</td><td>15.2%</td><td>0</td><td>194,494</td><td>194,494</td></tr>
<tr><td>Oracle(1)</td><td>294.57</td><td>14.8%</td><td>86,580</td><td>99,097</td><td>185,677</td></tr>
<tr class="policy-border">
  <td rowspan="3">LFU</td>
  <td>None</td><td>848.28</td><td>4.5%</td><td>559,018</td><td>0</td><td>559,018</td>
</tr>
<tr><td>Oracle</td><td>851.98</td><td>7.9%</td><td>0</td><td>582,071</td><td>582,071</td></tr>
<tr><td>Oracle(1)</td><td>844.31</td><td>5.1%</td><td>450,637</td><td>110,206</td><td>560,843</td></tr>
<tr class="policy-border">
  <td rowspan="3">LRU</td>
  <td>None</td><td>1153.51</td><td>3.3%</td><td>769,268</td><td>0</td><td>769,268</td>
</tr>
<tr><td>Oracle</td><td>1123.85</td><td>6.7%</td><td>0</td><td>769,268</td><td>769,268</td></tr>
<tr><td>Oracle(1)</td><td>1149.77</td><td>3.8%</td><td>653,339</td><td>115,929</td><td>769,268</td></tr>
</tbody>
</table>
