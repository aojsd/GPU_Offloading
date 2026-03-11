# GPU Trace Replay — Results

## Environment

- 1x GH200 480GB (96 GB HBM3), NVLink-C2C, ~309 GB/s H2D
- Full Mixtral-8x7B 32L × 8E (256 total experts, 336 MB each, ~86 GB total)
- Per-expert NVLink-C2C transfer time: 336 MB / 309 GB/s ≈ **1.1 ms**
- System RAM: 480 GB LPDDR5X
- Bench script: `scripts/batched_replay.py`, real-batch warmup, full trace replay (single trial)
- Each config tests 4 cache policies × 3 prefetch policies = 12 combinations

## Trace Pipeline

- **01 — Collect** (`01_collect_traces.sh`): GPU-based batched trace collection with
  continuous batching (greedy admission, LIFO preemption with recompute, dynamic page
  allocation). 200 ShareGPT conversations, epl=5 offloading, one run per cache fraction (70/80/90%).
  Outputs `batched_trace.json` + per-conversation traces.
- **02 — Simulate** (`02_policy_simulate.sh`): CPU-only policy simulation
  (4 cache × 3 prefetch = 12 policies per cache%). Saves `GPUReplayTrace` files.
- **03 — Replay** (`03_gpu_replay.sh`): GPU replay with CUDA event I/O timing.
  Loads precomputed policy traces from step 02, dispatches jobs across GPUs.

## Offloading Replay Results (Single GPU)

Workload: 200 ShareGPT conversations, continuous
batching with greedy admission, LIFO preemption with recompute, dynamic page allocation.
`max_seqs=32`, `max_graph_size=512`, `max_output_tokens=4096`.

| Cache% | Cached Experts | Steps | KV Budget (pages) | Avg Batch | Peak Batch | Preemptions | Avg Experts/Layer |
|--------|---------------|-------|--------------------|-----------|------------|-------------|-------------------|
| 90%    | 230           | 4555  | 5,188             | 21.0      | 32         | 0           | 6.92              |
| 80%    | 204           | 4555  | 9,556             | 21.0      | 32         | 0           | 6.92              |
| 70%    | 179           | 4555  | 13,756             | 21.0      | 32         | 0           | 6.92              |

#### Per-Layer Compute Breakdown (ms, averaged across No-Prefetch runs, 32 layers)

| Cache% | setup | stage1 | attention | stage4a | stage4b | finish | **Total Per-Layer Compute** | **Total Compute (32 Layers)** |
|--------|-------|--------|-----------|---------|---------|--------|-----------------------------|---------------------------------|
| 90%    | 0.028 | 0.072  | 0.057     | 0.039   | 0.754   | 0.003  | **0.952**     | **30.47**       |
| 80%    | 0.028 | 0.072  | 0.056     | 0.039   | 0.757   | 0.003  | **0.956**     | **30.58**       |
| 70%    | 0.028 | 0.072  | 0.056     | 0.039   | 0.757   | 0.003  | **0.956**     | **30.58**       |

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
<!-- 90% cache (230 experts, 4555 steps) -->
<tr class="cache-border">
  <td rowspan="12"><b>90%</b></td>
  <td rowspan="3">Belady</td>
  <td>None</td><td>51.72</td><td>59.2%</td><td>87,280</td><td>0</td><td>87,280</td>
</tr>
<tr><td>Oracle</td><td>40.54</td><td>76.8%</td><td>0</td><td>89,988</td><td>89,988</td></tr>
<tr><td>Oracle(1)</td><td>40.11</td><td>80.4%</td><td>4,769</td><td>84,985</td><td>89,754</td></tr>
<tr class="policy-border">
  <td rowspan="3">StaticFreq</td>
  <td>None</td><td>57.02</td><td>54.0%</td><td>109,987</td><td>0</td><td>109,987</td>
</tr>
<tr><td>Oracle</td><td>45.69</td><td>70.4%</td><td>0</td><td>120,288</td><td>120,288</td></tr>
<tr><td>Oracle(1)</td><td>48.79</td><td>64.7%</td><td>44,568</td><td>66,009</td><td>110,577</td></tr>
<tr class="policy-border">
  <td rowspan="3">LFU</td>
  <td>None</td><td>106.86</td><td>28.9%</td><td>319,849</td><td>0</td><td>319,849</td>
</tr>
<tr><td>Oracle</td><td>94.90</td><td>36.4%</td><td>0</td><td>332,534</td><td>332,534</td></tr>
<tr><td>Oracle(1)</td><td>101.83</td><td>31.3%</td><td>225,260</td><td>96,310</td><td>321,570</td></tr>
<tr class="policy-border">
  <td rowspan="3">LRU</td>
  <td>None</td><td>222.21</td><td>13.9%</td><td>815,788</td><td>0</td><td>815,788</td>
</tr>
<tr><td>Oracle</td><td>206.56</td><td>18.8%</td><td>0</td><td>815,788</td><td>815,788</td></tr>
<tr><td>Oracle(1)</td><td>220.99</td><td>14.5%</td><td>694,795</td><td>120,993</td><td>815,788</td></tr>
<!-- 80% cache (204 experts, 4555 steps) -->
<tr class="cache-border">
  <td rowspan="12"><b>80%</b></td>
  <td rowspan="3">Belady</td>
  <td>None</td><td>73.84</td><td>41.5%</td><td>180,645</td><td>0</td><td>180,645</td>
</tr>
<tr><td>Oracle</td><td>60.80</td><td>52.0%</td><td>0</td><td>186,321</td><td>186,321</td></tr>
<tr><td>Oracle(1)</td><td>65.96</td><td>49.3%</td><td>72,170</td><td>111,652</td><td>183,822</td></tr>
<tr class="policy-border">
  <td rowspan="3">StaticFreq</td>
  <td>None</td><td>81.79</td><td>37.8%</td><td>212,509</td><td>0</td><td>212,509</td>
</tr>
<tr><td>Oracle</td><td>68.49</td><td>47.8%</td><td>0</td><td>223,130</td><td>223,130</td></tr>
<tr><td>Oracle(1)</td><td>74.39</td><td>42.9%</td><td>100,539</td><td>112,657</td><td>213,196</td></tr>
<tr class="policy-border">
  <td rowspan="3">LFU</td>
  <td>None</td><td>175.22</td><td>17.6%</td><td>612,191</td><td>0</td><td>612,191</td>
</tr>
<tr><td>Oracle</td><td>167.11</td><td>22.5%</td><td>0</td><td>644,591</td><td>644,591</td></tr>
<tr><td>Oracle(1)</td><td>173.05</td><td>18.4%</td><td>492,223</td><td>122,779</td><td>615,002</td></tr>
<tr class="policy-border">
  <td rowspan="3">LRU</td>
  <td>None</td><td>237.81</td><td>13.0%</td><td>880,587</td><td>0</td><td>880,587</td>
</tr>
<tr><td>Oracle</td><td>222.24</td><td>17.8%</td><td>0</td><td>880,587</td><td>880,587</td></tr>
<tr><td>Oracle(1)</td><td>237.62</td><td>13.6%</td><td>747,721</td><td>132,866</td><td>880,587</td></tr>
<!-- 70% cache (179 experts, 4555 steps) -->
<tr class="cache-border">
  <td rowspan="12"><b>70%</b></td>
  <td rowspan="3">Belady</td>
  <td>None</td><td>96.43</td><td>32.0%</td><td>274,516</td><td>0</td><td>274,516</td>
</tr>
<tr><td>Oracle</td><td>83.50</td><td>38.6%</td><td>0</td><td>283,240</td><td>283,240</td></tr>
<tr><td>Oracle(1)</td><td>91.13</td><td>35.8%</td><td>157,669</td><td>120,238</td><td>277,907</td></tr>
<tr class="policy-border">
  <td rowspan="3">StaticFreq</td>
  <td>None</td><td>104.59</td><td>29.5%</td><td>309,248</td><td>0</td><td>309,248</td>
</tr>
<tr><td>Oracle</td><td>92.02</td><td>37.6%</td><td>0</td><td>323,439</td><td>323,439</td></tr>
<tr><td>Oracle(1)</td><td>101.31</td><td>31.9%</td><td>176,887</td><td>133,190</td><td>310,077</td></tr>
<tr class="policy-border">
  <td rowspan="3">LFU</td>
  <td>None</td><td>214.24</td><td>14.4%</td><td>778,709</td><td>0</td><td>778,709</td>
</tr>
<tr><td>Oracle</td><td>199.35</td><td>19.4%</td><td>0</td><td>782,888</td><td>782,888</td></tr>
<tr><td>Oracle(1)</td><td>214.47</td><td>15.1%</td><td>646,720</td><td>132,908</td><td>779,628</td></tr>
<tr class="policy-border">
  <td rowspan="3">LRU</td>
  <td>None</td><td>245.11</td><td>12.6%</td><td>914,272</td><td>0</td><td>914,272</td>
</tr>
<tr><td>Oracle</td><td>228.98</td><td>17.3%</td><td>0</td><td>914,272</td><td>914,272</td></tr>
<tr><td>Oracle(1)</td><td>245.19</td><td>13.1%</td><td>777,831</td><td>136,441</td><td>914,272</td></tr>
</tbody>
</table>
