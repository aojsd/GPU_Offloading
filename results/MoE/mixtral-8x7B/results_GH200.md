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
  allocation). 200 ShareGPT conversations, epl=5 offloading, one run per cache fraction (70/75/80/85/90/92.5/95/97.5%).
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
| 97.5%  | 249           | 4871  | 1,996             | 18.6      | 32         | 131         | 6.92              |
| 95%    | 243           | 4555  | 3,004             | 21.0      | 32         | 0           | 6.92              |
| 92.5%  | 236           | 4555  | 4,180             | 21.0      | 32         | 0           | 6.92              |
| 90%    | 230           | 4555  | 5,188             | 21.0      | 32         | 0           | 6.92              |
| 85%    | 217           | 4555  | 7,372             | 21.0      | 32         | 0           | 6.92              |
| 80%    | 204           | 4555  | 9,556             | 21.0      | 32         | 0           | 6.92              |
| 75%    | 192           | 4555  | 11,572             | 21.0      | 32         | 0           | 6.92              |
| 70%    | 179           | 4555  | 13,756             | 21.0      | 32         | 0           | 6.92              |

#### Per-Layer Compute Breakdown (ms, averaged across No-Prefetch runs, 32 layers)

| Cache% | setup | stage1 | attention | stage4a | stage4b | finish | **Total Per-Layer Compute** | **Total Compute (32 Layers)** |
|--------|-------|--------|-----------|---------|---------|--------|-----------------------------|---------------------------------|
| 97.5%  | 0.029 | 0.076  | 0.051     | 0.040   | 0.765   | 0.003  | **0.965**     | **30.87**       |
| 95%    | 0.028 | 0.072  | 0.057     | 0.039   | 0.753   | 0.003  | **0.951**     | **30.44**       |
| 92.5%  | 0.028 | 0.072  | 0.057     | 0.039   | 0.753   | 0.003  | **0.952**     | **30.47**       |
| 90%    | 0.028 | 0.072  | 0.057     | 0.039   | 0.754   | 0.003  | **0.952**     | **30.47**       |
| 85%    | 0.029 | 0.072  | 0.057     | 0.039   | 0.757   | 0.003  | **0.956**     | **30.59**       |
| 80%    | 0.028 | 0.072  | 0.056     | 0.039   | 0.757   | 0.003  | **0.956**     | **30.58**       |
| 75%    | 0.029 | 0.072  | 0.056     | 0.039   | 0.756   | 0.003  | **0.956**     | **30.58**       |
| 70%    | 0.028 | 0.072  | 0.056     | 0.039   | 0.757   | 0.003  | **0.956**     | **30.58**       |

### GPU Replay: Wall-Clock Timing + Transfer Counts

<table>
<style>
  .cache-border td { border-top: 5px solid #fff; }
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
<!-- 97.5% cache (249 experts, 4871 steps) -->
<tr class="cache-border">
  <td rowspan="12"><b>97.5%</b></td>
  <td rowspan="3">Belady</td>
  <td>None</td><td>36.30</td><td>85.7%</td><td>22,026</td><td>0</td><td>22,026</td>
</tr>
<tr><td>Oracle</td><td>32.55</td><td>96.7%</td><td>0</td><td>22,709</td><td>22,709</td></tr>
<tr><td>Oracle(1)</td><td>32.51</td><td>97.0%</td><td>111</td><td>22,591</td><td>22,702</td></tr>
<tr class="policy-border">
  <td rowspan="3">StaticFreq</td>
  <td>None</td><td>38.25</td><td>81.3%</td><td>30,954</td><td>0</td><td>30,954</td>
</tr>
<tr><td>Oracle</td><td>32.94</td><td>96.3%</td><td>0</td><td>34,153</td><td>34,153</td></tr>
<tr><td>Oracle(1)</td><td>33.31</td><td>95.0%</td><td>3,147</td><td>28,017</td><td>31,164</td></tr>
<tr class="policy-border">
  <td rowspan="3">LFU</td>
  <td>None</td><td>45.90</td><td>67.8%</td><td>65,088</td><td>0</td><td>65,088</td>
</tr>
<tr><td>Oracle</td><td>39.44</td><td>82.0%</td><td>0</td><td>69,035</td><td>69,035</td></tr>
<tr><td>Oracle(1)</td><td>41.18</td><td>77.6%</td><td>23,849</td><td>43,724</td><td>67,573</td></tr>
<tr class="policy-border">
  <td rowspan="3">LRU</td>
  <td>None</td><td>172.04</td><td>18.2%</td><td>638,456</td><td>0</td><td>638,456</td>
</tr>
<tr><td>Oracle</td><td>157.41</td><td>23.7%</td><td>0</td><td>638,456</td><td>638,456</td></tr>
<tr><td>Oracle(1)</td><td>171.96</td><td>18.7%</td><td>537,086</td><td>101,370</td><td>638,456</td></tr>
<!-- 95% cache (243 experts, 4555 steps) -->
<tr class="cache-border">
  <td rowspan="12"><b>95%</b></td>
  <td rowspan="3">Belady</td>
  <td>None</td><td>41.08</td><td>74.6%</td><td>42,470</td><td>0</td><td>42,470</td>
</tr>
<tr><td>Oracle</td><td>33.30</td><td>93.4%</td><td>0</td><td>43,781</td><td>43,781</td></tr>
<tr><td>Oracle(1)</td><td>33.14</td><td>94.4%</td><td>691</td><td>43,055</td><td>43,746</td></tr>
<tr class="policy-border">
  <td rowspan="3">StaticFreq</td>
  <td>None</td><td>44.46</td><td>68.9%</td><td>57,717</td><td>0</td><td>57,717</td>
</tr>
<tr><td>Oracle</td><td>34.18</td><td>92.8%</td><td>0</td><td>64,395</td><td>64,395</td></tr>
<tr><td>Oracle(1)</td><td>37.31</td><td>84.1%</td><td>13,630</td><td>44,459</td><td>58,089</td></tr>
<tr class="policy-border">
  <td rowspan="3">LFU</td>
  <td>None</td><td>71.51</td><td>43.0%</td><td>171,733</td><td>0</td><td>171,733</td>
</tr>
<tr><td>Oracle</td><td>61.20</td><td>53.6%</td><td>0</td><td>179,756</td><td>179,756</td></tr>
<tr><td>Oracle(1)</td><td>65.65</td><td>48.1%</td><td>102,083</td><td>71,839</td><td>173,922</td></tr>
<tr class="policy-border">
  <td rowspan="3">LRU</td>
  <td>None</td><td>211.03</td><td>14.6%</td><td>765,851</td><td>0</td><td>765,851</td>
</tr>
<tr><td>Oracle</td><td>196.37</td><td>19.5%</td><td>0</td><td>765,851</td><td>765,851</td></tr>
<tr><td>Oracle(1)</td><td>210.02</td><td>15.2%</td><td>655,310</td><td>110,541</td><td>765,851</td></tr>
<!-- 92.5% cache (236 experts, 4555 steps) -->
<tr class="cache-border">
  <td rowspan="12"><b>92.5%</b></td>
  <td rowspan="3">Belady</td>
  <td>None</td><td>46.76</td><td>65.5%</td><td>66,416</td><td>0</td><td>66,416</td>
</tr>
<tr><td>Oracle</td><td>36.57</td><td>85.5%</td><td>0</td><td>68,472</td><td>68,472</td></tr>
<tr><td>Oracle(1)</td><td>36.00</td><td>88.5%</td><td>2,376</td><td>65,976</td><td>68,352</td></tr>
<tr class="policy-border">
  <td rowspan="3">StaticFreq</td>
  <td>None</td><td>51.57</td><td>59.5%</td><td>87,227</td><td>0</td><td>87,227</td>
</tr>
<tr><td>Oracle</td><td>40.74</td><td>78.6%</td><td>0</td><td>97,347</td><td>97,347</td></tr>
<tr><td>Oracle(1)</td><td>44.00</td><td>71.3%</td><td>33,897</td><td>53,886</td><td>87,783</td></tr>
<tr class="policy-border">
  <td rowspan="3">LFU</td>
  <td>None</td><td>87.55</td><td>35.1%</td><td>239,187</td><td>0</td><td>239,187</td>
</tr>
<tr><td>Oracle</td><td>75.04</td><td>44.7%</td><td>0</td><td>246,029</td><td>246,029</td></tr>
<tr><td>Oracle(1)</td><td>81.79</td><td>38.8%</td><td>156,348</td><td>84,639</td><td>240,987</td></tr>
<tr class="policy-border">
  <td rowspan="3">LRU</td>
  <td>None</td><td>218.94</td><td>14.1%</td><td>797,131</td><td>0</td><td>797,131</td>
</tr>
<tr><td>Oracle</td><td>203.05</td><td>19.1%</td><td>0</td><td>797,131</td><td>797,131</td></tr>
<tr><td>Oracle(1)</td><td>217.21</td><td>14.8%</td><td>680,181</td><td>116,950</td><td>797,131</td></tr>
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
<!-- 85% cache (217 experts, 4555 steps) -->
<tr class="cache-border">
  <td rowspan="12"><b>85%</b></td>
  <td rowspan="3">Belady</td>
  <td>None</td><td>62.93</td><td>48.9%</td><td>133,394</td><td>0</td><td>133,394</td>
</tr>
<tr><td>Oracle</td><td>50.55</td><td>62.6%</td><td>0</td><td>137,573</td><td>137,573</td></tr>
<tr><td>Oracle(1)</td><td>52.87</td><td>62.4%</td><td>30,715</td><td>105,735</td><td>136,450</td></tr>
<tr class="policy-border">
  <td rowspan="3">StaticFreq</td>
  <td>None</td><td>69.04</td><td>44.7%</td><td>159,572</td><td>0</td><td>159,572</td>
</tr>
<tr><td>Oracle</td><td>57.21</td><td>56.6%</td><td>0</td><td>172,811</td><td>172,811</td></tr>
<tr><td>Oracle(1)</td><td>62.00</td><td>51.2%</td><td>72,593</td><td>90,542</td><td>163,135</td></tr>
<tr class="policy-border">
  <td rowspan="3">LFU</td>
  <td>None</td><td>142.15</td><td>21.7%</td><td>469,093</td><td>0</td><td>469,093</td>
</tr>
<tr><td>Oracle</td><td>129.05</td><td>27.7%</td><td>0</td><td>478,705</td><td>478,705</td></tr>
<tr><td>Oracle(1)</td><td>138.21</td><td>23.1%</td><td>360,166</td><td>110,500</td><td>470,666</td></tr>
<tr class="policy-border">
  <td rowspan="3">LRU</td>
  <td>None</td><td>234.26</td><td>13.2%</td><td>860,318</td><td>0</td><td>860,318</td>
</tr>
<tr><td>Oracle</td><td>218.44</td><td>18.0%</td><td>0</td><td>860,318</td><td>860,318</td></tr>
<tr><td>Oracle(1)</td><td>233.76</td><td>13.8%</td><td>730,836</td><td>129,482</td><td>860,318</td></tr>
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
<!-- 75% cache (192 experts, 4555 steps) -->
<tr class="cache-border">
  <td rowspan="12"><b>75%</b></td>
  <td rowspan="3">Belady</td>
  <td>None</td><td>84.53</td><td>36.4%</td><td>225,226</td><td>0</td><td>225,226</td>
</tr>
<tr><td>Oracle</td><td>71.47</td><td>45.0%</td><td>0</td><td>232,344</td><td>232,344</td></tr>
<tr><td>Oracle(1)</td><td>78.11</td><td>41.9%</td><td>112,386</td><td>116,126</td><td>228,512</td></tr>
<tr class="policy-border">
  <td rowspan="3">StaticFreq</td>
  <td>None</td><td>92.47</td><td>33.4%</td><td>258,784</td><td>0</td><td>258,784</td>
</tr>
<tr><td>Oracle</td><td>78.74</td><td>41.6%</td><td>0</td><td>269,459</td><td>269,459</td></tr>
<tr><td>Oracle(1)</td><td>86.72</td><td>36.5%</td><td>132,871</td><td>126,608</td><td>259,479</td></tr>
<tr class="policy-border">
  <td rowspan="3">LFU</td>
  <td>None</td><td>208.37</td><td>14.8%</td><td>751,893</td><td>0</td><td>751,893</td>
</tr>
<tr><td>Oracle</td><td>193.35</td><td>19.8%</td><td>0</td><td>756,926</td><td>756,926</td></tr>
<tr><td>Oracle(1)</td><td>207.04</td><td>15.5%</td><td>624,495</td><td>128,303</td><td>752,798</td></tr>
<tr class="policy-border">
  <td rowspan="3">LRU</td>
  <td>None</td><td>241.53</td><td>12.8%</td><td>893,161</td><td>0</td><td>893,161</td>
</tr>
<tr><td>Oracle</td><td>225.49</td><td>17.6%</td><td>0</td><td>893,161</td><td>893,161</td></tr>
<tr><td>Oracle(1)</td><td>241.30</td><td>13.3%</td><td>758,620</td><td>134,541</td><td>893,161</td></tr>
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
