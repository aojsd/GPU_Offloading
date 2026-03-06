# Scripts

Standalone experiment runners.
Run from `src/MoE/`.

## Scripts

### `run_all_policies.py` — Policy simulation & GPU replay sweep

Runs policy simulations and GPU replay for all combinations of cache policies
(LRU, LFU, Belady) x prefetch policies (None, Oracle, Oracle-1-layer) x cache
sizes. Can run CPU-only simulation (fast, reports transfer counts) or full GPU
replay (measures wall-clock timing).

```bash
# CPU-only simulation (transfer counts)
python scripts/run_all_policies.py --sim-only

# Full GPU replay (wall-clock timing)
python scripts/run_all_policies.py --replay

# Custom cache percentages
python scripts/run_all_policies.py --sim-only --cache-pcts 0.3 0.5 0.7 0.9
```
