"""Expert Offload Engine for MoE expert offloading.

Supports two modes depending on how MoEEngine was initialized:

1. **Simulated offloading** (experts_per_layer=None, 20L Mixtral):
   All expert weights stay on GPU at full [E, 2*I, H]. Residency is tracked
   in software only. Demand loading copies from CPU pinned memory into the
   existing GPU tensor slots. No expert_map remapping needed.

2. **True offloading** (experts_per_layer=K, full 32L Mixtral):
   GPU has a unified buffer holding all layers' resident experts at fixed
   slots: layer l's experts occupy slots [l*experts_per_layer, ..., (l+1)*experts_per_layer-1].
   Scratchpad slots at the end (scratchpad_start..scratchpad_start+7)
   hold demand-loaded non-resident experts. Before each layer's stage4b,
   process_layer() copies expert_map_abs[layer] into expert_map_buf
   and demand-loads any missing experts into scratchpad. This is the
   single call that guarantees all selected experts have valid weights.
"""

import json

import torch


class ExpertOffloadEngine:
    """Expert offload engine with scratchpad-based demand loading.

    Sits between CUDA graph stage4a (router) and stage4b (fused_experts).
    Reads routing decisions, loads missing experts from CPU pinned memory
    into the scratchpad portion of the unified buffer, records a trace of
    all expert activations and transfer times.

    Created automatically by MoEEngine when experts_per_layer is set.
    Step counting is automatic — begin_step() is called by the engine
    at the start of each piecewise decode step.

    Usage:
        engine = MoEEngine(model_path, experts_per_layer=2)
        # engine.offload_engine is auto-created

        # Capture graphs, then decode:
        for step in range(num_steps):
            logits = engine.mixed_step(...)  # begin_step() called internally

        engine.offload_engine.save_trace("trace.json")
    """

    def __init__(self, engine):
        """Create offload engine from a loaded MoEEngine.

        Detects whether engine uses true offloading (experts_per_layer set)
        or simulated offloading (full GPU tensors). Sets up CPU pinned
        copies accordingly.

        Args:
            engine: MoEEngine instance with loaded weights
        """
        self.num_layers = engine.num_layers
        self.num_experts = engine.num_experts
        self.device = engine.device
        self.true_offloading = engine.experts_per_layer is not None

        if self.true_offloading:
            self.w1_buf = engine.w1_buf         # unified GPU buffer
            self.w2_buf = engine.w2_buf         # unified GPU buffer
            self.w1_cpu = engine.w1_cpu         # list of [E, 2*I, H] CPU pinned
            self.w2_cpu = engine.w2_cpu         # list of [E, H, I] CPU pinned
            self.expert_map = engine.expert_map          # relative maps
            self.expert_map_abs = engine.expert_map_abs  # absolute maps
            self.expert_map_buf = engine.expert_map_buf  # [E] int32 GPU
            self.experts_per_layer = engine.experts_per_layer
            self.scratchpad_start = engine.scratchpad_start
            self.scratchpad_slots = engine.scratchpad_slots

            print(f"ExpertOffloadEngine (true): "
                  f"{self.experts_per_layer} experts_per_layer, "
                  f"{self.scratchpad_slots} scratchpad slots, "
                  f"{self.num_layers} layers")
        else:
            # Simulated offloading: full GPU tensors, CPU copies made here
            self.w1_gpu = engine.w1   # list of [E, 2*I, H] GPU tensors
            self.w2_gpu = engine.w2   # list of [E, H, I] GPU tensors
            print("ExpertOffloadEngine (simulated): copying expert weights "
                  "to CPU pinned memory...")
            self.w1_cpu = []
            self.w2_cpu = []
            for l in range(self.num_layers):
                self.w1_cpu.append(self.w1_gpu[l].cpu().pin_memory())
                self.w2_cpu.append(self.w2_gpu[l].cpu().pin_memory())
            print(f"  {self.num_layers} layers x {self.num_experts} experts "
                  f"({self.w1_cpu[0].shape}, {self.w2_cpu[0].shape})")

        # Residency per layer: set of expert IDs in persistent cache on GPU
        self.resident = [set() for _ in range(self.num_layers)]
        self._init_residency()

        # Trace and transfer logs
        self.trace = []
        self.transfers = []
        self._current_step = -1  # first begin_step() gives 0
        self._pending_evict = {}  # layer -> set of experts to evict (simulated only)

        # CUDA events for precise transfer timing
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)

    def _init_residency(self):
        """Initialize residency from current state."""
        if self.true_offloading:
            for l in range(self.num_layers):
                emap = self.expert_map[l].cpu().tolist()
                self.resident[l] = {eid for eid, slot in enumerate(emap)
                                    if slot >= 0}
        else:
            # Simulated: all experts start as resident
            for l in range(self.num_layers):
                self.resident[l] = set(range(self.num_experts))

    def configure(self, gpu_budget_per_layer, initial_experts=None):
        """Set which experts are resident in the unified buffer.

        For simulated mode: just updates residency tracking.
        For true mode: loads initial experts into unified buffer at fixed
        slots and updates both relative and absolute expert_maps.

        Args:
            gpu_budget_per_layer: number of experts to keep resident per layer
            initial_experts: optional list of expert IDs to keep resident
                (default: [0, ..., budget-1])
        """
        if gpu_budget_per_layer >= self.num_experts:
            print(f"ExpertOffloadEngine: budget={gpu_budget_per_layer} >= "
                  f"{self.num_experts} experts, all resident (no offloading)")
            if self.true_offloading:
                experts_per_layer = self.experts_per_layer
                if gpu_budget_per_layer > experts_per_layer:
                    print(f"  (capped to experts_per_layer="
                          f"{experts_per_layer})")
                    gpu_budget_per_layer = experts_per_layer
                    initial_experts = list(range(experts_per_layer))
                else:
                    # All cache slots filled
                    for l in range(self.num_layers):
                        base = l * experts_per_layer
                        self.expert_map[l].fill_(-1)
                        self.expert_map_abs[l].fill_(-1)
                        for slot in range(experts_per_layer):
                            self.w1_buf[base + slot].copy_(self.w1_cpu[l][slot])
                            self.w2_buf[base + slot].copy_(self.w2_cpu[l][slot])
                            self.expert_map[l][slot] = slot
                            self.expert_map_abs[l][slot] = base + slot
                    self.resident = [set(range(experts_per_layer))
                                     for _ in range(self.num_layers)]
                    return
            else:
                self.resident = [set(range(self.num_experts))
                                 for _ in range(self.num_layers)]
                return

        if initial_experts is None:
            initial_experts = list(range(gpu_budget_per_layer))
        assert len(initial_experts) == gpu_budget_per_layer

        if self.true_offloading:
            experts_per_layer = self.experts_per_layer
            assert gpu_budget_per_layer <= experts_per_layer, \
                f"budget {gpu_budget_per_layer} > experts_per_layer " \
                f"{experts_per_layer}"
            # Load initial experts into unified buffer at fixed slots
            for l in range(self.num_layers):
                base = l * experts_per_layer
                self.expert_map[l].fill_(-1)
                self.expert_map_abs[l].fill_(-1)
                for slot, eid in enumerate(initial_experts):
                    self.w1_buf[base + slot].copy_(self.w1_cpu[l][eid])
                    self.w2_buf[base + slot].copy_(self.w2_cpu[l][eid])
                    self.expert_map[l][eid] = slot          # relative
                    self.expert_map_abs[l][eid] = base + slot  # absolute

        for l in range(self.num_layers):
            self.resident[l] = set(initial_experts)

        offloaded = self.num_experts - gpu_budget_per_layer
        print(f"ExpertOffloadEngine: {gpu_budget_per_layer} resident, "
              f"{offloaded} offloaded per layer "
              f"(initial: {initial_experts})")

    def begin_step(self):
        """Mark the start of a new inference step.

        Called automatically by MoEEngine at the start of each piecewise
        decode step. Increments the internal step counter for trace recording.
        """
        self._current_step += 1

    def process_layer(self, layer, topk_ids_buf, n_tokens):
        """Prepare a layer for stage4b: update expert_map and load missing experts.

        This is the ONLY call needed between stage4a and stage4b. It:
        1. Copies expert_map_abs[layer] → expert_map_buf (for true offloading)
        2. Reads routing decisions from topk_ids_buf
        3. Demand-loads any missing experts from CPU → GPU scratchpad
        4. Updates expert_map_buf with scratchpad slot indices

        Args:
            layer: layer index
            topk_ids_buf: [N_padded, top_k] int64 tensor on GPU
            n_tokens: actual number of tokens (not padding)
        """
        # Step 1: set expert_map_buf to this layer's absolute map
        if self.true_offloading:
            self.expert_map_buf.copy_(self.expert_map_abs[layer])

        # Step 2: read routing decisions (GPU -> CPU)
        topk_ids = topk_ids_buf[:n_tokens].cpu()
        needed = set(topk_ids.flatten().unique().tolist())

        # Record trace
        self.trace.append({
            'step': self._current_step,
            'layer': layer,
            'expert_ids': sorted(needed),
        })

        # Load missing experts (before stage4b graph replay)
        missing = needed - self.resident[layer]
        if missing:
            torch.cuda.synchronize()
            if self.true_offloading:
                self._load_into_scratchpad(layer, missing)
            else:
                # Simulated: load into GPU tensor, evict in post_layer
                self._pending_evict[layer] = missing
                self._load_missing_simulated(layer, missing)

    def post_layer(self, layer):
        """Called after stage4b graph replay.

        For true offloading: no-op (scratchpad is ephemeral).
        For simulated: evicts demand-loaded experts from residency tracking.
        """
        if self.true_offloading:
            return  # scratchpad contents don't affect next layer

        missing = self._pending_evict.get(layer, set())
        if missing:
            for eid in missing:
                self.resident[layer].discard(eid)
            del self._pending_evict[layer]

    def _load_into_scratchpad(self, layer, missing):
        """Load missing experts into scratchpad portion of unified buffer.

        Copies from CPU pinned memory into w1_buf[scratchpad_start+i] and
        updates expert_map_buf[eid] = absolute_slot. The scratchpad slots
        are at the end of the unified buffer, shared across all layers.

        Args:
            layer: layer index (for CPU weight indexing)
            missing: set of expert IDs not in persistent cache
        """
        for i, eid in enumerate(sorted(missing)):
            slot = self.scratchpad_start + i  # absolute slot in unified buffer
            self._start_event.record()
            self.w1_buf[slot].copy_(self.w1_cpu[layer][eid],
                                     non_blocking=True)
            self.w2_buf[slot].copy_(self.w2_cpu[layer][eid],
                                     non_blocking=True)
            self._end_event.record()
            self._end_event.synchronize()

            # Update shared expert_map buffer with absolute slot index
            self.expert_map_buf[eid] = slot

            transfer_bytes = (self.w1_cpu[layer][eid].numel() *
                              self.w1_cpu[layer][eid].element_size() +
                              self.w2_cpu[layer][eid].numel() *
                              self.w2_cpu[layer][eid].element_size())
            elapsed_ms = self._start_event.elapsed_time(self._end_event)
            self.transfers.append({
                'step': self._current_step,
                'layer': layer,
                'expert_id': eid,
                'bytes': transfer_bytes,
                'time_ms': elapsed_ms,
            })

    def _load_missing_simulated(self, layer, missing):
        """Load missing experts in simulated mode (direct GPU tensor indexing)."""
        for eid in sorted(missing):
            self._start_event.record()
            self.w1_gpu[layer][eid].copy_(self.w1_cpu[layer][eid],
                                          non_blocking=True)
            self.w2_gpu[layer][eid].copy_(self.w2_cpu[layer][eid],
                                          non_blocking=True)
            self._end_event.record()
            self._end_event.synchronize()

            transfer_bytes = (self.w1_cpu[layer][eid].numel() *
                              self.w1_cpu[layer][eid].element_size() +
                              self.w2_cpu[layer][eid].numel() *
                              self.w2_cpu[layer][eid].element_size())
            elapsed_ms = self._start_event.elapsed_time(self._end_event)
            self.transfers.append({
                'step': self._current_step,
                'layer': layer,
                'expert_id': eid,
                'bytes': transfer_bytes,
                'time_ms': elapsed_ms,
            })
            self.resident[layer].add(eid)

    def save_trace(self, path):
        """Save expert activation trace and transfer log to JSON.

        Args:
            path: output file path
        """
        data = {
            'num_layers': self.num_layers,
            'num_experts': self.num_experts,
            'trace': self.trace,
            'transfers': self.transfers,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"ExpertOffloadEngine: saved trace ({len(self.trace)} entries, "
              f"{len(self.transfers)} transfers) to {path}")

    def get_transfer_stats(self):
        """Compute summary statistics for expert transfers.

        Returns:
            dict with total_transfers, total_bytes, total_time_ms,
            avg_time_ms, bandwidth_gb_s, miss_rate
        """
        if not self.transfers:
            return {
                'total_transfers': 0,
                'total_bytes': 0,
                'total_time_ms': 0.0,
                'avg_time_ms': 0.0,
                'bandwidth_gb_s': 0.0,
                'miss_rate': 0.0,
            }

        total_bytes = sum(t['bytes'] for t in self.transfers)
        total_time = sum(t['time_ms'] for t in self.transfers)
        n_transfers = len(self.transfers)

        # Miss rate: fraction of (layer, expert) lookups that required loading
        total_lookups = sum(len(e['expert_ids']) for e in self.trace)
        miss_rate = n_transfers / total_lookups if total_lookups > 0 else 0.0

        return {
            'total_transfers': n_transfers,
            'total_bytes': total_bytes,
            'total_time_ms': total_time,
            'avg_time_ms': total_time / n_transfers,
            'bandwidth_gb_s': (total_bytes / 1e9) / (total_time / 1e3)
                              if total_time > 0 else 0.0,
            'miss_rate': miss_rate,
        }

    def reset_trace(self):
        """Clear recorded trace and transfer data."""
        self.trace.clear()
        self.transfers.clear()
        self._current_step = -1
