"""Lightweight expert trace recorder for pipeline-parallel mode.

Implements the same begin_step/process_layer/post_layer interface as
ExpertOffloadEngine, but without any weight management. Used when all
experts are GPU-resident (PP mode or full-capacity single GPU) and we
only need to record routing decisions for offline analysis.
"""

import numpy as np
import torch


class TraceRecorder:
    """Records expert routing decisions without managing weights.

    Usage:
        recorder = TraceRecorder(num_layers=32, num_experts=8)
        engine.trace_recorder = recorder

        # Engine calls begin_step/process_layer/post_layer automatically
        for step in range(num_steps):
            logits = engine.step(...)

        trace_data = recorder.trace       # list of {step, layer, expert_ids}
        recorder.reset_trace()            # clear for next conversation
    """

    def __init__(self, num_layers, num_experts, record_router_inputs=False):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.record_router_inputs = record_router_inputs
        self.trace = []
        self._router_inputs = []
        self._step = -1

    def begin_step(self):
        """Mark the start of a new inference step."""
        self._step += 1

    def process_layer(self, layer, topk_ids_buf, n_tokens,
                      router_input_buf=None):
        """Record routing decisions for a layer.

        Args:
            layer: layer index
            topk_ids_buf: [N_padded, top_k] int64 tensor on GPU
            n_tokens: actual number of tokens (not padding)
            router_input_buf: optional [N_padded, hidden_dim] tensor
        """
        ids = topk_ids_buf[:n_tokens].cpu().tolist()
        unique_experts = sorted(set(e for row in ids for e in row))
        self.trace.append({
            'step': self._step,
            'layer': layer,
            'expert_ids': unique_experts,
            'topk_ids': ids,
        })
        if self.record_router_inputs and router_input_buf is not None:
            self._router_inputs.append({
                'step': self._step,
                'layer': layer,
                'hidden': router_input_buf[:n_tokens].float().cpu().numpy(),
            })

    def post_layer(self, layer):
        """Called after stage4b. No-op for trace-only recording."""
        pass

    def reset_trace(self):
        """Clear recorded trace and router inputs."""
        self.trace = []
        self._router_inputs = []
        self._step = -1

    def save_router_inputs(self, path):
        """Save router inputs to compressed .npz file.

        Keys are 'step{s}_layer{l}', values are float32 numpy arrays
        of shape [n_tokens, hidden_dim].
        """
        arrays = {
            f"step{e['step']}_layer{e['layer']}": e['hidden']
            for e in self._router_inputs
        }
        np.savez_compressed(path, **arrays)
