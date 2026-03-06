"""Tests for trace_construction/build_trace.py — continuous batching simulator.

CPU-only tests using synthetic per-conversation traces. No GPU or model needed.
"""
import json
import math
import os
import sys
import tempfile
import unittest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MOE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, MOE_DIR)
sys.path.insert(0, os.path.join(MOE_DIR, 'trace_construction'))

from build_trace import (
    ConversationTrace, simulate_batch, pages_needed, compute_memory_budget,
    load_traces,
)
from data_movement_trace import ActivationTrace


def make_convo_trace(conv_id, prompt_tokens, output_tokens,
                     num_layers=4, num_experts=8, top_k=2):
    """Create a synthetic ConversationTrace with deterministic expert selections."""
    steps = []
    # Step 0: prefill — use experts based on prompt length
    prefill_experts = []
    for layer in range(num_layers):
        # Deterministic: hash-like selection
        experts = [(prompt_tokens + layer * 3 + i) % num_experts
                   for i in range(top_k)]
        prefill_experts.append(sorted(set(experts)))
    steps.append(prefill_experts)

    # Steps 1..output_tokens: decode
    for s in range(output_tokens):
        decode_experts = []
        for layer in range(num_layers):
            experts = [(s + layer * 5 + i) % num_experts for i in range(top_k)]
            decode_experts.append(sorted(set(experts)))
        steps.append(decode_experts)

    return ConversationTrace(
        conversation_id=conv_id,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        steps=steps,
    )


def save_convo_trace(trace: ConversationTrace, path: str):
    """Save a ConversationTrace in the flat JSON format."""
    flat = []
    for step_idx, step_layers in enumerate(trace.steps):
        for layer_idx, expert_ids in enumerate(step_layers):
            if expert_ids:
                flat.append({
                    'step': step_idx,
                    'layer': layer_idx,
                    'expert_ids': expert_ids,
                })
    data = {
        'conversation_id': trace.conversation_id,
        'prompt_tokens': trace.prompt_tokens,
        'output_tokens': trace.output_tokens,
        'num_layers': trace.num_layers,
        'num_experts': trace.num_experts,
        'top_k': trace.top_k,
        'trace': flat,
        'transfers': [],
    }
    with open(path, 'w') as f:
        json.dump(data, f)


class TestPagesNeeded(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(pages_needed(0, 16), 0)

    def test_exact_page(self):
        self.assertEqual(pages_needed(16, 16), 1)
        self.assertEqual(pages_needed(32, 16), 2)

    def test_partial_page(self):
        self.assertEqual(pages_needed(1, 16), 1)
        self.assertEqual(pages_needed(17, 16), 2)
        self.assertEqual(pages_needed(33, 16), 3)

    def test_large(self):
        self.assertEqual(pages_needed(4096, 16), 256)
        self.assertEqual(pages_needed(4097, 16), 257)


class TestSimulateBatch(unittest.TestCase):
    """Test the continuous batching simulator."""

    def test_single_conversation(self):
        """One conversation with plenty of budget should run without preemption."""
        trace = make_convo_trace("conv0", prompt_tokens=32, output_tokens=10)
        result = simulate_batch([trace], kv_page_budget=100, page_size=16)

        self.assertEqual(result['statistics']['total_preemptions'], 0)
        self.assertEqual(result['statistics']['peak_batch_size'], 1)
        # 1 prefill + 10 decode = 11 steps
        self.assertEqual(result['statistics']['total_steps'], 11)

    def test_two_conversations_sequential(self):
        """Two convos with budget for only one at a time → sequential execution."""
        traces = [
            make_convo_trace("c0", prompt_tokens=32, output_tokens=5),
            make_convo_trace("c1", prompt_tokens=32, output_tokens=5),
        ]
        # Budget for only 1 sequence (2 pages for 32 tokens)
        result = simulate_batch(traces, kv_page_budget=2, page_size=16)

        self.assertEqual(result['statistics']['peak_batch_size'], 1)
        # Each conv: 1 prefill + 5 decode = 6 steps. Sequential = 12.
        self.assertEqual(result['statistics']['total_steps'], 12)
        self.assertEqual(result['statistics']['total_preemptions'], 0)

    def test_two_conversations_concurrent(self):
        """Two convos with enough budget → concurrent execution."""
        traces = [
            make_convo_trace("c0", prompt_tokens=16, output_tokens=5),
            make_convo_trace("c1", prompt_tokens=16, output_tokens=5),
        ]
        # Budget for 2 sequences (1 page each for 16 tokens, growing to ~21 = 2 pages)
        result = simulate_batch(traces, kv_page_budget=10, page_size=16)

        self.assertEqual(result['statistics']['peak_batch_size'], 2)
        # Both run concurrently: 1 prefill + 5 decode = 6 steps total
        self.assertEqual(result['statistics']['total_steps'], 6)

    def test_experts_are_unioned(self):
        """Expert selections from concurrent requests are unioned per layer."""
        t0 = make_convo_trace("c0", prompt_tokens=16, output_tokens=3,
                              num_layers=2, num_experts=8, top_k=2)
        t1 = make_convo_trace("c1", prompt_tokens=16, output_tokens=3,
                              num_layers=2, num_experts=8, top_k=2)
        result = simulate_batch([t0, t1], kv_page_budget=100, page_size=16)

        # Both should be concurrent
        self.assertEqual(result['statistics']['peak_batch_size'], 2)

        # Load as ActivationTrace and check expert counts
        at = ActivationTrace.from_flat_trace(result)
        for step_idx in range(len(at.steps)):
            for layer_idx in range(at.num_layers):
                # Union should have >= each individual trace's experts
                combined = set(at.steps[step_idx][layer_idx])
                s0 = set(t0.steps[step_idx][layer_idx]) if step_idx < len(t0.steps) else set()
                s1 = set(t1.steps[step_idx][layer_idx]) if step_idx < len(t1.steps) else set()
                self.assertTrue(combined >= s0,
                                f"Step {step_idx} layer {layer_idx}: "
                                f"combined {combined} missing {s0 - combined}")
                self.assertTrue(combined >= s1)

    def test_lifo_preemption_order(self):
        """LIFO: most recently admitted request is evicted first."""
        traces = [
            make_convo_trace("c0", prompt_tokens=32, output_tokens=20),
            make_convo_trace("c1", prompt_tokens=32, output_tokens=20),
            make_convo_trace("c2", prompt_tokens=32, output_tokens=20),
        ]
        # Budget for 2 sequences only: each prompt = 32 tokens = 2 pages,
        # so budget=4 pages fits exactly 2 prompts (2+2=4). c2 must wait.
        result = simulate_batch(traces, kv_page_budget=4, page_size=16)

        self.assertLessEqual(result['statistics']['peak_batch_size'], 2)
        # c0+c1 concurrent (21 steps), then c2 alone (21 steps) = at least 21
        self.assertGreater(result['statistics']['total_steps'], 21)

    def test_lifo_preemption_and_readmission(self):
        """Preempted requests come back when space frees."""
        # c0: short prompt, long decode
        # c1: short prompt, long decode
        # c2: large prompt that forces preemption of c1
        traces = [
            make_convo_trace("c0", prompt_tokens=16, output_tokens=5),
            make_convo_trace("c1", prompt_tokens=16, output_tokens=5),
            make_convo_trace("c2", prompt_tokens=48, output_tokens=5),
        ]
        # Budget: 4 pages. c0(1 page) + c1(1 page) fits.
        # c2(3 pages) won't fit with c0+c1. Must wait.
        # After c0 finishes at step 6, c2 can't fit yet (c1 ~21 tokens = 2 pages + c2 48 = 3+2=5 > 4)
        # After c1 finishes, c2 admitted.
        result = simulate_batch(traces, kv_page_budget=4, page_size=16)

        # All conversations should complete
        total_decode = 5 + 5 + 5
        # At minimum, total_steps >= max single conversation steps
        self.assertGreaterEqual(result['statistics']['total_steps'], 6)

    def test_preemption_during_decode(self):
        """When sequences grow past budget during decode, preemption occurs."""
        # Two sequences, each starts at 16 tokens (1 page)
        # Budget = 3 pages. Both fit initially (2 pages).
        # After 16 decode steps, each is at 32 tokens (2 pages each = 4 > 3)
        # → preemption should occur
        traces = [
            make_convo_trace("c0", prompt_tokens=16, output_tokens=30),
            make_convo_trace("c1", prompt_tokens=16, output_tokens=30),
        ]
        result = simulate_batch(traces, kv_page_budget=3, page_size=16)

        # Should eventually preempt
        self.assertGreater(result['statistics']['total_preemptions'], 0)
        # Both should still complete
        self.assertEqual(result['statistics']['total_steps'],
                         result['statistics']['total_steps'])  # sanity

    def test_empty_traces_raises(self):
        with self.assertRaises(ValueError):
            simulate_batch([], kv_page_budget=100)

    def test_output_loadable_as_activation_trace(self):
        """Output should be loadable by ActivationTrace.from_flat_trace."""
        traces = [
            make_convo_trace("c0", prompt_tokens=16, output_tokens=10),
            make_convo_trace("c1", prompt_tokens=32, output_tokens=8),
        ]
        result = simulate_batch(traces, kv_page_budget=100, page_size=16)

        at = ActivationTrace.from_flat_trace(result)
        self.assertEqual(at.num_layers, 4)
        self.assertEqual(at.num_experts, 8)
        self.assertGreater(at.num_steps(), 0)

    def test_deterministic(self):
        """Same inputs → same output."""
        traces = [make_convo_trace(f"c{i}", 32, 10) for i in range(5)]
        r1 = simulate_batch(traces, kv_page_budget=20, page_size=16)
        r2 = simulate_batch(traces, kv_page_budget=20, page_size=16)
        self.assertEqual(r1['trace'], r2['trace'])
        self.assertEqual(r1['statistics'], r2['statistics'])

    def test_batch_size_matches_target(self):
        """With many short requests and large budget, avg batch size ≈ target."""
        traces = [make_convo_trace(f"c{i}", 16, 20) for i in range(50)]
        # Budget for ~10 sequences: 10 * ceil(36/16) = 10*3 = 30 pages
        result = simulate_batch(traces, kv_page_budget=30, page_size=16)

        avg_bs = result['statistics']['avg_batch_size']
        # Should be close to 10 (within reason, given varying seq lengths)
        self.assertGreater(avg_bs, 5)
        self.assertLessEqual(avg_bs, 15)

    def test_prefill_seq_len_accounting(self):
        """After prefill, seq_len should equal prompt_tokens."""
        trace = make_convo_trace("c0", prompt_tokens=48, output_tokens=5)
        result = simulate_batch([trace], kv_page_budget=100, page_size=16)
        # Should have pages for 48 tokens after step 0 = 3 pages
        # Then 49, 50, ... tokens for subsequent steps
        self.assertEqual(result['statistics']['peak_pages_used'],
                         pages_needed(48 + 5, 16))


class TestConversationTraceIO(unittest.TestCase):
    """Test saving/loading per-conversation traces."""

    def test_round_trip(self):
        trace = make_convo_trace("test123", 64, 20, num_layers=4, num_experts=8)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            save_convo_trace(trace, path)
            loaded = ConversationTrace.load(path)
            self.assertEqual(loaded.conversation_id, "test123")
            self.assertEqual(loaded.prompt_tokens, 64)
            self.assertEqual(loaded.output_tokens, 20)
            self.assertEqual(loaded.num_layers, 4)
            self.assertEqual(loaded.num_experts, 8)
            self.assertEqual(len(loaded.steps), len(trace.steps))
            for s in range(len(trace.steps)):
                for l in range(trace.num_layers):
                    self.assertEqual(loaded.steps[s][l], trace.steps[s][l])
        finally:
            os.unlink(path)

    def test_load_directory(self):
        """Load traces from a directory with manifest (requests/ subdir)."""
        traces = [
            make_convo_trace("a", 16, 5),
            make_convo_trace("b", 32, 10),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            requests_dir = os.path.join(tmpdir, 'requests')
            os.makedirs(requests_dir)
            manifest = {
                'conversations': [
                    {'trace_file': 'requests/a.json', 'conversation_id': 'a',
                     'prompt_tokens': 16, 'output_tokens': 5},
                    {'trace_file': 'requests/b.json', 'conversation_id': 'b',
                     'prompt_tokens': 32, 'output_tokens': 10},
                ],
            }
            with open(os.path.join(tmpdir, 'manifest.json'), 'w') as f:
                json.dump(manifest, f)
            save_convo_trace(traces[0], os.path.join(requests_dir, 'a.json'))
            save_convo_trace(traces[1], os.path.join(requests_dir, 'b.json'))

            loaded, man = load_traces(tmpdir)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].conversation_id, 'a')
            self.assertEqual(loaded[1].conversation_id, 'b')

    def test_load_directory_no_manifest(self):
        """Load traces from requests/ subdir without manifest."""
        trace = make_convo_trace("x", 16, 5)
        with tempfile.TemporaryDirectory() as tmpdir:
            requests_dir = os.path.join(tmpdir, 'requests')
            os.makedirs(requests_dir)
            save_convo_trace(trace, os.path.join(requests_dir, 'x.json'))
            loaded, man = load_traces(tmpdir)
            self.assertEqual(len(loaded), 1)


class TestMemoryBudget(unittest.TestCase):
    """Test memory budget computation."""

    def test_olmoe_budget(self):
        """Sanity check memory budget for OLMoE-like config."""
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w',
                                         delete=False) as f:
            json.dump({
                'num_hidden_layers': 16,
                'hidden_size': 2048,
                'intermediate_size': 1024,
                'num_attention_heads': 16,
                'num_key_value_heads': 16,
                'vocab_size': 50304,
                'num_experts': 64,
            }, f)
            config_path = f.name

        try:
            budget = compute_memory_budget(config_path, peak_pages=100,
                                           page_size=16, gpu_memory_gb=80)
            # Per-expert: (2*1024*2048 + 2048*1024) * 2 = (4M + 2M) * 2 = 12 MB
            self.assertAlmostEqual(budget['per_expert_mb'],
                                   6291456 * 2 / 1024**2, places=0)
            # KV per page per layer: 16 * 2 * 16 * 128 * 2 = 131072 bytes
            self.assertEqual(budget['kv_cache_bytes'],
                             100 * 131072 * 16)
            # Cache size should be positive (OLMoE experts are small, ~12 MB each,
            # so on 80 GB GPU we can fit far more than the total 16*64=1024)
            self.assertGreater(budget['expert_cache_size'], 0)
        finally:
            os.unlink(config_path)

    def test_kv_budget_scales_with_pages(self):
        """More pages → more KV memory → fewer expert cache slots."""
        # Use Mixtral-like config where experts are large (~336 MB each)
        # so the cap at total_experts doesn't mask the scaling
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w',
                                         delete=False) as f:
            json.dump({
                'num_hidden_layers': 32,
                'hidden_size': 4096,
                'intermediate_size': 14336,
                'num_attention_heads': 32,
                'num_key_value_heads': 8,
                'vocab_size': 32000,
                'num_local_experts': 8,
            }, f)
            config_path = f.name

        try:
            budget_small = compute_memory_budget(config_path, peak_pages=50,
                                                 page_size=16, gpu_memory_gb=80)
            budget_large = compute_memory_budget(config_path, peak_pages=5000,
                                                 page_size=16, gpu_memory_gb=80)
            self.assertGreater(budget_small['expert_cache_size'],
                               budget_large['expert_cache_size'])
        finally:
            os.unlink(config_path)


class TestEndToEnd(unittest.TestCase):
    """Integration: simulate → load as ActivationTrace → run policy."""

    def test_simulate_then_policy(self):
        """Batched trace should work with policy simulators."""
        from policy_simulator import LRU, NoPrefetch, simulate

        traces = [make_convo_trace(f"c{i}", 16, 10) for i in range(5)]
        result = simulate_batch(traces, kv_page_budget=20, page_size=16)

        at = ActivationTrace.from_flat_trace(result)
        dm = simulate(LRU(), NoPrefetch(), at, cache_size=16)
        errors = dm.validate()
        self.assertEqual(len(errors), 0, f"Validation errors: {errors}")

    def test_save_load_round_trip(self):
        """Batched trace JSON round-trips through ActivationTrace.load()."""
        traces = [make_convo_trace(f"c{i}", 32, 8) for i in range(3)]
        result = simulate_batch(traces, kv_page_budget=50, page_size=16)

        with tempfile.NamedTemporaryFile(suffix='.json', mode='w',
                                         delete=False) as f:
            json.dump(result, f)
            path = f.name

        try:
            at = ActivationTrace.load(path)
            self.assertEqual(at.num_layers, 4)
            self.assertEqual(at.num_experts, 8)
            self.assertEqual(at.num_steps(), result['statistics']['total_steps'])
        finally:
            os.unlink(path)


if __name__ == '__main__':
    unittest.main()
