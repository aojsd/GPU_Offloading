"""Tests for trace_construction/build_trace.py — continuous batching simulator.

CPU-only tests using synthetic per-conversation traces. No GPU or model needed.

Tests use the fixed-chunk prefill semantics: each prefill chunk is a separate
trace step (matching collect_traces.py per-chunk expert routing), and convo_step
advances per chunk.
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
    ConversationTrace, PREFILL_CHUNK_SIZE, simulate_batch, pages_needed,
    compute_memory_budget, load_traces,
)
from gpu_replay_trace import ActivationTrace


def make_convo_trace(conv_id, prompt_tokens, output_tokens,
                     num_layers=4, num_experts=8, top_k=2,
                     prefill_chunk_size=PREFILL_CHUNK_SIZE):
    """Create a synthetic ConversationTrace with per-chunk prefill steps.

    Matches the format produced by collect_traces.py: each 256-token prefill
    chunk is a separate step with its own expert routing.
    """
    steps = []
    num_prefill_chunks = math.ceil(prompt_tokens / prefill_chunk_size)

    # Steps 0..K-1: prefill chunks (each with distinct experts)
    for chunk_idx in range(num_prefill_chunks):
        chunk_experts = []
        for layer in range(num_layers):
            experts = [(prompt_tokens + chunk_idx * 7 + layer * 3 + i) % num_experts
                       for i in range(top_k)]
            chunk_experts.append(sorted(set(experts)))
        steps.append(chunk_experts)

    # Steps K..K+N-1: decode
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
        """One conversation with plenty of budget should run without issue."""
        trace = make_convo_trace("conv0", prompt_tokens=32, output_tokens=10)
        result = simulate_batch([trace], kv_page_budget=100, page_size=16)

        self.assertEqual(result['scheduling']['preemption_policy'], 'none')
        self.assertEqual(result['statistics']['peak_batch_size'], 1)
        # ceil(32/256)=1 prefill + 10 decode = 11 steps
        self.assertEqual(result['statistics']['total_steps'], 11)

    def test_two_conversations_sequential(self):
        """Two convos with budget for only one at a time -> sequential."""
        traces = [
            make_convo_trace("c0", prompt_tokens=32, output_tokens=5),
            make_convo_trace("c1", prompt_tokens=32, output_tokens=5),
        ]
        # Full sequence pages: ceil((32+5)/16) = 3 pages each.
        # Budget for only 1 sequence at a time.
        result = simulate_batch(traces, kv_page_budget=3, page_size=16)

        self.assertEqual(result['statistics']['peak_batch_size'], 1)
        # Each conv: 1 prefill + 5 decode = 6 steps. Sequential = 12.
        self.assertEqual(result['statistics']['total_steps'], 12)

    def test_two_conversations_concurrent(self):
        """Two convos with enough budget -> concurrent execution."""
        traces = [
            make_convo_trace("c0", prompt_tokens=16, output_tokens=5),
            make_convo_trace("c1", prompt_tokens=16, output_tokens=5),
        ]
        # Full sequence: ceil((16+5)/16) = 2 pages each. Need 4 for both.
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
                combined = set(at.steps[step_idx][layer_idx])
                s0 = set(t0.steps[step_idx][layer_idx]) if step_idx < len(t0.steps) else set()
                s1 = set(t1.steps[step_idx][layer_idx]) if step_idx < len(t1.steps) else set()
                self.assertTrue(combined >= s0)
                self.assertTrue(combined >= s1)

    def test_full_sequence_preallocation(self):
        """Pages are pre-allocated for full sequence (prompt + output)."""
        traces = [
            make_convo_trace("c0", prompt_tokens=16, output_tokens=30),
            make_convo_trace("c1", prompt_tokens=16, output_tokens=30),
        ]
        # Full sequence: ceil((16+30)/16) = 3 pages each.
        # Budget = 5: only fits one at a time (3+3=6 > 5).
        result = simulate_batch(traces, kv_page_budget=5, page_size=16)
        self.assertEqual(result['statistics']['peak_batch_size'], 1)
        self.assertEqual(result['scheduling']['preemption_policy'], 'none')
        self.assertEqual(result['scheduling']['page_allocation'], 'full_sequence')

    def test_no_preemption_with_full_preallocation(self):
        """With full sequence pre-allocation, no preemptions should occur."""
        traces = [
            make_convo_trace("c0", prompt_tokens=16, output_tokens=30),
            make_convo_trace("c1", prompt_tokens=16, output_tokens=30),
        ]
        result = simulate_batch(traces, kv_page_budget=6, page_size=16)
        preemptions = sum(1 for e in result['scheduling_events']
                         if e['event'] == 'preempt')
        self.assertEqual(preemptions, 0)

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
        """Same inputs -> same output."""
        traces = [make_convo_trace(f"c{i}", 32, 10) for i in range(5)]
        r1 = simulate_batch(traces, kv_page_budget=200, page_size=16)
        r2 = simulate_batch(traces, kv_page_budget=200, page_size=16)
        self.assertEqual(r1['trace'], r2['trace'])
        self.assertEqual(r1['statistics'], r2['statistics'])

    def test_batch_size_matches_target(self):
        """With many short requests and large budget, avg batch size is reasonable."""
        traces = [make_convo_trace(f"c{i}", 16, 20) for i in range(50)]
        # Full seq: ceil((16+20)/16) = 3 pages each. Budget for ~10: 30 pages.
        result = simulate_batch(traces, kv_page_budget=30, page_size=16)

        avg_bs = result['statistics']['avg_batch_size']
        self.assertGreater(avg_bs, 5)
        self.assertLessEqual(avg_bs, 15)

    def test_prefill_seq_len_accounting(self):
        """Peak pages should reflect full sequence pre-allocation."""
        trace = make_convo_trace("c0", prompt_tokens=48, output_tokens=5)
        result = simulate_batch([trace], kv_page_budget=100, page_size=16)
        # Full sequence: ceil((48+5)/16) = 4 pages
        self.assertEqual(result['statistics']['peak_pages_used'],
                         pages_needed(48 + 5, 16))

    def test_all_requests_complete(self):
        """All requests should complete."""
        traces = [make_convo_trace(f"c{i}", 32, 10) for i in range(10)]
        result = simulate_batch(traces, kv_page_budget=100, page_size=16)
        completed = sum(1 for e in result['scheduling_events']
                       if e['event'] == 'complete')
        self.assertEqual(completed, 10)


class TestFixedChunkPrefill(unittest.TestCase):
    """Test fixed-size chunked prefill (256-token chunks)."""

    def test_max_graph_size_is_token_budget(self):
        """Total tokens per step should not exceed max_graph_size."""
        # Use small chunk size for testing
        traces = [make_convo_trace(f"c{i}", 64, 10, prefill_chunk_size=16)
                  for i in range(5)]
        result = simulate_batch(
            traces, kv_page_budget=1000, page_size=16,
            max_graph_size=32, prefill_chunk_size=16,
        )
        for ss in result['step_scheduling']:
            self.assertLessEqual(ss['total_tokens'], 32,
                                 f"Step {ss['step']}: total_tokens={ss['total_tokens']} > 32")

    def test_fixed_chunk_size_creates_continuations(self):
        """Prompts larger than chunk size produce continuation chunks."""
        # 512-token prompt with 256-token chunks = 2 chunks
        traces = [make_convo_trace("c0", prompt_tokens=512, output_tokens=5)]
        result = simulate_batch(
            traces, kv_page_budget=1000, page_size=16,
            max_graph_size=512,
        )
        has_continuation = any(
            ar['is_continuation']
            for ss in result['step_scheduling']
            for ar in ss['active_requests']
        )
        self.assertTrue(has_continuation,
                        "Expected continuation chunks for 512-token prompt")

    def test_convo_step_advances_per_chunk(self):
        """convo_step should advance by 1 for each prefill chunk.

        Each chunk uses its own trace step's expert routing (per-chunk
        routing from collect_traces.py).
        """
        # 600-token prompt: 3 chunks (256, 256, 88)
        traces = [make_convo_trace("c0", prompt_tokens=600, output_tokens=5)]
        result = simulate_batch(
            traces, kv_page_budget=1000, page_size=16,
            max_graph_size=512,
        )
        at = ActivationTrace.from_flat_trace(result)

        # Each prefill step should use a different convo_step's experts
        prefill_step_experts = []
        for step_idx, ss in enumerate(result['step_scheduling']):
            for ar in ss['active_requests']:
                if ar['is_prefill']:
                    prefill_step_experts.append(
                        [sorted(at.steps[step_idx][l]) for l in range(at.num_layers)]
                    )

        # Should have 3 prefill steps with per-chunk expert routing
        self.assertEqual(len(prefill_step_experts), 3)
        # Each should match the corresponding chunk's trace step
        for chunk_idx, experts in enumerate(prefill_step_experts):
            expected = [sorted(traces[0].steps[chunk_idx][l])
                       for l in range(traces[0].num_layers)]
            self.assertEqual(experts, expected,
                             f"Chunk {chunk_idx}: expert routing mismatch")

    def test_chunk_sizes_are_fixed(self):
        """Chunk sizes should be exactly prefill_chunk_size (except last)."""
        # 600-token prompt: chunks should be 256, 256, 88
        traces = [make_convo_trace("c0", prompt_tokens=600, output_tokens=5)]
        result = simulate_batch(
            traces, kv_page_budget=1000, page_size=16,
            max_graph_size=512,
        )
        chunks = []
        for ss in result['step_scheduling']:
            for ar in ss['active_requests']:
                if ar['is_prefill']:
                    chunks.append(ar['prefill_chunk_length'])

        self.assertEqual(chunks, [256, 256, 88],
                         f"Expected [256, 256, 88], got {chunks}")

    def test_force_admit_correctness(self):
        """force_admit should start with num_computed_tokens=0 and complete."""
        traces = [make_convo_trace("c0", prompt_tokens=256, output_tokens=50)]
        # Budget = 1 page (way too small for ceil((256+50)/16) = 20 pages)
        result = simulate_batch(traces, kv_page_budget=1, page_size=16)

        force_admits = [e for e in result['scheduling_events']
                       if e['event'] == 'force_admit']
        self.assertEqual(len(force_admits), 1)
        completed = [e for e in result['scheduling_events']
                    if e['event'] == 'complete']
        self.assertEqual(len(completed), 1)

    def test_continuation_offsets_sequential(self):
        """Continuation chunk offsets should be sequential with fixed sizes."""
        # Use small chunks for clearer testing
        traces = [make_convo_trace("c0", prompt_tokens=100, output_tokens=5,
                                   prefill_chunk_size=32)]
        result = simulate_batch(
            traces, kv_page_budget=1000, page_size=16,
            max_graph_size=512, prefill_chunk_size=32,
        )
        offsets = []
        for ss in result['step_scheduling']:
            for ar in ss['active_requests']:
                if ar['is_prefill']:
                    offsets.append((ar['prefill_chunk_start'],
                                   ar['prefill_chunk_length']))

        # Verify offsets are contiguous with fixed chunk sizes
        expected_start = 0
        for start, length in offsets:
            self.assertEqual(start, expected_start)
            expected_start += length
        self.assertEqual(expected_start, 100)

        # All chunks except last should be exactly 32
        for _, length in offsets[:-1]:
            self.assertEqual(length, 32)
        # Last chunk is remainder
        self.assertEqual(offsets[-1][1], 100 % 32)

    def test_max_graph_size_caps_tokens(self):
        """max_graph_size should cap total tokens per step."""
        traces = [
            make_convo_trace("c0", prompt_tokens=16, output_tokens=5),
            make_convo_trace("c1", prompt_tokens=512, output_tokens=5),
        ]
        result = simulate_batch(
            traces, kv_page_budget=1000, page_size=16,
            max_graph_size=512,
        )
        for ss in result['step_scheduling']:
            self.assertLessEqual(ss['total_tokens'], 512)

    def test_decode_interleaved_with_prefill_chunks(self):
        """Decode tokens should share budget with prefill chunks."""
        traces = [
            make_convo_trace("c0", prompt_tokens=16, output_tokens=20),
            make_convo_trace("c1", prompt_tokens=600, output_tokens=5),
        ]
        result = simulate_batch(
            traces, kv_page_budget=1000, page_size=16,
            max_graph_size=512,
        )
        mixed_steps = 0
        for ss in result['step_scheduling']:
            has_decode = any(not ar['is_prefill'] for ar in ss['active_requests'])
            has_prefill = any(ar['is_prefill'] for ar in ss['active_requests'])
            if has_decode and has_prefill:
                mixed_steps += 1
        self.assertGreater(mixed_steps, 0,
                           "Expected mixed decode+prefill steps")

    def test_no_token_budget_backward_compat(self):
        """Without max_graph_size, prefills fit in one chunk (prompt<=256)."""
        traces = [make_convo_trace(f"c{i}", 32, 10) for i in range(3)]
        result = simulate_batch(traces, kv_page_budget=100, page_size=16)
        # All prefills with prompt<=256 should complete in one step
        for ss in result['step_scheduling']:
            for ar in ss['active_requests']:
                if ar['is_prefill']:
                    self.assertFalse(ar['is_continuation'])

    def test_total_steps_with_chunked_prefill(self):
        """Total steps should account for per-chunk prefill steps."""
        # 600-token prompt: 3 prefill chunks + 5 decode = 8 steps
        traces = [make_convo_trace("c0", prompt_tokens=600, output_tokens=5)]
        result = simulate_batch(
            traces, kv_page_budget=1000, page_size=16,
            max_graph_size=512,
        )
        self.assertEqual(result['statistics']['total_steps'], 8)

    def test_scheduling_metadata_has_prefill_chunk_size(self):
        """Scheduling metadata should include prefill_chunk_size."""
        traces = [make_convo_trace("c0", prompt_tokens=32, output_tokens=5)]
        result = simulate_batch(traces, kv_page_budget=100, page_size=16)
        self.assertEqual(result['scheduling']['prefill_chunk_size'],
                         PREFILL_CHUNK_SIZE)


class TestSchedulingMetadata(unittest.TestCase):
    """Test per-step scheduling metadata for replay."""

    def test_step_scheduling_present(self):
        traces = [make_convo_trace("c0", prompt_tokens=16, output_tokens=5)]
        result = simulate_batch(traces, kv_page_budget=100, page_size=16)
        self.assertIn('step_scheduling', result)
        self.assertEqual(len(result['step_scheduling']),
                         result['statistics']['total_steps'])

    def test_active_requests_fields(self):
        traces = [make_convo_trace("c0", prompt_tokens=600, output_tokens=5)]
        result = simulate_batch(
            traces, kv_page_budget=100, page_size=16,
            max_graph_size=512,
        )
        for ss in result['step_scheduling']:
            for ar in ss['active_requests']:
                self.assertIn('request_id', ar)
                self.assertIn('is_prefill', ar)
                self.assertIn('prefill_chunk_start', ar)
                self.assertIn('prefill_chunk_length', ar)
                self.assertIn('is_continuation', ar)

    def test_admit_events_for_all_requests(self):
        traces = [make_convo_trace(f"c{i}", 16, 5) for i in range(5)]
        result = simulate_batch(traces, kv_page_budget=1000, page_size=16)
        admitted = set()
        for e in result['scheduling_events']:
            if e['event'] in ('admit', 'force_admit'):
                admitted.add(e['request_id'])
        self.assertEqual(admitted, set(range(5)))


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
            self.assertAlmostEqual(budget['per_expert_mb'],
                                   6291456 * 2 / 1024**2, places=0)
            self.assertEqual(budget['kv_cache_bytes'],
                             100 * 131072 * 16)
            self.assertGreater(budget['expert_cache_size'], 0)
        finally:
            os.unlink(config_path)

    def test_kv_budget_scales_with_pages(self):
        """More pages -> more KV memory -> fewer expert cache slots."""
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
    """Integration: simulate -> load as ActivationTrace -> run policy."""

    def test_simulate_then_policy(self):
        """Batched trace should work with policy simulators."""
        from policy_simulator import LRU, NoPrefetch, simulate

        traces = [make_convo_trace(f"c{i}", 16, 10) for i in range(5)]
        result = simulate_batch(traces, kv_page_budget=200, page_size=16)

        at = ActivationTrace.from_flat_trace(result)
        dm = simulate(LRU(), NoPrefetch(), at, cache_size=16)
        errors = dm.validate()
        self.assertEqual(len(errors), 0, f"Validation errors: {errors}")

    def test_simulate_chunked_then_policy(self):
        """Chunked prefill trace should also work with policy simulators."""
        from policy_simulator import LRU, NoPrefetch, simulate

        traces = [make_convo_trace(f"c{i}", 600, 10) for i in range(5)]
        result = simulate_batch(
            traces, kv_page_budget=2000, page_size=16,
            max_graph_size=512,
        )

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
