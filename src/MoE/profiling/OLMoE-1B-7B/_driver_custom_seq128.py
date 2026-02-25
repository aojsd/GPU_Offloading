import sys, torch
sys.path.insert(0, "/home/mw976/GPU_Offloading/src/MoE")
from moe_engine import MoEEngine

SEQ_LEN = 128
MAX_SEQ = 2560
NUM_WARMUP = 10
NUM_DECODE = 30

engine = MoEEngine("/home/mw976/GPU_Offloading/src/MoE/models/OLMoE-1B-7B", max_batch_size=1, max_seq_len=MAX_SEQ)
engine.capture_decode_cuda_graph(
    batch_size=1, warmup_seq_len=128,
    max_decode_tokens=MAX_SEQ - 128)

# Fill KV cache with random data to simulate target seq_len
engine.k_cache.normal_(0, 0.01)
engine.v_cache.normal_(0, 0.01)

token = torch.tensor([100], device="cuda")
pos = torch.tensor([SEQ_LEN], dtype=torch.int32, device="cuda")

# Warmup (graph already captured, just exercise replay)
for _ in range(NUM_WARMUP):
    engine.seq_lens[0] = SEQ_LEN
    engine._decode_step_graphed(token, pos)
torch.cuda.synchronize()

# Profiled region
torch.cuda.cudart().cudaProfilerStart()
for _ in range(NUM_DECODE):
    engine.seq_lens[0] = SEQ_LEN
    engine._decode_step_graphed(token, pos)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
