import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# -----------------------------------------------------------------------------
# 1. Setup & Configuration
# -----------------------------------------------------------------------------
device = "cuda"
dtype = torch.float16

BATCH = 1
HEADS = 4
DIM = 64

L_LAYERS = 20       # Number of layers
A_RESIDENT = 128    # Resident tokens per layer (permanent GPU memory)
B_OFFLOADED = 128   # Offloaded tokens per layer (swapped into scratchpad)

# Physical Tensor Layout:
# [ Layer0_Res | Layer1_Res | ... | Shared_Scratchpad ]
PHYSICAL_LEN = (L_LAYERS * A_RESIDENT) + B_OFFLOADED
SCRATCHPAD_START = PHYSICAL_LEN - B_OFFLOADED

print(f"--- Configuration ---")
print(f"Physical Tensor Size: {PHYSICAL_LEN}")
print(f"Scratchpad Range: [{SCRATCHPAD_START}, {PHYSICAL_LEN})")
print(f"Note: Scratchpad is overwritten for every layer.\n")

# -----------------------------------------------------------------------------
# 2. Mask Logic
# -----------------------------------------------------------------------------
def get_layer_mask_mod(layer_idx):
    res_start = layer_idx * A_RESIDENT
    res_end   = (layer_idx + 1) * A_RESIDENT
    
    # We capture the constants for this specific layer
    def mask_mod(b, h, q_idx, kv_idx):
        # 1. Is this token in the layer's dedicated RESIDENT section?
        is_resident = (kv_idx >= res_start) & (kv_idx < res_end)
        
        # 2. Is this token in the SHARED SCRATCHPAD section?
        # (We always allow attention to the scratchpad, assuming the correct 
        #  data has been loaded there before calling this)
        is_scratchpad = kv_idx >= SCRATCHPAD_START
        
        return is_resident | is_scratchpad
    return mask_mod

# -----------------------------------------------------------------------------
# 3. Main Loop: Layer-by-Layer Simulation
# -----------------------------------------------------------------------------
# Global Tensors (The "System Memory")
# We allocate these ONCE. The Resident parts will stay resident.
k_giant = torch.zeros(BATCH, HEADS, PHYSICAL_LEN, DIM, device=device, dtype=dtype)
v_giant = torch.zeros(BATCH, HEADS, PHYSICAL_LEN, DIM, device=device, dtype=dtype)

query = torch.randn(BATCH, HEADS, 1, DIM, device=device, dtype=dtype)

# --- Step 3a: Pre-populate Resident Memory ---
# In a real system, this data is already sitting in GPU memory from previous steps.
layer_resident_data = [] # To keep track for validation
layer_offload_data = []  # To keep track for validation

for i in range(L_LAYERS):
    # Generate unique data for this layer
    k_res = torch.randn(BATCH, HEADS, A_RESIDENT, DIM, device=device, dtype=dtype)
    v_res = torch.randn(BATCH, HEADS, A_RESIDENT, DIM, device=device, dtype=dtype)
    
    # Store logically for validation later
    layer_resident_data.append((k_res, v_res))
    
    # Write physically to the permanent slot
    start = i * A_RESIDENT
    end   = start + A_RESIDENT
    k_giant[:, :, start:end, :] = k_res
    v_giant[:, :, start:end, :] = v_res

    # Generate unique "Offloaded" data (currently on "CPU")
    k_off = torch.randn(BATCH, HEADS, B_OFFLOADED, DIM, device=device, dtype=dtype)
    v_off = torch.randn(BATCH, HEADS, B_OFFLOADED, DIM, device=device, dtype=dtype)
    layer_offload_data.append((k_off, v_off))


# --- Step 3b: Run The Layers ---
all_layers_passed = True
for layer_idx in range(L_LAYERS):
    print(f"--- Processing Layer {layer_idx} ---")
    
    # 1. LOAD: Copy this layer's offloaded data into the Shared Scratchpad
    #    (Simulating the Memory Offload Transfer)
    k_off_src, v_off_src = layer_offload_data[layer_idx]
    
    k_giant[:, :, SCRATCHPAD_START:, :] = k_off_src
    v_giant[:, :, SCRATCHPAD_START:, :] = v_off_src

    # 2. MASK: Create mask specific to this layer's resident slot + scratchpad
    mask_mod = get_layer_mask_mod(layer_idx)
    block_mask = create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=1, KV_LEN=PHYSICAL_LEN, device=device
    )

    # 3. EXECUTE: Flex Attention on the Giant Tensor
    out_offloaded = torch.compile(flex_attention)(query, k_giant, v_giant, block_mask=block_mask)

    # --- Validation ---
    # Retrieve the pure logic data to build a standard baseline
    k_res_pure, v_res_pure = layer_resident_data[layer_idx]
    
    # Baseline: Cat(Resident, Offloaded) -> Standard Attention
    k_base = torch.cat([k_res_pure, k_off_src], dim=2)
    v_base = torch.cat([v_res_pure, v_off_src], dim=2)
    
    out_base = torch.compile(flex_attention)(query, k_base, v_base)
    
    # Compare
    is_close = torch.allclose(out_base, out_offloaded, atol=1e-3, rtol=1e-3)
    diff = (out_base - out_offloaded).abs().max().item()
    
    print(f"   Max Diff: {diff:.6f}")
    print(f"   Result: {'SUCCESS' if is_close else 'FAILURE'}")
    all_layers_passed = all_layers_passed and is_close
if all_layers_passed:
    print(f"\nALL LAYERS PASSED!")
else:
    print(f"\nSOME LAYERS FAILED!")