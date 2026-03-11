#!/usr/bin/env bash
# Shared environment setup for experiment scripts.
# Source this at the top of each script: source "$(dirname "$0")/env.sh"

# Detect environment: NGC container vs bare-metal RHEL
if [ -f /etc/nv_tegra_release ] || [ -d /opt/pytorch ]; then
    # NGC container — CUDA, Python, PyTorch already configured
    :
else
    # Bare-metal RHEL 8.10 / x86 cluster
    # Conda (base env — PATH prepend, no conda activate needed)
    export PATH="$HOME/software/miniconda3/bin:$PATH"

    # CUDA 12.8
    export CUDA_HOME="$HOME/software/cuda-12.8"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

    # GCC 13.3 + binutils 2.42 (needed for torch.compile / Triton JIT on RHEL 8.10)
    export GCC_ROOT=/gpfs/radev/apps/avx512/software/GCCcore/13.3.0
    export BINUTILS=/gpfs/radev/apps/avx512/software/binutils/2.42-GCCcore-13.3.0
    export CC="$GCC_ROOT/bin/gcc"
    export CXX="$GCC_ROOT/bin/g++"
    export PATH="$BINUTILS/bin:$GCC_ROOT/bin:$PATH"
    export LD_LIBRARY_PATH="$GCC_ROOT/lib64:$BINUTILS/lib:${LD_LIBRARY_PATH:-}"

    # Ninja for FlashInfer RoPE JIT
    export PATH="$HOME/.local/bin:$PATH"
fi

# vLLM
export VLLM_ENABLE_V1_MULTIPROCESSING=0
