# Start from the NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:25.12-py3

# Set the working directory to ensure we are not in /root
WORKDIR /workspace

# Run the installation sequence in a single RUN instruction to keep the image layer clean.
# We also set MAX_JOBS to avoid running out of RAM during compilation.
RUN git clone https://github.com/vllm-project/vllm.git \
    && cd vllm \
    && git checkout v0.17.1 \
    && python use_existing_torch.py \
    && pip install -r requirements/build.txt \
    && export CFLAGS="-w" \
    && export CXXFLAGS="-w" \
    && export CUDAFLAGS="-w" \
    && export NVCC_PREPEND_FLAGS="-w" \
    && export MAX_JOBS=64 \
    && pip install -v -e . --no-build-isolation

# Set the default command to bash
CMD ["/bin/bash"]
