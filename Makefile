# Makefile for the CUDA PCIe Overlap Test
BIN_DIR = ./bin

# --- Auto-detect CUDA Installation ---
# 1. Prioritize the CUDA_HOME environment variable.
CUDA_PATH ?= $(CUDA_HOME)

# 2. If not set, try to find nvcc in the user's PATH.
ifeq ($(CUDA_PATH),)
    NVCC_PATH := $(shell which nvcc)
    # If nvcc is found, infer the CUDA_PATH from its location.
    ifneq ($(NVCC_PATH),)
        CUDA_PATH := $(shell dirname $(shell dirname $(NVCC_PATH)))
    else
        # 3. If nvcc isn't in PATH, fall back to a default location.
        CUDA_PATH := /usr/local/cuda
    endif
endif
# --- End Auto-detection ---

# Print the detected path for user feedback.
$(info CUDA installation detected at: $(CUDA_PATH))

# Compiler (use the detected path)
NVCC = $(CUDA_PATH)/bin/nvcc

# Compiler flags
NVCCFLAGS = -O3 -arch=native

# Linker flags
# Add the library path and set the runtime path (rpath) using the detected CUDA_PATH
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcurand -lcublas
LDFLAGS += -Xlinker -rpath -Xlinker $(CUDA_PATH)/lib64

# Target executable name
TARGET = memory_offload

# Source file
SRCS = src/cuda/memory_offload.cu

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(SRCS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $(BIN_DIR)/$@ $(LDFLAGS)

test: src/cuda/test.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $(BIN_DIR)/test $(LDFLAGS)

# Rule to create the bin directory
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Rule to clean up
clean:
	rm -f $(BIN_DIR)/$(TARGET) $(BIN_DIR)/test

.PHONY: all clean