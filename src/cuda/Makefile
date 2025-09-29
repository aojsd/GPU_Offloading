# Makefile for the CUDA PCIe Overlap Test
BIN_DIR = ../../bin

# Compiler
NVCC = nvcc

# Compiler flags
# -O3 for optimization
# -arch=native will detect the architecture of the GPU(s) on the build system.
# If you are cross-compiling, you must specify the arch manually, e.g., -arch=sm_86
NVCCFLAGS = -O3 -arch=native

# Linker flags
# We need to link against the cuRAND library for GPU-based random number generation
LDFLAGS = -lcurand

# Target executable name
TARGET = pcie_overlap

# Source file
SRCS = pcie_overlap.cu

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(SRCS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $(BIN_DIR)/$@ $(LDFLAGS)

test: test.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $(BIN_DIR)/test $(LDFLAGS)

# Rule to create the bin directory if it doesn't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Rule to clean up generated files
clean:
	rm -f $(BIN_DIR)/$(TARGET) $(BIN_DIR)/test

.PHONY: all clean

