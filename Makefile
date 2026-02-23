# GPU Database Metal Benchmark Makefile
# Builds C++ project using metal-cpp library

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_NAME = GPUDBMetalBenchmark
SOURCE_DIR   = src
KERNEL_DIR   = kernels
METAL_CPP_DIR = third_party/metal-cpp
DATA_DIR     = data
BUILD_DIR    = build
BIN_DIR      = $(BUILD_DIR)/bin
OBJ_DIR      = $(BUILD_DIR)/obj

CXX      = clang++
CXXFLAGS = -std=c++20 -Wall -Wextra -O3
INCLUDES = -I$(METAL_CPP_DIR)
FRAMEWORKS = -framework Metal -framework Foundation -framework QuartzCore

SOURCES = $(wildcard $(SOURCE_DIR)/*.cpp)
OBJECTS = $(SOURCES:$(SOURCE_DIR)/%.cpp=$(OBJ_DIR)/%.o)
TARGET  = $(BIN_DIR)/$(PROJECT_NAME)

METAL    = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib
KERNEL_AIR      = $(BUILD_DIR)/kernels.air
KERNEL_METALLIB = $(BUILD_DIR)/kernels.metallib

# Queries and scale factors understood by the binary
QUERIES = q1 q3 q6 q9 q13
SCALE_FACTORS = sf1 sf10 sf100

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
.PHONY: all rebuild clean

# Metal compiler (requires full Xcode, not just Command Line Tools).
# If unavailable, the binary falls back to runtime compilation from
# kernels/DatabaseKernels.metal — so the metallib is optional.
METAL_AVAILABLE := $(shell xcrun --find metal 2>/dev/null && echo 1)

ifeq ($(METAL_AVAILABLE),1)
  all: $(TARGET) $(KERNEL_METALLIB)
else
  all: $(TARGET)
	@echo "Note: Metal compiler not found (Xcode required). Shaders will be compiled at runtime."
endif

rebuild: clean all

$(TARGET): $(OBJECTS) | $(BIN_DIR)
	@echo "Linking $(PROJECT_NAME)..."
	$(CXX) $(OBJECTS) $(FRAMEWORKS) -o $@
	@echo "Build complete: $@"

$(KERNEL_AIR): $(KERNEL_DIR)/DatabaseKernels.metal | $(BUILD_DIR)
	@echo "Compiling Metal kernels (.air)..."
	$(METAL) -c $< -o $@

$(KERNEL_METALLIB): $(KERNEL_AIR)
	@echo "Linking Metal library (.metallib)..."
	$(METALLIB) $< -o $@
	cp $@ default.metallib

$(OBJ_DIR)/%.o: $(SOURCE_DIR)/%.cpp | $(OBJ_DIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BIN_DIR) $(OBJ_DIR) $(BUILD_DIR):
	@mkdir -p $@

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR) default.metallib
	@echo "Clean complete"

# ---------------------------------------------------------------------------
# Run  —  make run  /  make run SF=sf100  /  make run SF=sf10 Q=q1
# ---------------------------------------------------------------------------
.PHONY: run

SF ?=
Q  ?=
run: all
	@./$(TARGET) $(SF) $(Q)

# ---------------------------------------------------------------------------
# Convenience per-scale-factor targets
# ---------------------------------------------------------------------------
.PHONY: run-sf1 run-sf10 run-sf100
run-sf1 run-sf10 run-sf100: run-%: all
	@./$(TARGET) $*

# ---------------------------------------------------------------------------
# Convenience per-query targets  (run-q1 … run-q13)
# ---------------------------------------------------------------------------
.PHONY: run-q1 run-q3 run-q6 run-q9 run-q13
run-q1 run-q3 run-q6 run-q9 run-q13: run-%: all
	@./$(TARGET) $*

# ---------------------------------------------------------------------------
# Verify project structure
# ---------------------------------------------------------------------------
.PHONY: check
check:
	@echo "Checking project structure..."
	@test -f $(SOURCE_DIR)/main.cpp          || (echo "ERROR: $(SOURCE_DIR)/main.cpp not found" && exit 1)
	@test -f $(KERNEL_DIR)/DatabaseKernels.metal || (echo "ERROR: DatabaseKernels.metal not found" && exit 1)
	@test -d $(METAL_CPP_DIR)                || (echo "ERROR: metal-cpp not found" && exit 1)
	@for sf in SF-1 SF-10 SF-100; do \
	  if [ -d $(DATA_DIR)/$$sf ]; then \
	    echo "  $$sf: OK"; \
	  else \
	    echo "  $$sf: not found (run scripts/create_tpch_data.sh)"; \
	  fi; \
	done
	@echo "Check complete"

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
.PHONY: help
help:
	@echo "GPU Database Metal Benchmark"
	@echo "============================"
	@echo ""
	@echo "Build:"
	@echo "  make              Build binary + Metal kernels"
	@echo "  make rebuild      Clean + build"
	@echo "  make clean        Remove build artifacts"
	@echo ""
	@echo "Run:"
	@echo "  make run                   All queries, default SF"
	@echo "  make run SF=sf100          All queries, SF-100"
	@echo "  make run SF=sf10 Q=q1      Single query + scale factor"
	@echo "  make run-sf1               Shorthand for SF=sf1"
	@echo "  make run-q6                Single query (default SF)"
	@echo ""
	@echo "Other:"
	@echo "  make check        Verify project structure"
	@echo "  make help         Show this message"