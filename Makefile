# GPU Database Metal Benchmark Makefile
# Builds two binaries:
#   GPUDBMetalBenchmark  – hand-written GPU benchmarks (src/*.cpp + codegen library)
#   GPUDBCodegen         – codegen pipeline (codegen/codegen_main.cpp + codegen library + infra)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_NAME = GPUDBMetalBenchmark
CODEGEN_NAME = GPUDBCodegen
SOURCE_DIR   = src
KERNEL_DIR   = kernels
METAL_CPP_DIR = third_party/metal-cpp
DATA_DIR     = data
BUILD_DIR    = build
BIN_DIR      = $(BUILD_DIR)/bin
OBJ_DIR      = $(BUILD_DIR)/obj

CXX      = clang++
CXXFLAGS = -std=c++20 -Wall -Wextra -O3
INCLUDES = -I$(METAL_CPP_DIR) -I$(PG_QUERY_DIR) -Icodegen
FRAMEWORKS = -framework Metal -framework Foundation -framework QuartzCore -L/opt/homebrew/opt/llvm/lib/c++ -Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++

SOURCES = $(wildcard $(SOURCE_DIR)/*.cpp)
OBJECTS = $(SOURCES:$(SOURCE_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Codegen sources (exclude codegen_main.cpp from shared library objects)
CODEGEN_DIR = codegen
CODEGEN_ALL_SOURCES = $(wildcard $(CODEGEN_DIR)/*.cpp)
CODEGEN_LIB_SOURCES = $(filter-out $(CODEGEN_DIR)/codegen_main.cpp, $(CODEGEN_ALL_SOURCES))
CODEGEN_LIB_OBJECTS = $(CODEGEN_LIB_SOURCES:$(CODEGEN_DIR)/%.cpp=$(OBJ_DIR)/codegen_%.o)
CODEGEN_MAIN_OBJ    = $(OBJ_DIR)/codegen_codegen_main.o

BENCH_TARGET  = $(BIN_DIR)/$(PROJECT_NAME)
CODEGEN_TARGET = $(BIN_DIR)/$(CODEGEN_NAME)

# libpg_query
PG_QUERY_DIR = third_party/libpg_query
PG_QUERY_LIB = $(PG_QUERY_DIR)/libpg_query.a

METAL    = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib
KERNEL_AIR      = $(BUILD_DIR)/kernels.air
KERNEL_METALLIB = $(BUILD_DIR)/kernels.metallib

# Queries and scale factors understood by the binary
QUERIES = q1 q2 q3 q4 q5 q6 q7 q8 q9 q10 q11 q12 q13 q14 q15 q16 q17 q18 q19 q20 q21 q22
SCALE_FACTORS = sf1 sf10 sf100

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
.PHONY: all rebuild clean

# Metal compiler (requires full Xcode, not just Command Line Tools).
# If unavailable, the binary falls back to runtime compilation from
# kernels/DatabaseKernels.metal — so the metallib is optional.
METAL_AVAILABLE := $(shell xcrun --find metal >/dev/null 2>/dev/null && echo 1)

ifeq ($(METAL_AVAILABLE),1)
  all: $(BENCH_TARGET) $(CODEGEN_TARGET) $(KERNEL_METALLIB)
else
  all: $(BENCH_TARGET) $(CODEGEN_TARGET)
	@echo "Note: Metal compiler not found (Xcode required). Shaders will be compiled at runtime."
endif

rebuild: clean all

# Benchmark binary: src/*.o only (no codegen dependency)
$(BENCH_TARGET): $(OBJECTS) | $(BIN_DIR)
	@echo "Linking $(PROJECT_NAME)..."
	$(CXX) $(OBJECTS) $(FRAMEWORKS) -o $@
	@echo "Build complete: $@"

# Codegen binary: codegen_main.o + codegen library objects + infra.o (no src/main.o)
$(CODEGEN_TARGET): $(CODEGEN_MAIN_OBJ) $(CODEGEN_LIB_OBJECTS) $(OBJ_DIR)/infra.o | $(BIN_DIR)
	@echo "Linking $(CODEGEN_NAME)..."
	$(CXX) $(CODEGEN_MAIN_OBJ) $(CODEGEN_LIB_OBJECTS) $(OBJ_DIR)/infra.o $(PG_QUERY_LIB) $(FRAMEWORKS) -o $@
	@echo "Build complete: $@"

$(KERNEL_AIR): $(KERNEL_DIR)/DatabaseKernels.metal $(wildcard $(KERNEL_DIR)/*.metal $(KERNEL_DIR)/*.h) | $(BUILD_DIR)
	@echo "Compiling Metal kernels (.air)..."
	$(METAL) -I $(KERNEL_DIR) -c $< -o $@

$(KERNEL_METALLIB): $(KERNEL_AIR)
	@echo "Linking Metal library (.metallib)..."
	$(METALLIB) $< -o $@

$(OBJ_DIR)/%.o: $(SOURCE_DIR)/%.cpp | $(OBJ_DIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/codegen_%.o: $(CODEGEN_DIR)/%.cpp | $(OBJ_DIR)
	@echo "Compiling codegen/$<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Isrc -c $< -o $@

$(BIN_DIR) $(OBJ_DIR) $(BUILD_DIR):
	@mkdir -p $@

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@echo "Clean complete"

# ---------------------------------------------------------------------------
# Run  —  make run  /  make run SF=sf100  /  make run SF=sf10 Q=q1
# ---------------------------------------------------------------------------
.PHONY: run

SF ?=
Q  ?=
run: all
	@./$(BENCH_TARGET) $(SF) $(Q)

# Run codegen binary:  make codegen SF=sf1 Q=q6
.PHONY: codegen
codegen: all
	@./$(CODEGEN_TARGET) $(SF) $(Q)

# ---------------------------------------------------------------------------
# Convenience per-scale-factor targets
# ---------------------------------------------------------------------------
.PHONY: run-sf1 run-sf10 run-sf100
run-sf1 run-sf10 run-sf100: run-%: all
	@./$(BENCH_TARGET) $*

# ---------------------------------------------------------------------------
# Convenience per-query targets  (run-q1 … run-q19)
# ---------------------------------------------------------------------------
.PHONY: $(addprefix run-,$(QUERIES))
$(addprefix run-,$(QUERIES)): run-%: all
	@./$(BENCH_TARGET) $*

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