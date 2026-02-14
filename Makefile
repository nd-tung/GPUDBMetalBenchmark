# GPU Database Metal Benchmark Makefile
# Builds C++ project using metal-cpp library

# Project Configuration
PROJECT_NAME = GPUDBMetalBenchmark
SOURCE_DIR = src
KERNEL_DIR = kernels
METAL_CPP_DIR = third_party/metal-cpp
DATA_DIR = data
BUILD_DIR = build
BIN_DIR = $(BUILD_DIR)/bin
OBJ_DIR = $(BUILD_DIR)/obj

# Compiler and flags
CXX = clang++
CXXFLAGS = -std=c++20 -Wall -Wextra -O3

# Include paths
INCLUDES = -I$(METAL_CPP_DIR)

# Framework flags for macOS
FRAMEWORKS = -framework Metal -framework Foundation -framework QuartzCore

# Source files
SOURCES = $(wildcard $(SOURCE_DIR)/*.cpp)
OBJECTS = $(SOURCES:$(SOURCE_DIR)/%.cpp=$(OBJ_DIR)/%.o)
KERNELS = $(wildcard $(KERNEL_DIR)/*.metal)

# Target executable
TARGET = $(BIN_DIR)/$(PROJECT_NAME)

# Metal compiler tools
METAL = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib
KERNEL_AIR = $(BUILD_DIR)/kernels.air
KERNEL_METALLIB = $(BUILD_DIR)/kernels.metallib
BUILD_SENTINEL = $(BUILD_DIR)/.dir

# Default target
.PHONY: all
all: $(TARGET) $(KERNEL_METALLIB)

# Create target executable
$(TARGET): $(OBJECTS) | $(BIN_DIR)
	@echo "Linking $(PROJECT_NAME)..."
	$(CXX) $(OBJECTS) $(FRAMEWORKS) -o $@
	@echo "Build complete: $@"

# Build Metal kernels and place fresh metallib alongside the app assets
$(KERNEL_AIR): $(KERNEL_DIR)/DatabaseKernels.metal | $(BUILD_SENTINEL)
	@echo "Compiling Metal kernels (.air)..."
	$(METAL) -c $(KERNEL_DIR)/DatabaseKernels.metal -o $(KERNEL_AIR)

$(KERNEL_METALLIB): $(KERNEL_AIR)
	@echo "Linking Metal library (.metallib)..."
	$(METALLIB) $(KERNEL_AIR) -o $(KERNEL_METALLIB)
	@# Copy to runtime location so device->newLibrary("default.metallib") finds the latest
	cp $(KERNEL_METALLIB) default.metallib

# Compile source files
$(OBJ_DIR)/%.o: $(SOURCE_DIR)/%.cpp | $(OBJ_DIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Create directories
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

$(BUILD_SENTINEL):
	@mkdir -p $(BUILD_DIR)
	@touch $(BUILD_SENTINEL)

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@echo "Clean complete"

# Install (copy to /usr/local/bin)
.PHONY: install
install: $(TARGET)
	@echo "Installing $(PROJECT_NAME)..."
	@cp $(TARGET) /usr/local/bin/
	@echo "Installation complete"

# Uninstall
.PHONY: uninstall
uninstall:
	@echo "Uninstalling $(PROJECT_NAME)..."
	@rm -f /usr/local/bin/$(PROJECT_NAME)
	@echo "Uninstall complete"

# Run the program (default: all queries with SF-10)
.PHONY: run
run: $(TARGET) $(KERNEL_METALLIB)
	@echo "Running $(PROJECT_NAME)..."
	@cd GPUDBMetalBenchmark && ../$(TARGET)

# Run with different datasets
.PHONY: run-sf1
run-sf1: $(TARGET) $(KERNEL_METALLIB)
	@echo "Running $(PROJECT_NAME) with SF-1 dataset..."
	@cd GPUDBMetalBenchmark && ../$(TARGET) sf1

.PHONY: run-sf10
run-sf10: $(TARGET) $(KERNEL_METALLIB)
	@echo "Running $(PROJECT_NAME) with SF-10 dataset..."
	@cd GPUDBMetalBenchmark && ../$(TARGET) sf10

.PHONY: run-sf100
run-sf100: $(TARGET) $(KERNEL_METALLIB)
	@echo "Running $(PROJECT_NAME) with SF-100 dataset (chunked execution)..."
	@test -d data/SF-100 || (echo "ERROR: data/SF-100 not found. Run: scripts/create_tpch_data.sh 100" && exit 1)
	@cd GPUDBMetalBenchmark && ../$(TARGET) sf100

# Run SF100 individual queries
.PHONY: run-sf100-q1 run-sf100-q3 run-sf100-q6 run-sf100-q9 run-sf100-q13
run-sf100-q1: $(TARGET) $(KERNEL_METALLIB)
	@cd GPUDBMetalBenchmark && ../$(TARGET) sf100 q1
run-sf100-q3: $(TARGET) $(KERNEL_METALLIB)
	@cd GPUDBMetalBenchmark && ../$(TARGET) sf100 q3
run-sf100-q6: $(TARGET) $(KERNEL_METALLIB)
	@cd GPUDBMetalBenchmark && ../$(TARGET) sf100 q6
run-sf100-q9: $(TARGET) $(KERNEL_METALLIB)
	@cd GPUDBMetalBenchmark && ../$(TARGET) sf100 q9
run-sf100-q13: $(TARGET) $(KERNEL_METALLIB)
	@cd GPUDBMetalBenchmark && ../$(TARGET) sf100 q13

# Run TPC-H Query benchmarks individually
.PHONY: run-q1
run-q1: $(TARGET) $(KERNEL_METALLIB)
	@echo "Running TPC-H Query 1 benchmark..."
	@cd GPUDBMetalBenchmark && ../$(TARGET) q1

.PHONY: run-q3
run-q3: $(TARGET) $(KERNEL_METALLIB)
	@echo "Running TPC-H Query 3 benchmark..."
	@cd GPUDBMetalBenchmark && ../$(TARGET) q3

.PHONY: run-q6
run-q6: $(TARGET) $(KERNEL_METALLIB)
	@echo "Running TPC-H Query 6 benchmark..."
	@cd GPUDBMetalBenchmark && ../$(TARGET) q6

.PHONY: run-q9
run-q9: $(TARGET) $(KERNEL_METALLIB)
	@echo "Running TPC-H Query 9 benchmark..."
	@cd GPUDBMetalBenchmark && ../$(TARGET) q9

.PHONY: run-q13
run-q13: $(TARGET) $(KERNEL_METALLIB)
	@echo "Running TPC-H Query 13 benchmark..."
	@cd GPUDBMetalBenchmark && ../$(TARGET) q13

# Run all TPC-H queries
.PHONY: run-all-queries
run-all-queries: $(TARGET) $(KERNEL_METALLIB)
	@echo "Running all TPC-H Query benchmarks..."
	@cd GPUDBMetalBenchmark && ../$(TARGET)

# Check if required files exist
.PHONY: check
check:
	@echo "Checking project structure..."
	@test -f $(SOURCE_DIR)/main.cpp || (echo "ERROR: main.cpp not found in $(SOURCE_DIR)" && exit 1)
	@test -d $(KERNEL_DIR) || (echo "ERROR: Kernels directory not found" && exit 1)
	@test -f $(KERNEL_DIR)/DatabaseKernels.metal || (echo "ERROR: DatabaseKernels.metal not found" && exit 1)
	@test -d $(METAL_CPP_DIR) || (echo "ERROR: metal-cpp directory not found" && exit 1)
	@test -d $(DATA_DIR) || (echo "ERROR: Data directory not found" && exit 1)
	@test -d $(DATA_DIR)/SF-1 || echo "WARNING: SF-1 dataset not found"
	@test -d $(DATA_DIR)/SF-10 || echo "WARNING: SF-10 dataset not found"
	@echo "Checking TPC-H data files..."
	@test -f $(DATA_DIR)/SF-1/lineitem.tbl || echo "WARNING: SF-1 lineitem.tbl not found"
	@test -f $(DATA_DIR)/SF-1/orders.tbl || echo "WARNING: SF-1 orders.tbl not found"
	@test -f $(DATA_DIR)/SF-1/customer.tbl || echo "WARNING: SF-1 customer.tbl not found"
	@test -f $(DATA_DIR)/SF-1/part.tbl || echo "WARNING: SF-1 part.tbl not found"
	@test -f $(DATA_DIR)/SF-1/supplier.tbl || echo "WARNING: SF-1 supplier.tbl not found"
	@test -f $(DATA_DIR)/SF-1/partsupp.tbl || echo "WARNING: SF-1 partsupp.tbl not found"
	@test -f $(DATA_DIR)/SF-1/nation.tbl || echo "WARNING: SF-1 nation.tbl not found"
	@test -f $(DATA_DIR)/SF-10/lineitem.tbl || echo "WARNING: SF-10 lineitem.tbl not found"
	@test -f $(DATA_DIR)/SF-10/orders.tbl || echo "WARNING: SF-10 orders.tbl not found"
	@test -f $(DATA_DIR)/SF-10/customer.tbl || echo "WARNING: SF-10 customer.tbl not found"
	@echo "Project structure check complete"

# Show help
.PHONY: help
help:
	@echo "GPU Database Mental Benchmark - Makefile Help"
	@echo "=============================================="
	@echo ""
	@echo "Available targets:"
	@echo "  run-sf1           - Build and run with SF-1 dataset"
	@echo "  run-sf10          - Build and run with SF-10 dataset"
	@echo "  run-sf100         - Build and run with SF-100 dataset (chunked)"
	@echo "  run-sf100-q{N}    - Run individual SF-100 query (q1,q3,q6,q9,q13)"
	@echo "  run-q1            - Run TPC-H Query 1 benchmark only"
	@echo "  run-q3            - Run TPC-H Query 3 benchmark only"
	@echo "  run-q6            - Run TPC-H Query 6 benchmark only"
	@echo "  run-q9            - Run TPC-H Query 9 benchmark only"
	@echo "  run-q13           - Run TPC-H Query 13 benchmark only"
	@echo "  clean             - Remove all build artifacts"
	@echo "  check             - Verify project structure"
	@echo "  help              - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make run-q1       # Run only TPC-H Query 1"
	@echo "  make run-q3       # Run only TPC-H Query 3"
	@echo "  make clean        # Clean build files"
	@echo "  make check        # Verify all files exist"

# Show project info
.PHONY: info
info:
	@echo "Project: $(PROJECT_NAME)"
	@echo "Source Directory: $(SOURCE_DIR)"
	@echo "Kernel Directory: $(KERNEL_DIR)"
	@echo "Metal-CPP Directory: $(METAL_CPP_DIR)"
	@echo "Data Directory: $(DATA_DIR)"
	@echo "Build Directory: $(BUILD_DIR)"
	@echo "Compiler: $(CXX)"
	@echo "C++ Standard: C++20"
	@echo "Frameworks: Metal, Foundation, QuartzCore"
	@echo ""
	@echo "Supported TPC-H Queries:"
	@echo "  Q1  - Pricing Summary Report Query"
	@echo "  Q3  - Shipping Priority Query"
	@echo "  Q6  - Forecasting Revenue Change Query"
	@echo "  Q9  - Product Type Profit Measure Query"
	@echo "  Q13 - Customer Distribution Query"
	@echo ""
	@echo "Available datasets: SF-1, SF-10, SF-100"

# Print variables (for debugging the Makefile)
.PHONY: print-vars
print-vars:
	@echo "SOURCES: $(SOURCES)"
	@echo "OBJECTS: $(OBJECTS)"
	@echo "KERNELS: $(KERNELS)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "INCLUDES: $(INCLUDES)"
	@echo "FRAMEWORKS: $(FRAMEWORKS)"

# Force rebuild
.PHONY: rebuild
rebuild: clean all

# Compile only (no linking)
.PHONY: compile
compile: $(OBJECTS)
	@echo "Compilation complete"

# Quick test build (just compile, don't run)
.PHONY: test-build
test-build: compile
	@echo "Test build successful - source compiles without errors"

# Build target (kept for compatibility)
# Include the Metal library so kernel changes are picked up by `make build`.
.PHONY: build
build: $(TARGET) $(KERNEL_METALLIB)

# Create a distributable package
.PHONY: package
package: $(TARGET)
	@echo "Creating distribution package..."
	@mkdir -p dist/$(PROJECT_NAME)
	@cp $(TARGET) dist/$(PROJECT_NAME)/
	@cp -r $(KERNEL_DIR) dist/$(PROJECT_NAME)/
	@cp README.md dist/$(PROJECT_NAME)/ 2>/dev/null || true
	@cp *.md dist/$(PROJECT_NAME)/ 2>/dev/null || true
	@echo "Package created in dist/$(PROJECT_NAME)/"