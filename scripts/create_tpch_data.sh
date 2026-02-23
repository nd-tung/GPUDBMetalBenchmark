#!/bin/bash

# TPC-H Data Generation Script for GPU Database Metal Benchmark
# This script downloads, compiles, and runs the TPC-H dbgen tool to generate benchmark data

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"
TEMP_DIR="$PROJECT_DIR/temp_tpch"
DBGEN_URL="https://github.com/electrum/tpch-dbgen.git"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check for required tools
    local missing_tools=()
    
    if ! command_exists git; then
        missing_tools+=("git")
    fi
    
    if ! command_exists make; then
        missing_tools+=("make") 
    fi
    
    if ! command_exists gcc; then
        if ! command_exists clang; then
            missing_tools+=("gcc or clang")
        fi
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install the missing tools and run this script again."
        log_info "On macOS: brew install git make"
        exit 1
    fi
    
    log_success "All required tools are available"
}

# Function to create directory structure
create_directories() {
    log_info "Creating data directory structure..."
    
    mkdir -p "$DATA_DIR/SF-1"
    mkdir -p "$DATA_DIR/SF-10"
    mkdir -p "$DATA_DIR/SF-100"
    mkdir -p "$TEMP_DIR"
    
    log_success "Directory structure created"
}

# Function to download and compile dbgen
setup_dbgen() {
    log_info "Setting up TPC-H dbgen tool..."
    
    cd "$TEMP_DIR"
    
    # Clone the TPC-H dbgen if not already present
    if [ ! -d "tpch-dbgen" ]; then
        log_info "Downloading TPC-H dbgen..."
        git clone "$DBGEN_URL" tpch-dbgen
    else
        log_info "TPC-H dbgen already downloaded"
    fi
    
    cd tpch-dbgen
    
    # The electrum/tpch-dbgen repo has a simpler build process
    log_info "Building dbgen..."
    
    # Simply run make - this repo is already configured for modern systems
    make clean || true  # Clean any previous builds (ignore errors if no previous build)
    make
    
    # Check if dbgen was built successfully
    if [ ! -f "dbgen" ]; then
        log_error "Failed to build dbgen tool"
        exit 1
    fi
    
    log_success "TPC-H dbgen tool compiled successfully"
}

# Function to generate data for a specific scale factor
generate_data() {
    local scale_factor=$1
    local output_dir="$DATA_DIR/SF-$scale_factor"
    
    log_info "Generating TPC-H data for scale factor $scale_factor..."
    
    cd "$TEMP_DIR/tpch-dbgen"
    
    # Generate the data
    log_info "Running dbgen for SF-$scale_factor (this may take a while)..."
    ./dbgen -s "$scale_factor"
    
    # Move generated files to the appropriate directory
    log_info "Moving generated files to $output_dir..."
    
    # List of TPC-H table files
    local tables=("customer.tbl" "lineitem.tbl" "nation.tbl" "orders.tbl" "part.tbl" "partsupp.tbl" "region.tbl" "supplier.tbl")
    
    for table in "${tables[@]}"; do
        if [ -f "$table" ]; then
            mv "$table" "$output_dir/"
            log_info "  ✓ $table"
        else
            log_warning "  ✗ $table not found"
        fi
    done
    
    log_success "Data generation complete for SF-$scale_factor"
}

# Function to verify generated data
verify_data() {
    local scale_factor=$1
    local data_dir="$DATA_DIR/SF-$scale_factor"
    
    log_info "Verifying data for SF-$scale_factor..."
    
    local tables=("customer.tbl" "lineitem.tbl" "nation.tbl" "orders.tbl" "part.tbl" "partsupp.tbl" "region.tbl" "supplier.tbl")
    local missing_files=()
    
    for table in "${tables[@]}"; do
        local file_path="$data_dir/$table"
        if [ -f "$file_path" ]; then
            local file_size=$(du -h "$file_path" | cut -f1)
            local line_count=$(wc -l < "$file_path")
            log_info "  ✓ $table ($file_size, $line_count lines)"
        else
            missing_files+=("$table")
        fi
    done
    
    if [ ${#missing_files[@]} -eq 0 ]; then
        log_success "All TPC-H tables present for SF-$scale_factor"
    else
        log_error "Missing files for SF-$scale_factor: ${missing_files[*]}"
        return 1
    fi
}

# Function to cleanup temporary files
cleanup() {
    log_info "Cleaning up temporary files..."
    if [ -d "$TEMP_DIR" ] && [ "$1" != "--keep-temp" ]; then
        rm -rf "$TEMP_DIR"
        log_success "Temporary files cleaned up"
    else
        log_info "Temporary files kept at: $TEMP_DIR"
    fi
}

# Function to show disk usage
show_disk_usage() {
    log_info "Data directory disk usage:"
    if [ -d "$DATA_DIR" ]; then
        du -sh "$DATA_DIR"/* 2>/dev/null || true
        echo
        du -sh "$DATA_DIR" 2>/dev/null || true
    fi
}

# Function to show usage
show_help() {
    echo "TPC-H Data Generation Script"
    echo "Usage: $0 [OPTIONS] [SCALE_FACTORS...]"
    echo
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -c, --check         Check existing data without generating"
    echo "  -k, --keep-temp     Keep temporary files after generation"
    echo "  --clean             Remove all generated data"
    echo "  --clean-temp        Remove only temporary files"
    echo
    echo "Scale Factors:"
    echo "  If no scale factors are specified, both SF-1 and SF-10 will be generated"
    echo "  Examples: $0 1 10"
    echo "           $0 --check"
    echo "           $0 --clean"
    echo
    echo "Common scale factors:"
    echo "  1   - ~1GB dataset (good for testing)"
    echo "  10  - ~10GB dataset (standard benchmark)"
    echo "  100 - ~100GB dataset (large benchmark)"
}

# Main execution function
main() {
    local scale_factors=()
    local check_only=false
    local keep_temp=false
    local clean_data=false
    local clean_temp=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--check)
                check_only=true
                shift
                ;;
            -k|--keep-temp)
                keep_temp=true
                shift
                ;;
            --clean)
                clean_data=true
                shift
                ;;
            --clean-temp)
                clean_temp=true
                shift
                ;;
            [0-9]*)
                scale_factors+=("$1")
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Handle cleanup options
    if [ "$clean_temp" = true ]; then
        cleanup --keep-temp
        exit 0
    fi
    
    if [ "$clean_data" = true ]; then
        log_warning "This will remove all generated TPC-H data!"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$DATA_DIR"
            cleanup
            log_success "All data removed"
        else
            log_info "Operation cancelled"
        fi
        exit 0
    fi
    
    # If no scale factors specified, use default
    if [ ${#scale_factors[@]} -eq 0 ]; then
        scale_factors=(1 10)
    fi
    
    log_info "TPC-H Data Generation Script starting..."
    log_info "Project directory: $PROJECT_DIR"
    log_info "Data directory: $DATA_DIR"
    log_info "Scale factors: ${scale_factors[*]}"
    
    # Check only mode
    if [ "$check_only" = true ]; then
        log_info "Checking existing data..."
        create_directories
        for sf in "${scale_factors[@]}"; do
            verify_data "$sf" || true
        done
        show_disk_usage
        exit 0
    fi
    
    # Full generation process
    check_requirements
    create_directories
    setup_dbgen
    
    # Generate data for each scale factor
    for sf in "${scale_factors[@]}"; do
        generate_data "$sf"
        verify_data "$sf"
    done
    
    # Show results
    show_disk_usage
    
    # Cleanup
    if [ "$keep_temp" = true ]; then
        cleanup --keep-temp
    else
        cleanup
    fi
    
    log_success "TPC-H data generation completed successfully!"
    log_info "Generated data is available in: $DATA_DIR"
    log_info "You can now run: make check"
}

# Trap to cleanup on script exit
trap 'cleanup' EXIT

# Run main function with all arguments
main "$@"