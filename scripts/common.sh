#!/bin/bash
# common.sh — Shared utilities for benchmark scripts
# Source this file: source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_ROOT/results"
DATA_DIR="$PROJECT_ROOT/data"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Detect GPU name and system memory
GPU_NAME=$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Chipset Model/{print $2}' | head -1 | xargs)
MEMORY_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))

# ---------------------------------------------------------------------------
# Parse --query-results flag from arguments
# Sets: SHOW_QUERY_RESULTS=1 if present, remaining args in POSITIONAL_ARGS
# ---------------------------------------------------------------------------
SHOW_QUERY_RESULTS=0
POSITIONAL_ARGS=()
parse_common_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --query-results) SHOW_QUERY_RESULTS=1; shift ;;
            *) POSITIONAL_ARGS+=("$1"); shift ;;
        esac
    done
}

# ---------------------------------------------------------------------------
# Normalize scale factor labels
# Maps: sf1→SF-1, sf10→SF-10, sf100→SF-100, SF-1→SF-1 (passthrough)
# ---------------------------------------------------------------------------
normalize_sf() {
    local input="$1"
    case "$input" in
        sf1|SF-1)   echo "SF-1"   ;;
        sf10|SF-10)  echo "SF-10"  ;;
        sf100|SF-100) echo "SF-100" ;;
        *) echo "$input" ;;
    esac
}

# Map normalized SF label to binary arg: SF-1→sf1
sf_to_bin_arg() {
    local sf="$1"
    case "$sf" in
        SF-1)   echo "sf1"   ;;
        SF-10)  echo "sf10"  ;;
        SF-100) echo "sf100" ;;
        *) echo "$sf" ;;
    esac
}

# ---------------------------------------------------------------------------
# Round a timing value to integer milliseconds
# ---------------------------------------------------------------------------
round_ms() {
    local raw="$1"
    raw="$(echo "$raw" | head -n 1 | tr -d '[:space:],')"
    if [ -z "$raw" ]; then
        echo "0"
        return 0
    fi
    awk -v t="$raw" 'BEGIN{ if(t=="" || t==".") {print 0} else {printf "%.0f", t} }'
}

# ---------------------------------------------------------------------------
# Ensure CSV file exists with header
# ---------------------------------------------------------------------------
ensure_csv() {
    local csv_file="$1"
    local header="$2"
    mkdir -p "$(dirname "$csv_file")"
    if [[ ! -f "$csv_file" ]]; then
        echo "$header" > "$csv_file"
    fi
}
