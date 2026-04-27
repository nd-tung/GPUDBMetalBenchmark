#!/bin/bash
# GPU Metal TPC-H Benchmark Script
# Runs Q1, Q3, Q6, Q9, Q13 and records timing metrics to CSV
# Usage: benchmark_gpu.sh [--query-results] [sf1|sf10|sf100|all]
#   --query-results: show live GPU output and save per-query logs

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
parse_common_args "$@"

BUILD_BIN="$PROJECT_ROOT/build/bin/GPUDBMetalBenchmark"
LOG_DIR="$RESULTS_DIR/gpu_logs"
GPU_CSV="$RESULTS_DIR/gpu_results.csv"

QUERIES="Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 Q10 Q11 Q12 Q13 Q14 Q15 Q16 Q17 Q18 Q19 Q20 Q21 Q22"

if [[ "$SHOW_QUERY_RESULTS" -eq 1 ]]; then
    echo "=== GPU Metal TPC-H Benchmark with Results Export ==="
    echo "Binary: ${BUILD_BIN}"
    echo "Results CSV: ${GPU_CSV}"
    echo "Timestamp: ${TIMESTAMP}"
    echo ""
fi

mkdir -p "$RESULTS_DIR" "${LOG_DIR}/${TIMESTAMP}"

if [[ ! -x "$BUILD_BIN" ]]; then
  echo "Error: GPUDBMetalBenchmark binary not found at $BUILD_BIN" >&2
  echo "Run 'make build' first."
  exit 1
fi

# CSV header (now includes cpu_parse_ms, pso_ms, buffer_alloc_ms — matches the
# machine-readable TIMING_DETAIL_CSV line emitted by printTimingSummary in
# src/infra.h).
GPU_CSV_HEADER="timestamp,scale_factor,query,cpu_parse_ms,pso_ms,buffer_alloc_ms,gpu_exec_ms,cpu_post_ms,total_exec_ms,gpu_name,memory_gb"

# If an older CSV exists with a different schema, rotate it out of the way so
# we don't append rows that no longer match the header.
if [[ -f "$GPU_CSV" ]]; then
  existing_header=$(head -1 "$GPU_CSV")
  if [[ "$existing_header" != "$GPU_CSV_HEADER" ]]; then
    backup="${GPU_CSV%.csv}.pre_${TIMESTAMP}.csv"
    mv "$GPU_CSV" "$backup"
    echo "Note: rotated legacy CSV to $backup (schema changed, added cpu_parse_ms)"
  fi
fi
ensure_csv "$GPU_CSV" "$GPU_CSV_HEADER"

# ---------------------------------------------------------------------------
# Timing extraction
# ---------------------------------------------------------------------------
# Fallback label-based extractor (used if the binary ever omits a TIMING_CSV line).
extract_metric() {
  local block="$1"
  local label="$2"
  echo "$block" | grep -F "$label:" | head -1 \
    | grep -oE '[0-9]+\.[0-9]+[[:space:]]+ms' | head -1 | awk '{print $1}'
}

# Extract the i-th numeric field from the last TIMING_CSV line for a given query.
# TIMING_CSV,<sf>,<query>,<cpu_parse>,<gpu>,<cpu_post>,<total>
extract_timing_csv_field() {
  local block="$1"
  local q="$2"
  local field="$3"   # 4=cpu_parse 5=gpu 6=cpu_post 7=total
  echo "$block" | awk -F, -v q="$q" -v f="$field" '
    $1 == "TIMING_CSV" && toupper($3) == toupper(q) { val = $f }
    END { if (val != "") print val }
  '
}

# ---------------------------------------------------------------------------
# Parse benchmark output → CSV rows
# ---------------------------------------------------------------------------
parse_and_record() {
  local sf_label="$1"
  local out_file="$2"

  for q in $QUERIES; do
    local header_line
    header_line=$(grep -E "^(SF100 )?${q} \|" "$out_file" || true)
    [[ -z "$header_line" ]] && continue

    local block
    block=$(awk "/^(SF100 )?${q} \|/,/Total Execution:/" "$out_file")

    # Prefer the machine-readable TIMING_DETAIL_CSV line (9 fields, with
    # pso/buffer_alloc). Fall back to TIMING_CSV (7 fields) if detail is
    # absent, then to the pretty-printed summary as a last resort.
    local q_lower
    q_lower=$(echo "$q" | tr '[:upper:]' '[:lower:]')
    local detail_line timing_csv_line
    detail_line=$(grep -E "^TIMING_DETAIL_CSV,[^,]*,${q_lower}," "$out_file" | tail -1 || true)
    timing_csv_line=$(grep -E "^TIMING_CSV,[^,]*,${q_lower}," "$out_file" | tail -1 || true)

    local cpu_parse pso_ms buf_alloc gpu_exec cpu_post total_exec
    if [[ -n "$detail_line" ]]; then
      # TIMING_DETAIL_CSV,sf,query,parse,pso,bufalloc,gpu,post,total
      cpu_parse=$( echo "$detail_line" | awk -F, '{print $4}')
      pso_ms=$(    echo "$detail_line" | awk -F, '{print $5}')
      buf_alloc=$( echo "$detail_line" | awk -F, '{print $6}')
      gpu_exec=$(  echo "$detail_line" | awk -F, '{print $7}')
      cpu_post=$(  echo "$detail_line" | awk -F, '{print $8}')
      total_exec=$(echo "$detail_line" | awk -F, '{print $9}')
    elif [[ -n "$timing_csv_line" ]]; then
      cpu_parse=$( echo "$timing_csv_line" | awk -F, '{print $4}')
      pso_ms="0"
      buf_alloc="0"
      gpu_exec=$(  echo "$timing_csv_line" | awk -F, '{print $5}')
      cpu_post=$(  echo "$timing_csv_line" | awk -F, '{print $6}')
      total_exec=$(echo "$timing_csv_line" | awk -F, '{print $7}')
    else
      # Fallback: scrape from pretty-printed summary.
      cpu_parse=$(extract_metric "$block" "CPU Parsing (.tbl)")
      pso_ms="0"
      buf_alloc="0"
      gpu_exec=$( extract_metric "$block" "GPU Execution")
      cpu_post=$( extract_metric "$block" "CPU Post Process")
      total_exec=$(extract_metric "$block" "Total Execution")
    fi

    if [[ -n "$gpu_exec" ]]; then
      echo "$TIMESTAMP,$sf_label,$q,${cpu_parse:-0},${pso_ms:-0},${buf_alloc:-0},${gpu_exec},${cpu_post:-0},${total_exec:-0},${GPU_NAME},${MEMORY_GB}" >> "$GPU_CSV"
      printf "  %-4s parse=%-10s pso=%-9s buf=%-9s gpu=%-10s post=%-9s total=%s ms\n" \
        "$q" "${cpu_parse:-0} ms" "${pso_ms:-0} ms" "${buf_alloc:-0} ms" \
        "${gpu_exec} ms" "${cpu_post:-0} ms" "${total_exec:-0}"
    fi

    # Save per-query log
    local q_log="${LOG_DIR}/${TIMESTAMP}/${sf_label}_${q}.log"
    local q_num="${q#Q}"
    {
      echo "=== GPU Metal ${q} Results ==="
      echo "Timestamp: ${TIMESTAMP}"
      echo "Scale Factor: ${sf_label}"
      echo ""
      awk "/Running.*Query ${q_num}/,/Total Execution:/" "$out_file" 2>/dev/null ||
        awk "/Running.*${q}/,/Total Execution:/" "$out_file" 2>/dev/null || true
    } > "$q_log"
    [[ "$SHOW_QUERY_RESULTS" -eq 1 ]] && [[ -s "$q_log" ]] && echo "    → ${q_log}"
  done
}

# ---------------------------------------------------------------------------
# Run benchmark for one scale factor
# ---------------------------------------------------------------------------
run_benchmark() {
  local sf_arg="$1"
  local sf_label="$2"

  echo "=== GPU benchmark: ${sf_label} ==="
  local out_file="${LOG_DIR}/${TIMESTAMP}/${sf_label}_full.log"

  if [[ "$SHOW_QUERY_RESULTS" -eq 1 ]]; then
    echo "Running GPU benchmarks for ${sf_label}..."
    (cd "$PROJECT_ROOT" && "$BUILD_BIN" "$sf_arg" all) 2>&1 | tee "$out_file"
  else
    (cd "$PROJECT_ROOT" && "$BUILD_BIN" "$sf_arg" all) > "$out_file" 2>&1 || true
  fi

  echo ""
  [[ "$SHOW_QUERY_RESULTS" -eq 1 ]] && echo "Extracting results for ${sf_label}..."
  parse_and_record "$sf_label" "$out_file"
  echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
MODE="${POSITIONAL_ARGS[0]:-all}"

case "$MODE" in
  sf1|SF-1)     run_benchmark sf1  SF-1   ;;
  sf10|SF-10)   run_benchmark sf10 SF-10  ;;
  sf100|SF-100) run_benchmark sf100 SF-100 ;;
  all)
    [[ -d "$PROJECT_ROOT/data/SF-1" ]]   && run_benchmark sf1  SF-1
    [[ -d "$PROJECT_ROOT/data/SF-10" ]]  && run_benchmark sf10 SF-10  || true
    [[ -d "$PROJECT_ROOT/data/SF-100" ]] && run_benchmark sf100 SF-100 || true
    ;;
  *)
    echo "Usage: $0 [--query-results] [sf1|sf10|sf100|all]"
    exit 1
    ;;
esac

echo "Results saved to: $GPU_CSV"
echo "Logs: ${LOG_DIR}/${TIMESTAMP}/"
echo ""
echo "Latest results (${TIMESTAMP}):"
grep "^${TIMESTAMP}" "$GPU_CSV" | column -t -s,

if [[ "$SHOW_QUERY_RESULTS" -eq 1 ]]; then
  echo ""
  echo "Log files:"
  ls -lh "${LOG_DIR}/${TIMESTAMP}/"
fi
