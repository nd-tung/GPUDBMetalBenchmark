# GPU Database Benchmark

GPU-accelerated database operations using Apple Metal vs DuckDB vs CedarDB comparison.

## Latest Benchmark Results

**Timestamp**: 2026-02-01  
**System**: Apple M4 Pro, 14 CPU cores (10 performance + 4 efficiency), 48 GB RAM, 20 GPU cores, macOS Sequoia 15.5, Metal 3  
**Methodology**: Warm cache (data pre-loaded), execution time only

<img width="3600" height="1800" alt="sf1_comparison" src="https://github.com/user-attachments/assets/4e1d9462-6db5-4bcb-bb8d-0113a77d7184" />
<img width="3600" height="1800" alt="sf10_comparison" src="https://github.com/user-attachments/assets/66ab971c-ca04-4c71-b111-3248a50f8819" />

### Metal API Results

#### SF-1 Dataset (6M lineitem rows)
| Query | Execution time (ms) |
|-------|---------------------|
| Q1    | 6.52                |
| Q3    | 1.75                |
| Q6    | 0.86                |
| Q9    | 5.80                |
| Q13   | 3.53                |

#### SF-10 Dataset (60M lineitem rows)
| Query | Execution time (ms) |
|-------|---------------------|
| Q1    | 34.70               |
| Q3    | 13.53               |
| Q6    | 2.12                |
| Q9    | 82.23               |
| Q13   | 21.85               |

### DuckDB Results (SF-1: 20260201_124955, SF-10: 20260201_125031)

#### SF-1 Dataset
| Query | Execution time (ms) |
|-------|---------------------|
| Q1    | 19.12               |
| Q3    | 11.47               |
| Q6    | 4.48                |
| Q9    | 30.33               |
| Q13   | 35.15               |

#### SF-10 Dataset
| Query | Execution time (ms) |
|-------|---------------------|
| Q1    | 153.08              |
| Q3    | 81.7                |
| Q6    | 33.67               |
| Q9    | 253.83              |
| Q13   | 210.57              |

### CedarDB Results (SF-1: 20260201_131951, SF-10: 20260201_132006)

#### SF-1 Dataset
| Query | Execution time (ms) |
|-------|---------------------|
| Q1    | 27.274              |
| Q3    | 10.559              |
| Q6    | 1.043               |
| Q9    | 38.951              |
| Q13   | 33.253              |

#### SF-10 Dataset
| Query | Execution time (ms) |
|-------|---------------------|
| Q1    | 141.963             |
| Q3    | 178.606             |
| Q6    | 5.737               |
| Q9    | 479.747             |
| Q13   | 605.358             |

## How to Run Benchmarks

### Prerequisites
```bash
# Install DuckDB
brew install duckdb

# Install CedarDB via Docker
docker pull cedardb/cedardb
```

### Quick Start
```bash
# 1. Build GPU benchmark
make

# 2. Generate test data (if not already generated)
./scripts/create_tpch_data.sh

# 3. Run all benchmarks
./scripts/benchmark_gpu.sh sf1
./scripts/benchmark_duckdb.sh SF-1
./scripts/benchmark_cedardb.sh SF-1  # Requires Docker

# 4. View results
cat results/gpu_results.csv
cat results/duckdb_results.csv
cat results/cedardb_results.csv
```

### Manual Execution
```bash
# Run individual queries manually
./build/bin/GPUDBMetalBenchmark sf1 q1
./build/bin/GPUDBMetalBenchmark sf10 q13
```

## Benchmark Scripts

The project includes automated benchmark scripts for running comprehensive performance tests:

### Data Generation
```bash
./scripts/create_tpch_data.sh
```
Generates TPC-H benchmark data at different scale factors (SF-1, SF-10). Downloads and compiles the TPC-H dbgen tool, then generates `.tbl` files in `data/`.

### GPU Benchmarks
```bash
./scripts/benchmark_gpu.sh sf1
```
Runs all TPC-H queries (Q1, Q3, Q6, Q9, Q13) on GPU Metal implementation and saves timing results to `results/gpu_results.csv`.

```bash
./scripts/benchmark_gpu_with_query_results.sh sf1
```
Extended version that also saves complete query results to `results/gpu_logs/` for verification.

### DuckDB Benchmarks
```bash
./scripts/benchmark_duckdb.sh SF-1
```
Runs all queries on DuckDB and saves results to `results/duckdb_results.csv`.

```bash
./scripts/benchmark_duckdb_with_query_results.sh SF-1
```
Extended version with query result export to `results/duckdb_logs/`.

### CedarDB Benchmarks
```bash
# Start CedarDB container first
docker run -d --name cedardb -p 5432:5432 -e CEDAR_PASSWORD=cedar --memory=12g cedardb/cedardb

# Run benchmarks
./scripts/benchmark_cedardb.sh SF-1
```
Runs queries on CedarDB via Docker and saves to `results/cedardb_results.csv`.

```bash
./scripts/benchmark_cedardb_with_query_results.sh SF-1
```
Extended version with query result logging.

## Benchmark Details

- **TPC-H Queries**: Q1 (Pricing Summary), Q3 (Shipping Priority), Q6 (Revenue Forecasting), Q9 (Product Profit), Q13 (Customer Distribution)
- **Data Format**: TPC-H standard `.tbl` files
- **Cache Strategy**: Warm cache (data pre-loaded, queries run on hot cache)
- **Timing Method**: Execution time only (excludes I/O and data loading)



