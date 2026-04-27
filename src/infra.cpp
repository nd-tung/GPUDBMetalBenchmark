// Metal-cpp private implementation — must be in exactly ONE translation unit
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "infra.h"

// Global dataset configuration — definitions
std::string g_dataset_path = "data/SF-1/"; // Default to SF-1
bool g_sf100_mode = false; // true when running SF100 chunked execution

// Current query / scale-factor labels (set by main.cpp before each benchmark).
// Consumed by printTimingSummary() to emit a machine-readable TIMING_CSV line.
std::string g_current_query = "";
std::string g_current_sf    = "SF-1";

// Detailed timing accumulator — reset by main.cpp before each benchmark runs.
// createPipeline() (and opt-in helpers) add their elapsed time here, so
// printTimingSummary() can produce a per-stage breakdown without requiring
// every query to thread a timing pointer through its code.
DetailedTiming g_detailed_timing;
