#pragma once
// ===================================================================
// Metal Query Plan Builder — converts AnalyzedQuery → operator trees
// ===================================================================
//
// Produces MetalQueryPlan containing per-phase operator trees that
// can be fed to MetalCodegen for Metal shader generation.
//
// This sits alongside the existing query_planner.h (which produces
// flat PlanOp vectors). Once all queries are migrated, the old
// planner can be retired.
// ===================================================================

#include "metal_operators.h"
#include "metal_codegen_base.h"
#include "query_analyzer.h"
#include <memory>
#include <string>
#include <vector>
#include <optional>

namespace codegen {

struct MetalQueryPlan {
    std::string name;  // "Q1", "Q6", etc.

    // Each phase is one Metal kernel
    struct Phase {
        std::string name;
        std::unique_ptr<MetalOperator> root;
        int threadgroupSize = 1024;
        bool singleThread = false;
        // Bitmap buffers to register as read-only params (name, sizeExpr)
        // Used when an expression references a bitmap from a prior phase
        // without going through a BitmapProbe operator.
        std::vector<std::pair<std::string, std::string>> bitmapReads;
        // Scalar constant params (name, type) — registered before operator production
        std::vector<std::pair<std::string, std::string>> scalarParams;
        // Extra buffer params not added by operators (e.g., pre-built hash tables)
        struct ExtraBuffer { std::string name; std::string type; bool readOnly = true; };
        std::vector<ExtraBuffer> extraBuffers;
    };
    std::vector<Phase> phases;

    // Helper device functions emitted before all kernels
    std::vector<std::string> helpers;

    // CPU-side post-processing
    struct CpuSort {
        struct SortKey { std::string column; bool descending; };
        std::vector<SortKey> keys;
        int limit = -1;
    };
    std::optional<CpuSort> cpuSort;
};

// Build a MetalQueryPlan from an analyzed query.
// queryName (e.g. "Q7") is used for patterns the analyzer can't match by structure alone.
// Returns nullopt if the query pattern is not yet supported.
std::optional<MetalQueryPlan> buildMetalPlan(const AnalyzedQuery& aq,
                                              const std::string& queryName = "");

// Generate Metal source from a MetalQueryPlan using the operator framework.
// Returns the configured MetalCodegen (with bindings, result schema, etc.)
MetalCodegen generateFromPlan(const MetalQueryPlan& plan);

} // namespace codegen
