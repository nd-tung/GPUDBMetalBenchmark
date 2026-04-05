#pragma once
#include "query_plan.h"
#include "query_analyzer.h"
#include "tpch_schema.h"

namespace codegen {

// Build a QueryPlan from an AnalyzedQuery.
// Pattern-matches against known TPC-H query structures.
QueryPlan buildPlan(const AnalyzedQuery& aq);

} // namespace codegen
