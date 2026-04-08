#include "metal_plan_builder.h"
#include "tpch_schema.h"
#include <sstream>
#include <algorithm>
#include <iostream>
#include <set>

namespace codegen {

// ===================================================================
// Helper: expression to Metal code string (columnar access: col[idx])
// ===================================================================

namespace {

// Collect all column references in an expression
void collectColumns(const ExprPtr& expr, std::set<std::string>& cols) {
    if (!expr) return;
    std::visit([&](auto&& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, ColRef>) {
            cols.insert(node.column);
        } else if constexpr (std::is_same_v<T, BinaryExpr>) {
            collectColumns(node.left, cols);
            collectColumns(node.right, cols);
        } else if constexpr (std::is_same_v<T, FuncCall>) {
            for (auto& a : node.args) collectColumns(a, cols);
        } else if constexpr (std::is_same_v<T, CaseWhen>) {
            for (auto& b : node.branches) {
                collectColumns(b.result, cols);
            }
            if (node.elseResult) collectColumns(node.elseResult, cols);
        }
    }, expr->node);
}

void collectColumns(const PredPtr& pred, std::set<std::string>& cols) {
    if (!pred) return;
    std::visit([&](auto&& node) {
        using T = std::decay_t<decltype(node)>;
        if constexpr (std::is_same_v<T, Comparison>) {
            collectColumns(node.left, cols);
            collectColumns(node.right, cols);
        } else if constexpr (std::is_same_v<T, Between>) {
            collectColumns(node.expr, cols);
            collectColumns(node.low, cols);
            collectColumns(node.high, cols);
        } else if constexpr (std::is_same_v<T, InList>) {
            collectColumns(node.expr, cols);
            for (auto& v : node.values) collectColumns(v, cols);
        } else if constexpr (std::is_same_v<T, LogicalAnd>) {
            for (auto& c : node.children) collectColumns(c, cols);
        } else if constexpr (std::is_same_v<T, LogicalOr>) {
            for (auto& c : node.children) collectColumns(c, cols);
        } else if constexpr (std::is_same_v<T, LogicalNot>) {
            collectColumns(node.child, cols);
        } else if constexpr (std::is_same_v<T, Like>) {
            collectColumns(node.expr, cols);
        }
    }, pred->node);
}

std::string exprToMetal(const ExprPtr& expr, const std::string& idxVar) {
    if (!expr) return "";

    return std::visit([&](auto&& node) -> std::string {
        using T = std::decay_t<decltype(node)>;

        if constexpr (std::is_same_v<T, ColRef>) {
            // Columnar access: column_name[idx]
            return node.column + "[" + idxVar + "]";
        }
        else if constexpr (std::is_same_v<T, Literal>) {
            return std::visit([](auto&& v) -> std::string {
                using V = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<V, int>)
                    return std::to_string(v);
                else if constexpr (std::is_same_v<V, float>)
                    return std::to_string(v) + "f";
                else
                    return "\"" + v + "\"";
            }, node.value);
        }
        else if constexpr (std::is_same_v<T, BinaryExpr>) {
            std::string l = exprToMetal(node.left, idxVar);
            std::string r = exprToMetal(node.right, idxVar);
            switch (node.op) {
                case ExprOp::ADD: return "(" + l + " + " + r + ")";
                case ExprOp::SUB: return "(" + l + " - " + r + ")";
                case ExprOp::MUL: return "(" + l + " * " + r + ")";
                case ExprOp::DIV: return "(" + l + " / " + r + ")";
            }
            return l;
        }
        else if constexpr (std::is_same_v<T, FuncCall>) {
            std::ostringstream os;
            os << node.name << "(";
            for (size_t i = 0; i < node.args.size(); i++) {
                if (i) os << ", ";
                os << exprToMetal(node.args[i], idxVar);
            }
            os << ")";
            return os.str();
        }
        else if constexpr (std::is_same_v<T, CaseWhen>) {
            if (!node.branches.empty())
                return exprToMetal(node.branches[0].result, idxVar);
            if (node.elseResult)
                return exprToMetal(node.elseResult, idxVar);
            return "0";
        }
        else {
            return "/* unknown expr */";
        }
    }, expr->node);
}

std::string predToMetal(const PredPtr& pred, const std::string& idxVar) {
    if (!pred) return "true";

    return std::visit([&](auto&& node) -> std::string {
        using T = std::decay_t<decltype(node)>;

        if constexpr (std::is_same_v<T, Comparison>) {
            std::string l = exprToMetal(node.left, idxVar);
            std::string r = exprToMetal(node.right, idxVar);
            switch (node.op) {
                case CmpOp::EQ: return l + " == " + r;
                case CmpOp::NE: return l + " != " + r;
                case CmpOp::LT: return l + " < " + r;
                case CmpOp::LE: return l + " <= " + r;
                case CmpOp::GT: return l + " > " + r;
                case CmpOp::GE: return l + " >= " + r;
            }
            return l + " == " + r;
        }
        else if constexpr (std::is_same_v<T, Between>) {
            std::string e = exprToMetal(node.expr, idxVar);
            std::string lo = exprToMetal(node.low, idxVar);
            std::string hi = exprToMetal(node.high, idxVar);
            return "(" + e + " >= " + lo + " && " + e + " <= " + hi + ")";
        }
        else if constexpr (std::is_same_v<T, InList>) {
            std::string e = exprToMetal(node.expr, idxVar);
            std::string cond;
            for (size_t i = 0; i < node.values.size(); i++) {
                if (i) cond += " || ";
                cond += e + " == " + exprToMetal(node.values[i], idxVar);
            }
            return "(" + cond + ")";
        }
        else if constexpr (std::is_same_v<T, LogicalAnd>) {
            std::string cond;
            for (size_t i = 0; i < node.children.size(); i++) {
                if (i) cond += " && ";
                cond += "(" + predToMetal(node.children[i], idxVar) + ")";
            }
            return cond;
        }
        else if constexpr (std::is_same_v<T, LogicalOr>) {
            std::string cond;
            for (size_t i = 0; i < node.children.size(); i++) {
                if (i) cond += " || ";
                cond += "(" + predToMetal(node.children[i], idxVar) + ")";
            }
            return "(" + cond + ")";
        }
        else if constexpr (std::is_same_v<T, LogicalNot>) {
            return "!(" + predToMetal(node.child, idxVar) + ")";
        }
        else if constexpr (std::is_same_v<T, Like>) {
            return "/* LIKE not directly translatable */true";
        }
        else if constexpr (std::is_same_v<T, ExistsPred>) {
            return "/* EXISTS */true";
        }
        else {
            return "true";
        }
    }, pred->node);
}

// Combine all filter predicates into a single Metal condition string
std::string combineFilters(const std::vector<PredPtr>& filters, const std::string& idxVar) {
    if (filters.empty()) return "";
    if (filters.size() == 1) return predToMetal(filters[0], idxVar);

    std::string cond;
    for (size_t i = 0; i < filters.size(); i++) {
        if (i) cond += " && ";
        cond += "(" + predToMetal(filters[i], idxVar) + ")";
    }
    return cond;
}

// Map column name to Metal type using TPC-H schema
std::string colMetalType(const std::string& table, const std::string& colName) {
    const auto& schema = TPCHSchema::instance();
    auto& tdef = schema.table(table);
    auto& cdef = tdef.col(colName);
    switch (cdef.type) {
        case DataType::INT:        return "int";
        case DataType::FLOAT:      return "float";
        case DataType::DATE:       return "int";
        case DataType::CHAR1:      return "char";
        case DataType::CHAR_FIXED: return "char";
    }
    return "int";
}

} // anonymous namespace

// ===================================================================
// Q6 Plan Builder
// ===================================================================

static std::optional<MetalQueryPlan> buildQ6Plan(const AnalyzedQuery& aq) {
    // Q6: single-table lineitem, SUM aggregate, no GROUP BY
    if (!aq.isSingleTable()) return std::nullopt;
    if (aq.tables[0] != "lineitem") return std::nullopt;
    if (!aq.hasAggregation() || aq.hasGroupBy()) return std::nullopt;

    // Should have exactly 1 SUM aggregate
    bool hasSumAgg = false;
    for (const auto& t : aq.targets) {
        if (t.isAgg && t.agg && t.agg->func == AggFunc::SUM)
            hasSumAgg = true;
    }
    if (!hasSumAgg) return std::nullopt;

    MetalQueryPlan plan;
    plan.name = "Q6";

    // Collect all referenced columns from filters and aggregates
    std::set<std::string> usedCols;
    for (const auto& f : aq.filters) collectColumns(f, usedCols);
    for (const auto& t : aq.targets) {
        if (t.agg && t.agg->innerExpr) collectColumns(t.agg->innerExpr, usedCols);
    }

    // Build the aggregate expression using columnar indexing
    std::string idxVar = "i";
    std::string aggExpr;
    std::string alias = "revenue";
    for (const auto& t : aq.targets) {
        if (t.isAgg && t.agg && t.agg->func == AggFunc::SUM) {
            aggExpr = exprToMetal(t.agg->innerExpr, idxVar);
            if (!t.alias.empty()) alias = t.alias;
        }
    }

    // Build filter predicate using columnar indexing
    std::string filterCond = combineFilters(aq.filters, idxVar);

    // Build operator tree: TGReduce ← Selection ← GridStrideScan
    auto scan = std::make_unique<MetalGridStrideScan>("lineitem", "row", idxVar);
    // Register needed columns
    for (const auto& colName : usedCols) {
        scan->addColumn(colName, colMetalType("lineitem", colName));
    }

    std::unique_ptr<MetalOperator> filtered;
    if (!filterCond.empty()) {
        filtered = std::make_unique<MetalSelection>(std::move(scan), filterCond);
    } else {
        filtered = std::move(scan);
    }

    auto reduce = std::make_unique<MetalTGReduce>(std::move(filtered), "d_" + alias);
    // For Q6, use long (fixed-point 100x) accumulation for precision
    reduce->addAccumulator(alias, "(long)(" + aggExpr + " * 100.0f)", "long");

    // Register result schema: scalar aggregate, 1 column, scale down by 100
    reduce->setResultAlias(alias, 100);

    MetalQueryPlan::Phase phase;
    phase.name = "Q6_reduce";
    phase.root = std::move(reduce);
    phase.threadgroupSize = 1024;
    plan.phases.push_back(std::move(phase));

    return plan;
}

// ===================================================================
// Q1 Plan Builder
// ===================================================================

static std::optional<MetalQueryPlan> buildQ1Plan(const AnalyzedQuery& aq) {
    // Q1: single-table lineitem, GROUP BY l_returnflag, l_linestatus (6 bins)
    if (!aq.isSingleTable()) return std::nullopt;
    if (aq.tables[0] != "lineitem") return std::nullopt;
    if (!aq.hasAggregation() || !aq.hasGroupBy()) return std::nullopt;

    // Check for 2 GROUP BY columns (returnflag + linestatus)
    if (aq.groupBy.size() != 2) return std::nullopt;

    MetalQueryPlan plan;
    plan.name = "Q1";

    std::string idxVar = "i";
    std::string filterCond = combineFilters(aq.filters, idxVar);

    // Collect all columns used in filters, group by, and aggregates
    std::set<std::string> usedCols;
    for (const auto& f : aq.filters) collectColumns(f, usedCols);
    for (const auto& g : aq.groupBy) collectColumns(g, usedCols);
    for (const auto& t : aq.targets) {
        if (t.agg && t.agg->innerExpr) collectColumns(t.agg->innerExpr, usedCols);
    }

    auto scan = std::make_unique<MetalGridStrideScan>("lineitem", "row", idxVar);
    for (const auto& colName : usedCols) {
        scan->addColumn(colName, colMetalType("lineitem", colName));
    }

    std::unique_ptr<MetalOperator> filtered;
    if (!filterCond.empty()) {
        filtered = std::make_unique<MetalSelection>(std::move(scan), filterCond);
    } else {
        filtered = std::move(scan);
    }

    // Q1 uses 6-bin keyed aggregation with columnar access
    std::string bucketExpr = "((l_returnflag[" + idxVar + "] == 'A' ? 0 : (l_returnflag[" + idxVar + "] == 'N' ? 2 : 4)) + (l_linestatus[" + idxVar + "] == 'F' ? 0 : 1))";

    auto agg = std::make_unique<MetalKeyedAgg>(
        std::move(filtered), "d_q1_aggs", bucketExpr,
        /*numBuckets=*/6, /*valuesPerBucket=*/10, "60");

    // Add aggregates using columnar indexing — large sums use lo/hi pairs
    agg->addAggregate("sum_qty", 0, "(uint)(l_quantity[" + idxVar + "] * 100.0f)", "add", true, 100);
    agg->addAggregate("sum_base_price", 2, "(uint)(l_extendedprice[" + idxVar + "] * 100.0f)", "add", true, 100);
    agg->addAggregate("sum_disc_price", 4,
                      "(uint)(l_extendedprice[" + idxVar + "] * (1.0f - l_discount[" + idxVar + "]) * 100.0f)", "add", true, 100);
    agg->addAggregate("sum_charge", 6,
                      "(uint)(l_extendedprice[" + idxVar + "] * (1.0f - l_discount[" + idxVar + "]) * (1.0f + l_tax[" + idxVar + "]) * 100.0f)", "add", true, 100);
    agg->addAggregate("sum_disc", 8, "(uint)(l_discount[" + idxVar + "] * 10000.0f)", "add", false, 0);
    agg->addAggregate("count_order", 9, "1u", "add", false, 0);

    MetalQueryPlan::Phase phase;
    phase.name = "Q1_reduce";
    phase.root = std::move(agg);
    phase.threadgroupSize = 1024;
    plan.phases.push_back(std::move(phase));

    return plan;
}

// ===================================================================
// Q14 Plan Builder — Promotion Effect
// Pattern: BitmapBuild(part, PROMO) → Filter+TGReduce(lineitem, date)
// ===================================================================

static std::optional<MetalQueryPlan> buildQ14Plan(const AnalyzedQuery& aq) {
    // Q14: two tables (lineitem, part), scalar aggregate, no GROUP BY
    if (aq.tables.size() != 2) return std::nullopt;
    bool hasLineitem = false, hasPart = false;
    for (auto& t : aq.tables) {
        if (t == "lineitem") hasLineitem = true;
        if (t == "part") hasPart = true;
    }
    if (!hasLineitem || !hasPart) return std::nullopt;
    // Q14 has a complex expression 100*SUM(CASE...)/SUM(...) as a single target.
    // The analyzer may not flag isAgg since the top-level is BinaryExpr, not FuncCall.
    // Just require: not GROUP BY, has exactly 1 target.
    if (aq.hasGroupBy()) return std::nullopt;
    if (aq.targets.size() != 1) return std::nullopt;

    // lineitem+part, 1 target, no GROUP BY is unique to Q14 in TPC-H

    MetalQueryPlan plan;
    plan.name = "Q14";
    std::string idxVar = "i";

    // Separate filters into lineitem-only and join conditions
    // For Q14: lineitem date filters + join on l_partkey = p_partkey
    // We build a bitmap on p_partkey where p_type starts with "PROMO"
    // Then probe it in the lineitem scan

    // Phase 1: Build promo bitmap from part table
    {
        auto scan = std::make_unique<MetalGridStrideScan>("part", "row", idxVar);
        scan->addColumn("p_partkey", "int");
        scan->addColumn("p_type", "char");

        // Filter: p_type starts with "PROMO" → check first 5 chars
        // p_type is CHAR_FIXED(25), accessed as p_type[i * 25 + offset]
        std::string promoFilter =
            "p_type[" + idxVar + " * 25] == 'P' && "
            "p_type[" + idxVar + " * 25 + 1] == 'R' && "
            "p_type[" + idxVar + " * 25 + 2] == 'O' && "
            "p_type[" + idxVar + " * 25 + 3] == 'M' && "
            "p_type[" + idxVar + " * 25 + 4] == 'O'";

        auto filter = std::make_unique<MetalSelection>(std::move(scan), promoFilter);

        // Build bitmap keyed on p_partkey
        auto bitmapBuild = std::make_unique<MetalBitmapBuild>(
            std::move(filter), "d_promo_bitmap",
            "p_partkey[" + idxVar + "]", "(maxPartkey + 31) / 32 + 1");

        MetalQueryPlan::Phase phase;
        phase.name = "Q14_build_bitmap";
        phase.root = std::move(bitmapBuild);
        phase.threadgroupSize = 1024;
        plan.phases.push_back(std::move(phase));
    }

    // Phase 2: Scan lineitem, filter by date, reduce with promo/total
    {
        auto scan = std::make_unique<MetalGridStrideScan>("lineitem", "row", idxVar);
        scan->addColumn("l_partkey", "int");
        scan->addColumn("l_shipdate", "int");
        scan->addColumn("l_extendedprice", "float");
        scan->addColumn("l_discount", "float");

        // Filter: date range from analyzed query (l_shipdate predicates)
        // Extract date filters that reference l_shipdate
        std::vector<PredPtr> dateFilters;
        PredPtr joinPred;
        for (auto& f : aq.filters) {
            std::set<std::string> cols;
            collectColumns(f, cols);
            if (cols.count("l_partkey") && cols.count("p_partkey")) {
                joinPred = f; // join condition, handled via bitmap
            } else if (cols.count("l_shipdate") || cols.count("l_receiptdate")) {
                dateFilters.push_back(f);
            }
        }

        std::string filterCond = combineFilters(dateFilters, idxVar);
        std::unique_ptr<MetalOperator> filtered;
        if (!filterCond.empty()) {
            filtered = std::make_unique<MetalSelection>(std::move(scan), filterCond);
        } else {
            filtered = std::move(scan);
        }

        // TGReduce with 2 accumulators:
        // total_sum: l_extendedprice * (1 - l_discount) for all qualifying rows
        // promo_sum: same, but only when bitmap_test passes for l_partkey
        std::string revenue = "l_extendedprice[" + idxVar + "] * (1.0f - l_discount[" + idxVar + "] * 0.01f)";
        auto reduce = std::make_unique<MetalTGReduce>(std::move(filtered), "d_q14");
        reduce->addAccumulator("promo",
            "(long)(bitmap_test(d_promo_bitmap, l_partkey[" + idxVar + "]) ? " + revenue + " * 100.0f : 0)", "long");
        reduce->addAccumulator("total", "(long)(" + revenue + " * 100.0f)", "long");
        reduce->setResultAlias("promo_revenue", 100);
        reduce->setResultAlias("total_revenue", 100);

        MetalQueryPlan::Phase phase;
        phase.name = "Q14_reduce";
        phase.root = std::move(reduce);
        phase.threadgroupSize = 1024;
        phase.bitmapReads.push_back({"d_promo_bitmap", ""});
        plan.phases.push_back(std::move(phase));
    }

    return plan;
}

// ===================================================================
// Q4 Plan Builder — Order Priority Checking
// Pattern: BitmapBuild(lineitem, late) → Filter+KeyedAgg(orders, date)
// ===================================================================

static std::optional<MetalQueryPlan> buildQ4Plan(const AnalyzedQuery& aq) {
    // Q4: orders table with GROUP BY o_orderpriority
    // + EXISTS subquery on lineitem (commitdate < receiptdate)
    if (aq.tables.size() < 1) return std::nullopt;
    bool hasOrders = false;
    for (auto& t : aq.tables) if (t == "orders") hasOrders = true;
    if (!hasOrders) return std::nullopt;
    if (!aq.hasGroupBy()) return std::nullopt;

    // Check for EXISTS subquery and o_orderpriority GROUP BY
    bool hasExists = false;
    for (auto& f : aq.filters) {
        std::visit([&](auto&& node) {
            using T = std::decay_t<decltype(node)>;
            if constexpr (std::is_same_v<T, ExistsPred>) hasExists = true;
        }, f->node);
    }
    if (!hasExists) return std::nullopt;

    // Verify GROUP BY o_orderpriority
    bool groupByPriority = false;
    for (auto& g : aq.groupBy) {
        std::visit([&](auto&& node) {
            using T = std::decay_t<decltype(node)>;
            if constexpr (std::is_same_v<T, ColRef>) {
                if (node.column == "o_orderpriority") groupByPriority = true;
            }
        }, g->node);
    }
    if (!groupByPriority) return std::nullopt;

    MetalQueryPlan plan;
    plan.name = "Q4";
    std::string idxVar = "i";

    // Phase 1: Build late-delivery bitmap from lineitem
    // Set bit for l_orderkey where l_commitdate < l_receiptdate
    {
        auto scan = std::make_unique<MetalGridStrideScan>("lineitem", "row", idxVar);
        scan->addColumn("l_orderkey", "int");
        scan->addColumn("l_commitdate", "int");
        scan->addColumn("l_receiptdate", "int");

        auto filter = std::make_unique<MetalSelection>(std::move(scan),
            "l_commitdate[" + idxVar + "] < l_receiptdate[" + idxVar + "]");

        auto bitmapBuild = std::make_unique<MetalBitmapBuild>(
            std::move(filter), "d_late_bitmap",
            "l_orderkey[" + idxVar + "]", "(maxOrderkey + 31) / 32 + 1");

        MetalQueryPlan::Phase phase;
        phase.name = "Q4_build_bitmap";
        phase.root = std::move(bitmapBuild);
        phase.threadgroupSize = 1024;
        plan.phases.push_back(std::move(phase));
    }

    // Phase 2: Scan orders, filter by date + bitmap probe, count by priority
    {
        auto scan = std::make_unique<MetalGridStrideScan>("orders", "row", idxVar);
        scan->addColumn("o_orderkey", "int");
        scan->addColumn("o_orderdate", "int");
        scan->addColumn("o_orderpriority", "char");

        // Extract date filter from the analyzed query
        std::vector<PredPtr> dateFilters;
        for (auto& f : aq.filters) {
            std::set<std::string> cols;
            collectColumns(f, cols);
            if (cols.count("o_orderdate")) dateFilters.push_back(f);
        }
        std::string filterCond = combineFilters(dateFilters, idxVar);

        std::unique_ptr<MetalOperator> filtered;
        if (!filterCond.empty()) {
            filtered = std::make_unique<MetalSelection>(std::move(scan), filterCond);
        } else {
            filtered = std::move(scan);
        }

        // Bitmap probe: only orders with late lineitem deliveries
        auto probed = std::make_unique<MetalBitmapProbe>(
            std::move(filtered), "d_late_bitmap",
            "o_orderkey[" + idxVar + "]");

        // KeyedAgg: 5 priority bins (o_orderpriority first char '1'..'5' → bins 0..4)
        // o_orderpriority is CHAR_FIXED(15), first char at o_orderpriority[i * 15]
        std::string bucketExpr = "(o_orderpriority[" + idxVar + " * 15] - '1')";
        auto agg = std::make_unique<MetalKeyedAgg>(
            std::move(probed), "d_q4_counts", bucketExpr,
            /*numBuckets=*/5, /*valuesPerBucket=*/1, "5");
        agg->addAggregate("order_count", 0, "1u", "add", false, 0);

        MetalQueryPlan::Phase phase;
        phase.name = "Q4_count";
        phase.root = std::move(agg);
        phase.threadgroupSize = 1024;
        plan.phases.push_back(std::move(phase));
    }

    return plan;
}

// ===================================================================
// Q12 Plan Builder — Shipping Modes and Order Priority
// Pattern: BitmapBuild(orders, priority) → Filter+KeyedAgg(lineitem)
// ===================================================================

static std::optional<MetalQueryPlan> buildQ12Plan(const AnalyzedQuery& aq) {
    // Q12: lineitem + orders, GROUP BY l_shipmode, SUM(CASE priority)
    if (aq.tables.size() != 2) return std::nullopt;
    bool hasLineitem = false, hasOrders = false;
    for (auto& t : aq.tables) {
        if (t == "lineitem") hasLineitem = true;
        if (t == "orders") hasOrders = true;
    }
    if (!hasLineitem || !hasOrders) return std::nullopt;
    if (!aq.hasGroupBy()) return std::nullopt;

    // Check GROUP BY l_shipmode (not o_orderpriority — that's Q4)
    bool groupByShipmode = false;
    for (auto& g : aq.groupBy) {
        std::visit([&](auto&& node) {
            using T = std::decay_t<decltype(node)>;
            if constexpr (std::is_same_v<T, ColRef>) {
                if (node.column == "l_shipmode") groupByShipmode = true;
            }
        }, g->node);
    }
    if (!groupByShipmode) return std::nullopt;

    MetalQueryPlan plan;
    plan.name = "Q12";
    std::string idxVar = "i";

    // Phase 1: Build priority bitmap from orders
    // Set bit for o_orderkey where o_orderpriority is '1-URGENT' or '2-HIGH'
    {
        auto scan = std::make_unique<MetalGridStrideScan>("orders", "row", idxVar);
        scan->addColumn("o_orderkey", "int");
        scan->addColumn("o_orderpriority", "char");

        // o_orderpriority CHAR_FIXED(15), first char: '1' or '2' → high priority
        auto filter = std::make_unique<MetalSelection>(std::move(scan),
            "o_orderpriority[" + idxVar + " * 15] == '1' || o_orderpriority[" + idxVar + " * 15] == '2'");

        auto bitmapBuild = std::make_unique<MetalBitmapBuild>(
            std::move(filter), "d_priority_bitmap",
            "o_orderkey[" + idxVar + "]", "(maxOrderkey + 31) / 32 + 1");

        MetalQueryPlan::Phase phase;
        phase.name = "Q12_build_bitmap";
        phase.root = std::move(bitmapBuild);
        phase.threadgroupSize = 1024;
        plan.phases.push_back(std::move(phase));
    }

    // Phase 2: Scan lineitem, filter by shipmode + date constraints, probe bitmap, 4-bin count
    {
        auto scan = std::make_unique<MetalGridStrideScan>("lineitem", "row", idxVar);
        scan->addColumn("l_orderkey", "int");
        scan->addColumn("l_shipmode", "char");
        scan->addColumn("l_shipdate", "int");
        scan->addColumn("l_commitdate", "int");
        scan->addColumn("l_receiptdate", "int");

        // Lineitem filters:
        // 1. l_shipmode IN ('MAIL', 'SHIP') → first char 'M' or 'S'
        // 2. l_commitdate < l_receiptdate
        // 3. l_shipdate < l_commitdate
        // 4. l_receiptdate >= start_date AND l_receiptdate < end_date
        // l_shipmode is CHAR_FIXED(10), first char at l_shipmode[i * 10]

        // Extract date filters from analyzed query
        std::vector<PredPtr> dateFilters;
        for (auto& f : aq.filters) {
            std::set<std::string> cols;
            collectColumns(f, cols);
            if (cols.count("l_receiptdate") && !cols.count("l_commitdate"))
                dateFilters.push_back(f);
        }
        std::string dateCond = combineFilters(dateFilters, idxVar);

        std::string filterCond =
            "(l_shipmode[" + idxVar + " * 10] == 'M' || l_shipmode[" + idxVar + " * 10] == 'S') && "
            "l_commitdate[" + idxVar + "] < l_receiptdate[" + idxVar + "] && "
            "l_shipdate[" + idxVar + "] < l_commitdate[" + idxVar + "]";
        if (!dateCond.empty()) filterCond += " && " + dateCond;

        auto filtered = std::make_unique<MetalSelection>(std::move(scan), filterCond);

        // 4-bin keyed agg:
        // shipmode MAIL(0)/SHIP(2), crossed with high(+0)/low(+1) priority
        // Bucket = (l_shipmode first char == 'S' ? 2 : 0) + (bitmap_test ? 0 : 1)
        std::string bucketExpr =
            "((l_shipmode[" + idxVar + " * 10] == 'S' ? 2 : 0) + "
            "(bitmap_test(d_priority_bitmap, l_orderkey[" + idxVar + "]) ? 0 : 1))";

        auto agg = std::make_unique<MetalKeyedAgg>(
            std::move(filtered), "d_q12_counts", bucketExpr,
            /*numBuckets=*/4, /*valuesPerBucket=*/1, "4");
        agg->addAggregate("count", 0, "1u", "add", false, 0);

        MetalQueryPlan::Phase phase;
        phase.name = "Q12_count";
        phase.root = std::move(agg);
        phase.threadgroupSize = 1024;
        phase.bitmapReads.push_back({"d_priority_bitmap", ""});
        plan.phases.push_back(std::move(phase));
    }

    return plan;
}

// ===================================================================
// Dispatch: try all known patterns
// ===================================================================
// Q10 Plan Builder — Returned Item Reporting
// Pattern: ArrayStore(orders) → ArrayLookup+AtomicFloatAgg(lineitem)
// ===================================================================

static std::optional<MetalQueryPlan> buildQ10Plan(const AnalyzedQuery& aq) {
    // Q10: 4 tables (customer, orders, lineitem, nation), GROUP BY c_custkey + many cols
    // Detect: has lineitem+orders, GROUP BY includes c_custkey, SUM of revenue
    if (aq.tables.size() < 2) return std::nullopt;
    bool hasLineitem = false, hasOrders = false;
    for (auto& t : aq.tables)  {
        if (t == "lineitem") hasLineitem = true;
        if (t == "orders") hasOrders = true;
    }
    if (!hasLineitem || !hasOrders) return std::nullopt;

    // Check for l_returnflag filter and GROUP BY c_custkey
    bool hasReturnflagFilter = false;
    for (auto& f : aq.filters) {
        std::set<std::string> cols;
        collectColumns(f, cols);
        if (cols.count("l_returnflag")) hasReturnflagFilter = true;
    }
    if (!hasReturnflagFilter) return std::nullopt;

    bool groupByCustkey = false;
    for (auto& g : aq.groupBy) {
        std::visit([&](auto&& node) {
            using T = std::decay_t<decltype(node)>;
            if constexpr (std::is_same_v<T, ColRef>) {
                if (node.column == "c_custkey") groupByCustkey = true;
            }
        }, g->node);
    }
    if (!groupByCustkey) return std::nullopt;

    MetalQueryPlan plan;
    plan.name = "Q10";
    std::string idxVar = "i";

    // Phase 1: Build orders direct-address map
    // orders_map[orderkey] = custkey (filtered by date)
    {
        auto scan = std::make_unique<MetalGridStrideScan>("orders", "row", idxVar);
        scan->addColumn("o_orderkey", "int");
        scan->addColumn("o_custkey", "int");
        scan->addColumn("o_orderdate", "int");

        // Extract date filters from analyzed query
        std::vector<PredPtr> dateFilters;
        for (auto& f : aq.filters) {
            std::set<std::string> cols;
            collectColumns(f, cols);
            if (cols.count("o_orderdate")) dateFilters.push_back(f);
        }
        std::string filterCond = combineFilters(dateFilters, idxVar);

        std::unique_ptr<MetalOperator> filtered;
        if (!filterCond.empty()) {
            filtered = std::make_unique<MetalSelection>(std::move(scan), filterCond);
        } else {
            filtered = std::move(scan);
        }

        // ArrayStore: orders_map[orderkey] = custkey
        auto store = std::make_unique<MetalArrayStore>(
            std::move(filtered), "d_orders_map",
            "o_orderkey[" + idxVar + "]",
            "o_custkey[" + idxVar + "]",
            "int", "maxOrderkey");

        MetalQueryPlan::Phase phase;
        phase.name = "Q10_build_orders_map";
        phase.root = std::move(store);
        phase.threadgroupSize = 1024;
        plan.phases.push_back(std::move(phase));
    }

    // Phase 2: Probe lineitem, aggregate revenue per custkey
    {
        auto scan = std::make_unique<MetalGridStrideScan>("lineitem", "row", idxVar);
        scan->addColumn("l_orderkey", "int");
        scan->addColumn("l_returnflag", "char");
        scan->addColumn("l_extendedprice", "float");
        scan->addColumn("l_discount", "float");

        // Filter: l_returnflag = 'R'
        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "l_returnflag[" + idxVar + "] == 'R'");

        // ArrayLookup: custkey = orders_map[l_orderkey]
        auto lookup = std::make_unique<MetalArrayLookup>(
            std::move(filtered), "d_orders_map",
            "l_orderkey[" + idxVar + "]",
            "_custkey", "int", -1);

        // AtomicFloatAgg: cust_revenue[custkey] += revenue
        std::string revenue = "l_extendedprice[" + idxVar + "] * (1.0f - l_discount[" + idxVar + "])";
        auto agg = std::make_unique<MetalAtomicAgg>(
            std::move(lookup), "d_cust_revenue",
            "_custkey", revenue, "maxCustkey",
            "atomic_float", "float");

        MetalQueryPlan::Phase phase;
        phase.name = "Q10_probe_aggregate";
        phase.root = std::move(agg);
        phase.threadgroupSize = 1024;
        plan.phases.push_back(std::move(phase));
    }

    return plan;
}

// ===================================================================
// Q7: Volume Shipping — 4 phases
// ===================================================================
// Phase 1: Scan supplier → filter (FRANCE | GERMANY) → ArrayStore(supp_nation_map)
// Phase 2: Scan customer → filter (FRANCE | GERMANY) → ArrayStore(cust_nation_map)
// Phase 3: Scan orders → ArrayStore(orders_map[orderkey] = custkey)
// Phase 4: Scan lineitem → date filter → 3 ArrayLookups → pair filter → AtomicFloatAgg(4 bins)
// Result: 4 bins = 2 nation pairs × 2 years
//   bin = pair_idx * 2 + year_idx
//   pair 0: FRANCE→GERMANY, pair 1: GERMANY→FRANCE
//   year 0: 1995, year 1: 1996

static std::optional<MetalQueryPlan> buildQ7Plan_byName();

static std::optional<MetalQueryPlan> buildQ7Plan(const AnalyzedQuery& aq) {
    // Detection: 6 table refs (supplier, lineitem, orders, customer, nation×2),
    // GROUP BY with l_year, and joins linking all tables
    bool hasNation = false, hasLineitem = false, hasOrders = false;
    bool hasSupplier = false, hasCustomer = false;
    int nationCount = 0;
    for (auto& t : aq.tables) {
        if (t == "nation") { hasNation = true; nationCount++; }
        if (t == "lineitem") hasLineitem = true;
        if (t == "orders") hasOrders = true;
        if (t == "supplier") hasSupplier = true;
        if (t == "customer") hasCustomer = true;
    }
    if (!(hasNation && hasLineitem && hasOrders && hasSupplier && hasCustomer))
        return std::nullopt;
    if (nationCount < 2) return std::nullopt;

    if (aq.groupBy.size() < 3) return std::nullopt;

    return buildQ7Plan_byName();
}

static std::optional<MetalQueryPlan> buildQ7Plan_byName() {
    std::string idx = "i";

    MetalQueryPlan plan;

    // Phase 1: Build supplier nation map
    {
        auto scan = std::make_unique<MetalGridStrideScan>("supplier", "row", idx);
        scan->addColumn("s_suppkey", "int");
        scan->addColumn("s_nationkey", "int");

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "s_nationkey[" + idx + "] == france_nk || s_nationkey[" + idx + "] == germany_nk");

        // ArrayStore: supp_nation_map[suppkey] = nationkey
        auto store = std::make_unique<MetalArrayStore>(
            std::move(filtered), "d_supp_nation_map",
            "s_suppkey[" + idx + "]", "s_nationkey[" + idx + "]",
            "int", "maxSuppkey");

        MetalQueryPlan::Phase phase;
        phase.name = "Q7_build_supp_map";
        phase.root = std::move(store);
        phase.threadgroupSize = 256;
        phase.scalarParams = {{"france_nk", "int"}, {"germany_nk", "int"}};
        plan.phases.push_back(std::move(phase));
    }

    // Phase 2: Build customer nation map
    {
        auto scan = std::make_unique<MetalGridStrideScan>("customer", "row", idx);
        scan->addColumn("c_custkey", "int");
        scan->addColumn("c_nationkey", "int");

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "c_nationkey[" + idx + "] == france_nk || c_nationkey[" + idx + "] == germany_nk");

        // ArrayStore: cust_nation_map[custkey] = nationkey
        auto store = std::make_unique<MetalArrayStore>(
            std::move(filtered), "d_cust_nation_map",
            "c_custkey[" + idx + "]", "c_nationkey[" + idx + "]",
            "int", "maxCustkey");

        MetalQueryPlan::Phase phase;
        phase.name = "Q7_build_cust_map";
        phase.root = std::move(store);
        phase.threadgroupSize = 256;
        phase.scalarParams = {{"france_nk", "int"}, {"germany_nk", "int"}};
        plan.phases.push_back(std::move(phase));
    }

    // Phase 3: Build orders map (orderkey → custkey)
    {
        auto scan = std::make_unique<MetalGridStrideScan>("orders", "row", idx);
        scan->addColumn("o_orderkey", "int");
        scan->addColumn("o_custkey", "int");

        // ArrayStore: orders_map[orderkey] = custkey
        auto store = std::make_unique<MetalArrayStore>(
            std::move(scan), "d_orders_map",
            "o_orderkey[" + idx + "]", "o_custkey[" + idx + "]",
            "int", "maxOrderkey");

        MetalQueryPlan::Phase phase;
        phase.name = "Q7_build_orders_map";
        phase.root = std::move(store);
        phase.threadgroupSize = 256;
        plan.phases.push_back(std::move(phase));
    }

    // Phase 4: Probe lineitem → cascaded lookups → aggregate into 4 bins
    {
        auto scan = std::make_unique<MetalGridStrideScan>("lineitem", "row", idx);
        scan->addColumn("l_orderkey", "int");
        scan->addColumn("l_suppkey", "int");
        scan->addColumn("l_shipdate", "int");
        scan->addColumn("l_extendedprice", "float");
        scan->addColumn("l_discount", "float");

        // Date filter: 1995-01-01 to 1996-12-31
        auto dateFiltered = std::make_unique<MetalSelection>(std::move(scan),
            "l_shipdate[" + idx + "] >= 19950101 && l_shipdate[" + idx + "] <= 19961231");

        // ArrayLookup: ck = orders_map[l_orderkey]
        auto lookupOrders = std::make_unique<MetalArrayLookup>(
            std::move(dateFiltered), "d_orders_map",
            "l_orderkey[" + idx + "]",
            "_ck", "int", -1);

        // ArrayLookup: supp_nk = supp_nation_map[l_suppkey]
        auto lookupSupp = std::make_unique<MetalArrayLookup>(
            std::move(lookupOrders), "d_supp_nation_map",
            "l_suppkey[" + idx + "]",
            "_supp_nk", "int", -1);

        // ArrayLookup: cust_nk = cust_nation_map[ck]
        auto lookupCust = std::make_unique<MetalArrayLookup>(
            std::move(lookupSupp), "d_cust_nation_map",
            "_ck",
            "_cust_nk", "int", -1);

        // Pair filter: (FRANCE→GERMANY) or (GERMANY→FRANCE)
        auto pairFiltered = std::make_unique<MetalSelection>(std::move(lookupCust),
            "((_supp_nk == france_nk && _cust_nk == germany_nk) || "
            "(_supp_nk == germany_nk && _cust_nk == france_nk))");

        // AtomicFloatAgg: revenue_bins[pair_idx * 2 + year_idx]
        // pair_idx: 0 if supp=FRANCE, 1 if supp=GERMANY
        // year_idx: shipdate/10000 - 1995
        std::string bucketExpr = "((_supp_nk == france_nk) ? 0 : 1) * 2 + (l_shipdate[" + idx + "] / 10000 - 1995)";
        std::string valueExpr = "l_extendedprice[" + idx + "] * (1.0f - l_discount[" + idx + "])";

        auto agg = std::make_unique<MetalAtomicAgg>(
            std::move(pairFiltered), "d_revenue_bins",
            bucketExpr, valueExpr, "4",
            "atomic_float", "float");

        MetalQueryPlan::Phase phase;
        phase.name = "Q7_probe_aggregate";
        phase.root = std::move(agg);
        phase.threadgroupSize = 1024;
        phase.scalarParams = {{"france_nk", "int"}, {"germany_nk", "int"}};
        plan.phases.push_back(std::move(phase));
    }

    return plan;
}

// ===================================================================
// Q5: Local Supplier Volume — 5 phases
// ===================================================================
// Phase 0: Scan nation → filter(n_regionkey == asia_rk) → BitmapBuild(nation_bitmap)
// Phase 1: Scan customer → BitmapProbe(nation_bitmap, c_nationkey) → ArrayStore(cust_nation_map)
// Phase 2: Scan supplier → BitmapProbe(nation_bitmap, s_nationkey) → ArrayStore(supp_nation_map)
// Phase 3: Scan orders → date filter → ArrayLookup(cust_nation_map) → ArrayStore(orders_nation_map)
// Phase 4: Scan lineitem → ArrayLookup(orders_nation_map) → ArrayLookup(supp_nation_map)
//          → same-nation filter → AtomicFloatAgg(nation_revenue[nationkey])
// Result: 25-element array indexed by nationkey

static std::optional<MetalQueryPlan> buildQ5Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Phase 0: Build nation bitmap (ASIA nations only)
    {
        auto scan = std::make_unique<MetalGridStrideScan>("nation", "row", idx);
        scan->addColumn("n_nationkey", "int");
        scan->addColumn("n_regionkey", "int");

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "n_regionkey[" + idx + "] == asia_rk");

        auto bitmap = std::make_unique<MetalBitmapBuild>(
            std::move(filtered), "d_nation_bitmap",
            "n_nationkey[" + idx + "]", "(25 + 31) / 32");

        MetalQueryPlan::Phase phase;
        phase.name = "Q5_build_nation_bitmap";
        phase.root = std::move(bitmap);
        phase.threadgroupSize = 32;  // only 25 rows
        phase.scalarParams = {{"asia_rk", "int"}};
        plan.phases.push_back(std::move(phase));
    }

    // Phase 1: Build customer nation map (ASIA customers only)
    {
        auto scan = std::make_unique<MetalGridStrideScan>("customer", "row", idx);
        scan->addColumn("c_custkey", "int");
        scan->addColumn("c_nationkey", "int");

        auto probed = std::make_unique<MetalBitmapProbe>(std::move(scan),
            "d_nation_bitmap", "c_nationkey[" + idx + "]");

        auto store = std::make_unique<MetalArrayStore>(
            std::move(probed), "d_cust_nation_map",
            "c_custkey[" + idx + "]", "c_nationkey[" + idx + "]",
            "int", "maxCustkey");

        MetalQueryPlan::Phase phase;
        phase.name = "Q5_build_cust_map";
        phase.root = std::move(store);
        phase.threadgroupSize = 256;
        plan.phases.push_back(std::move(phase));
    }

    // Phase 2: Build supplier nation map (ASIA suppliers only)
    {
        auto scan = std::make_unique<MetalGridStrideScan>("supplier", "row", idx);
        scan->addColumn("s_suppkey", "int");
        scan->addColumn("s_nationkey", "int");

        auto probed = std::make_unique<MetalBitmapProbe>(std::move(scan),
            "d_nation_bitmap", "s_nationkey[" + idx + "]");

        auto store = std::make_unique<MetalArrayStore>(
            std::move(probed), "d_supp_nation_map",
            "s_suppkey[" + idx + "]", "s_nationkey[" + idx + "]",
            "int", "maxSuppkey");

        MetalQueryPlan::Phase phase;
        phase.name = "Q5_build_supp_map";
        phase.root = std::move(store);
        phase.threadgroupSize = 256;
        plan.phases.push_back(std::move(phase));
    }

    // Phase 3: Build orders nation map (date-filtered, customer-in-ASIA)
    {
        auto scan = std::make_unique<MetalGridStrideScan>("orders", "row", idx);
        scan->addColumn("o_orderkey", "int");
        scan->addColumn("o_custkey", "int");
        scan->addColumn("o_orderdate", "int");

        // Date filter: 1994-01-01 to 1994-12-31
        auto dateFiltered = std::make_unique<MetalSelection>(std::move(scan),
            "o_orderdate[" + idx + "] >= 19940101 && o_orderdate[" + idx + "] < 19950101");

        // Lookup customer nation map
        auto lookup = std::make_unique<MetalArrayLookup>(
            std::move(dateFiltered), "d_cust_nation_map",
            "o_custkey[" + idx + "]",
            "_cust_nk", "int", -1);

        // Store: orders_nation_map[orderkey] = cust_nationkey
        auto store = std::make_unique<MetalArrayStore>(
            std::move(lookup), "d_orders_nation_map",
            "o_orderkey[" + idx + "]", "_cust_nk",
            "int", "maxOrderkey");

        MetalQueryPlan::Phase phase;
        phase.name = "Q5_build_orders_map";
        phase.root = std::move(store);
        phase.threadgroupSize = 1024;
        plan.phases.push_back(std::move(phase));
    }

    // Phase 4: Probe lineitem → same-nation check → aggregate
    {
        auto scan = std::make_unique<MetalGridStrideScan>("lineitem", "row", idx);
        scan->addColumn("l_orderkey", "int");
        scan->addColumn("l_suppkey", "int");
        scan->addColumn("l_extendedprice", "float");
        scan->addColumn("l_discount", "float");

        // Lookup: cust_nk = orders_nation_map[l_orderkey]
        auto lookupOrders = std::make_unique<MetalArrayLookup>(
            std::move(scan), "d_orders_nation_map",
            "l_orderkey[" + idx + "]",
            "_cust_nk", "int", -1);

        // Lookup: supp_nk = supp_nation_map[l_suppkey]
        auto lookupSupp = std::make_unique<MetalArrayLookup>(
            std::move(lookupOrders), "d_supp_nation_map",
            "l_suppkey[" + idx + "]",
            "_supp_nk", "int", -1);

        // Same-nation filter: customer and supplier must be in same nation
        auto sameNation = std::make_unique<MetalSelection>(std::move(lookupSupp),
            "_cust_nk == _supp_nk");

        // AtomicFloatAgg: nation_revenue[nationkey] += revenue
        std::string valueExpr = "l_extendedprice[" + idx + "] * (1.0f - l_discount[" + idx + "])";
        auto agg = std::make_unique<MetalAtomicAgg>(
            std::move(sameNation), "d_nation_revenue",
            "_cust_nk", valueExpr, "25",
            "atomic_float", "float");

        MetalQueryPlan::Phase phase;
        phase.name = "Q5_probe_aggregate";
        phase.root = std::move(agg);
        phase.threadgroupSize = 1024;
        plan.phases.push_back(std::move(phase));
    }

    return plan;
}

// ===================================================================
// Q8: National Market Share — 6 phases
// ===================================================================
// Phase 0: Build nation bitmap (AMERICA nations)
// Phase 1: Build part bitmap (p_type = 'ECONOMY ANODIZED STEEL')
// Phase 2: Build customer nation map (AMERICA customers only)
// Phase 3: Build supplier nation map (all suppliers)
// Phase 4: Build orders year map (date-filtered, AMERICA customer)
// Phase 5: Probe lineitem → part bitmap → orders year → supp nation
//          → aggregate total revenue and Brazil revenue into 4 bins
// Result bins: [brazil_95, brazil_96, total_95, total_96]

static std::optional<MetalQueryPlan> buildQ8Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Phase 0: Build nation bitmap for AMERICA region
    {
        auto scan = std::make_unique<MetalGridStrideScan>("nation", "row", idx);
        scan->addColumn("n_nationkey", "int");
        scan->addColumn("n_regionkey", "int");

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "n_regionkey[" + idx + "] == america_rk");

        auto bitmap = std::make_unique<MetalBitmapBuild>(
            std::move(filtered), "d_america_bitmap",
            "n_nationkey[" + idx + "]", "(25 + 31) / 32");

        MetalQueryPlan::Phase phase;
        phase.name = "Q8_build_nation_bitmap";
        phase.root = std::move(bitmap);
        phase.threadgroupSize = 32;
        phase.scalarParams = {{"america_rk", "int"}};
        plan.phases.push_back(std::move(phase));
    }

    // Phase 1: Build part bitmap for 'ECONOMY ANODIZED STEEL'
    {
        auto scan = std::make_unique<MetalGridStrideScan>("part", "row", idx);
        scan->addColumn("p_partkey", "int");
        scan->addColumn("p_type", "char");

        // Compare 22 chars of p_type (CHAR_FIXED stride 25)
        std::string cond =
            "p_type[" + idx + " * 25] == 'E' && "
            "p_type[" + idx + " * 25 + 1] == 'C' && "
            "p_type[" + idx + " * 25 + 2] == 'O' && "
            "p_type[" + idx + " * 25 + 3] == 'N' && "
            "p_type[" + idx + " * 25 + 4] == 'O' && "
            "p_type[" + idx + " * 25 + 5] == 'M' && "
            "p_type[" + idx + " * 25 + 6] == 'Y' && "
            "p_type[" + idx + " * 25 + 7] == ' ' && "
            "p_type[" + idx + " * 25 + 8] == 'A' && "
            "p_type[" + idx + " * 25 + 9] == 'N' && "
            "p_type[" + idx + " * 25 + 10] == 'O' && "
            "p_type[" + idx + " * 25 + 11] == 'D' && "
            "p_type[" + idx + " * 25 + 12] == 'I' && "
            "p_type[" + idx + " * 25 + 13] == 'Z' && "
            "p_type[" + idx + " * 25 + 14] == 'E' && "
            "p_type[" + idx + " * 25 + 15] == 'D' && "
            "p_type[" + idx + " * 25 + 16] == ' ' && "
            "p_type[" + idx + " * 25 + 17] == 'S' && "
            "p_type[" + idx + " * 25 + 18] == 'T' && "
            "p_type[" + idx + " * 25 + 19] == 'E' && "
            "p_type[" + idx + " * 25 + 20] == 'E' && "
            "p_type[" + idx + " * 25 + 21] == 'L'";

        auto filtered = std::make_unique<MetalSelection>(std::move(scan), cond);

        auto bitmap = std::make_unique<MetalBitmapBuild>(
            std::move(filtered), "d_part_bitmap",
            "p_partkey[" + idx + "]", "(maxPartkey + 31) / 32");

        MetalQueryPlan::Phase phase;
        phase.name = "Q8_build_part_bitmap";
        phase.root = std::move(bitmap);
        phase.threadgroupSize = 256;
        plan.phases.push_back(std::move(phase));
    }

    // Phase 2: Build customer nation map (only AMERICA customers)
    {
        auto scan = std::make_unique<MetalGridStrideScan>("customer", "row", idx);
        scan->addColumn("c_custkey", "int");
        scan->addColumn("c_nationkey", "int");

        auto probed = std::make_unique<MetalBitmapProbe>(std::move(scan),
            "d_america_bitmap", "c_nationkey[" + idx + "]");

        auto store = std::make_unique<MetalArrayStore>(
            std::move(probed), "d_cust_nation_map",
            "c_custkey[" + idx + "]", "c_nationkey[" + idx + "]",
            "int", "maxCustkey");

        MetalQueryPlan::Phase phase;
        phase.name = "Q8_build_cust_map";
        phase.root = std::move(store);
        phase.threadgroupSize = 256;
        plan.phases.push_back(std::move(phase));
    }

    // Phase 3: Build supplier nation map (all suppliers)
    {
        auto scan = std::make_unique<MetalGridStrideScan>("supplier", "row", idx);
        scan->addColumn("s_suppkey", "int");
        scan->addColumn("s_nationkey", "int");

        auto store = std::make_unique<MetalArrayStore>(
            std::move(scan), "d_supp_nation_map",
            "s_suppkey[" + idx + "]", "s_nationkey[" + idx + "]",
            "int", "maxSuppkey");

        MetalQueryPlan::Phase phase;
        phase.name = "Q8_build_supp_map";
        phase.root = std::move(store);
        phase.threadgroupSize = 256;
        plan.phases.push_back(std::move(phase));
    }

    // Phase 4: Build orders year map (date-filtered, AMERICA customer)
    {
        auto scan = std::make_unique<MetalGridStrideScan>("orders", "row", idx);
        scan->addColumn("o_orderkey", "int");
        scan->addColumn("o_custkey", "int");
        scan->addColumn("o_orderdate", "int");

        auto dateFiltered = std::make_unique<MetalSelection>(std::move(scan),
            "o_orderdate[" + idx + "] >= 19950101 && o_orderdate[" + idx + "] <= 19961231");

        auto lookup = std::make_unique<MetalArrayLookup>(
            std::move(dateFiltered), "d_cust_nation_map",
            "o_custkey[" + idx + "]",
            "_cust_nk", "int", -1);

        auto store = std::make_unique<MetalArrayStore>(
            std::move(lookup), "d_orders_year_map",
            "o_orderkey[" + idx + "]", "o_orderdate[" + idx + "] / 10000",
            "int", "maxOrderkey");

        MetalQueryPlan::Phase phase;
        phase.name = "Q8_build_orders_map";
        phase.root = std::move(store);
        phase.threadgroupSize = 1024;
        plan.phases.push_back(std::move(phase));
    }

    // Phase 5: Probe lineitem — dual aggregation into result_bins
    // [0]=brazil_95, [1]=brazil_96, [2]=total_95, [3]=total_96
    {
        auto scan = std::make_unique<MetalGridStrideScan>("lineitem", "row", idx);
        scan->addColumn("l_orderkey", "int");
        scan->addColumn("l_partkey", "int");
        scan->addColumn("l_suppkey", "int");
        scan->addColumn("l_extendedprice", "float");
        scan->addColumn("l_discount", "float");

        auto partProbed = std::make_unique<MetalBitmapProbe>(std::move(scan),
            "d_part_bitmap", "l_partkey[" + idx + "]");

        auto lookupYear = std::make_unique<MetalArrayLookup>(
            std::move(partProbed), "d_orders_year_map",
            "l_orderkey[" + idx + "]",
            "_year", "int", -1);

        auto lookupSupp = std::make_unique<MetalArrayLookup>(
            std::move(lookupYear), "d_supp_nation_map",
            "l_suppkey[" + idx + "]",
            "_supp_nk", "int", -1);

        std::string revenue = "l_extendedprice[" + idx + "] * (1.0f - l_discount[" + idx + "])";

        // Total aggregation (always): result_bins[2 + (year - 1995)]
        auto totalAgg = std::make_unique<MetalAtomicAgg>(
            std::move(lookupSupp), "d_result_bins",
            "2 + (_year - 1995)", revenue, "4",
            "atomic_float", "float");

        // Brazil aggregation (conditional): result_bins[year - 1995]
        auto brazilFilter = std::make_unique<MetalSelection>(std::move(totalAgg),
            "_supp_nk == brazil_nk");

        auto brazilAgg = std::make_unique<MetalAtomicAgg>(
            std::move(brazilFilter), "d_result_bins",
            "_year - 1995", revenue, "4",
            "atomic_float", "float");

        MetalQueryPlan::Phase phase;
        phase.name = "Q8_probe_aggregate";
        phase.root = std::move(brazilAgg);
        phase.threadgroupSize = 1024;
        phase.scalarParams = {{"brazil_nk", "int"}};
        plan.phases.push_back(std::move(phase));
    }

    return plan;
}

// ===================================================================
// Q3: Shipping Priority — 3 phases
// ===================================================================
static std::optional<MetalQueryPlan> buildQ3Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Phase 1: Build customer bitmap (BUILDING segment)
    {
        auto scan = std::make_unique<MetalGridStrideScan>("customer", "row", idx);
        scan->addColumn("c_custkey", "int");
        scan->addColumn("c_mktsegment", "char");

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "c_mktsegment[" + idx + " * 10] == 'B'");

        auto bitmap = std::make_unique<MetalBitmapBuild>(
            std::move(filtered), "d_cust_bitmap",
            "c_custkey[" + idx + "]", "(maxCustkey + 31) / 32");

        MetalQueryPlan::Phase phase;
        phase.name = "Q3_build_cust_bitmap";
        phase.root = std::move(bitmap);
        phase.threadgroupSize = 256;
        plan.phases.push_back(std::move(phase));
    }

    // Phase 2: Build orders maps (date + priority, dual ArrayStore)
    {
        auto scan = std::make_unique<MetalGridStrideScan>("orders", "row", idx);
        scan->addColumn("o_orderkey", "int");
        scan->addColumn("o_custkey", "int");
        scan->addColumn("o_orderdate", "int");
        scan->addColumn("o_shippriority", "int");

        auto dateFiltered = std::make_unique<MetalSelection>(std::move(scan),
            "o_orderdate[" + idx + "] < 19950315");

        auto custProbed = std::make_unique<MetalBitmapProbe>(std::move(dateFiltered),
            "d_cust_bitmap", "o_custkey[" + idx + "]");

        auto storeDate = std::make_unique<MetalArrayStore>(
            std::move(custProbed), "d_orders_date_map",
            "o_orderkey[" + idx + "]", "o_orderdate[" + idx + "]",
            "int", "maxOrderkey");

        auto storePrio = std::make_unique<MetalArrayStore>(
            std::move(storeDate), "d_orders_prio_map",
            "o_orderkey[" + idx + "]", "o_shippriority[" + idx + "]",
            "int", "maxOrderkey");

        MetalQueryPlan::Phase phase;
        phase.name = "Q3_build_orders_maps";
        phase.root = std::move(storePrio);
        phase.threadgroupSize = 256;
        plan.phases.push_back(std::move(phase));
    }

    // Phase 3: Probe lineitem → aggregate revenue per orderkey
    {
        auto scan = std::make_unique<MetalGridStrideScan>("lineitem", "row", idx);
        scan->addColumn("l_orderkey", "int");
        scan->addColumn("l_shipdate", "int");
        scan->addColumn("l_extendedprice", "float");
        scan->addColumn("l_discount", "float");

        auto dateFiltered = std::make_unique<MetalSelection>(std::move(scan),
            "l_shipdate[" + idx + "] > 19950315");

        auto lookup = std::make_unique<MetalArrayLookup>(
            std::move(dateFiltered), "d_orders_date_map",
            "l_orderkey[" + idx + "]",
            "_odate", "int", -1);

        std::string revenue = "l_extendedprice[" + idx + "] * (1.0f - l_discount[" + idx + "])";
        auto agg = std::make_unique<MetalAtomicAgg>(
            std::move(lookup), "d_order_revenue",
            "l_orderkey[" + idx + "]", revenue, "maxOrderkey",
            "atomic_float", "float");

        MetalQueryPlan::Phase phase;
        phase.name = "Q3_probe_aggregate";
        phase.root = std::move(agg);
        phase.threadgroupSize = 1024;
        plan.phases.push_back(std::move(phase));
    }

    return plan;
}

// ===================================================================
// Q13: Customer Distribution — 2 phases
// ===================================================================
static std::optional<MetalQueryPlan> buildQ13Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Helper: two-segment LIKE match for 'special...requests' in comment
    plan.helpers.push_back(R"(
static bool q13_comment_match(const device char* comment, uint idx) {
    const device char* c = comment + idx * 79;
    for (int p = 0; p <= 72; p++) {
        if (c[p]=='s' && c[p+1]=='p' && c[p+2]=='e' && c[p+3]=='c' &&
            c[p+4]=='i' && c[p+5]=='a' && c[p+6]=='l') {
            for (int q = p + 7; q <= 71; q++) {
                if (c[q]=='r' && c[q+1]=='e' && c[q+2]=='q' && c[q+3]=='u' &&
                    c[q+4]=='e' && c[q+5]=='s' && c[q+6]=='t' && c[q+7]=='s') {
                    return true;
                }
            }
        }
    }
    return false;
}
)");

    // Phase 1: Scan orders, filter NOT LIKE, count per custkey
    {
        auto scan = std::make_unique<MetalGridStrideScan>("orders", "row", idx);
        scan->addColumn("o_custkey", "int");
        scan->addColumn("o_comment", "char");

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "!q13_comment_match(o_comment, " + idx + ")");

        auto count = std::make_unique<MetalAtomicCount>(
            std::move(filtered), "d_order_counts",
            "o_custkey[" + idx + "]", "maxCustkey");

        MetalQueryPlan::Phase phase;
        phase.name = "Q13_count_orders";
        phase.root = std::move(count);
        phase.threadgroupSize = 256;
        plan.phases.push_back(std::move(phase));
    }

    // Phase 2: Scan customers, read order count, build histogram
    {
        auto scan = std::make_unique<MetalGridStrideScan>("customer", "row", idx);
        scan->addColumn("c_custkey", "int");

        auto lookup = std::make_unique<MetalArrayLookup>(
            std::move(scan), "d_order_counts",
            "c_custkey[" + idx + "]",
            "_cnt", "int", 0x7FFFFFFF);

        auto hist = std::make_unique<MetalAtomicCount>(
            std::move(lookup), "d_histogram",
            "_cnt", "256");

        MetalQueryPlan::Phase phase;
        phase.name = "Q13_build_histogram";
        phase.root = std::move(hist);
        phase.threadgroupSize = 256;
        plan.phases.push_back(std::move(phase));
    }

    return plan;
}

// ===================================================================
// Q22: Global Sales Opportunity — 2 phases
// ===================================================================
static std::optional<MetalQueryPlan> buildQ22Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Phase 1: Build orders bitmap
    {
        auto scan = std::make_unique<MetalGridStrideScan>("orders", "row", idx);
        scan->addColumn("o_custkey", "int");

        auto bitmap = std::make_unique<MetalBitmapBuild>(
            std::move(scan), "d_cust_order_bitmap",
            "o_custkey[" + idx + "]", "(maxCustkey + 31) / 32");

        MetalQueryPlan::Phase phase;
        phase.name = "Q22_build_bitmap";
        phase.root = std::move(bitmap);
        phase.threadgroupSize = 1024;
        plan.phases.push_back(std::move(phase));
    }

    // Phase 2: Scan customer, filter, anti-bitmap, dual aggregate
    {
        auto scan = std::make_unique<MetalGridStrideScan>("customer", "row", idx);
        scan->addColumn("c_custkey", "int");
        scan->addColumn("c_phone", "char");
        scan->addColumn("c_acctbal", "float");

        auto computePrefix = std::make_unique<MetalComputeExpr>(
            std::move(scan), "_prefix", "int",
            "(c_phone[" + idx + " * 15] - '0') * 10 + (c_phone[" + idx + " * 15 + 1] - '0')");

        std::string validPrefixCond =
            "(_prefix == 13 || _prefix == 17 || _prefix == 18 || "
            "_prefix == 23 || _prefix == 29 || _prefix == 30 || _prefix == 31) && "
            "c_acctbal[" + idx + "] > avg_bal";
        auto filtered = std::make_unique<MetalSelection>(std::move(computePrefix), validPrefixCond);

        auto antiProbed = std::make_unique<MetalAntiBitmapProbe>(
            std::move(filtered), "d_cust_order_bitmap",
            "c_custkey[" + idx + "]");

        auto computeBin = std::make_unique<MetalComputeExpr>(
            std::move(antiProbed), "_bin", "int",
            "(_prefix == 13 ? 0 : _prefix == 17 ? 1 : _prefix == 18 ? 2 : "
            "_prefix == 23 ? 3 : _prefix == 29 ? 4 : _prefix == 30 ? 5 : 6)");

        auto count = std::make_unique<MetalAtomicCount>(
            std::move(computeBin), "d_q22_count", "_bin", "7");

        auto sum = std::make_unique<MetalAtomicAgg>(
            std::move(count), "d_q22_sum",
            "_bin", "c_acctbal[" + idx + "]", "7",
            "atomic_float", "float");

        MetalQueryPlan::Phase phase;
        phase.name = "Q22_final_aggregate";
        phase.root = std::move(sum);
        phase.threadgroupSize = 256;
        phase.scalarParams = {{"avg_bal", "float"}};
        plan.phases.push_back(std::move(phase));
    }

    return plan;
}

// ===================================================================
// Q11: Important Stock Identification — 2 phases
// ===================================================================
static std::optional<MetalQueryPlan> buildQ11Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Phase 1: Build supplier bitmap for GERMANY
    {
        auto scan = std::make_unique<MetalGridStrideScan>("supplier", "row", idx);
        scan->addColumn("s_suppkey", "int");
        scan->addColumn("s_nationkey", "int");

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "s_nationkey[" + idx + "] == germany_nk");

        auto bitmap = std::make_unique<MetalBitmapBuild>(
            std::move(filtered), "d_supp_bitmap",
            "s_suppkey[" + idx + "]", "(maxSuppkey + 31) / 32");

        MetalQueryPlan::Phase phase;
        phase.name = "Q11_build_supp_bitmap";
        phase.root = std::move(bitmap);
        phase.threadgroupSize = 256;
        phase.scalarParams = {{"germany_nk", "int"}};
        plan.phases.push_back(std::move(phase));
    }

    // Phase 2: Scan partsupp → bitmap probe → per-part value aggregation
    {
        auto scan = std::make_unique<MetalGridStrideScan>("partsupp", "row", idx);
        scan->addColumn("ps_partkey", "int");
        scan->addColumn("ps_suppkey", "int");
        scan->addColumn("ps_supplycost", "float");
        scan->addColumn("ps_availqty", "int");

        auto probed = std::make_unique<MetalBitmapProbe>(std::move(scan),
            "d_supp_bitmap", "ps_suppkey[" + idx + "]");

        std::string valueExpr = "ps_supplycost[" + idx + "] * (float)ps_availqty[" + idx + "]";
        auto agg = std::make_unique<MetalAtomicAgg>(
            std::move(probed), "d_part_value",
            "ps_partkey[" + idx + "]", valueExpr, "maxPartkey",
            "atomic_float", "float");

        MetalQueryPlan::Phase phase;
        phase.name = "Q11_aggregate";
        phase.root = std::move(agg);
        phase.threadgroupSize = 256;
        plan.phases.push_back(std::move(phase));
    }

    return plan;
}

// ===================================================================
// Q19: Discounted Revenue — 2 phases
// ===================================================================
static std::optional<MetalQueryPlan> buildQ19Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    plan.helpers.push_back(R"(
static bool brand_eq(const device char* brand, uint idx, char d1, char d2) {
    const device char* b = brand + idx * 10;
    return b[0]=='B' && b[1]=='r' && b[2]=='a' && b[3]=='n' && b[4]=='d' && b[5]=='#' && b[6]==d1 && b[7]==d2;
}
static int container_match(const device char* cont, uint idx) {
    const device char* c = cont + idx * 10;
    // SM CASE/BOX/PACK/PKG -> 1
    if (c[0]=='S' && c[1]=='M' && c[2]==' ') {
        char c3=c[3],c4=c[4],c5=c[5];
        if ((c3=='C'&&c4=='A'&&c5=='S') || (c3=='B'&&c4=='O') ||
            (c3=='P'&&c4=='A'&&c5=='C') || (c3=='P'&&c4=='K')) return 1;
    }
    // MED BAG/BOX/PKG/PACK -> 2
    if (c[0]=='M' && c[1]=='E' && c[2]=='D' && c[3]==' ') {
        char c4=c[4],c5=c[5],c6=c[6];
        if ((c4=='B'&&c5=='A'&&c6=='G') || (c4=='B'&&c5=='O') ||
            (c4=='P'&&c5=='K') || (c4=='P'&&c5=='A'&&c6=='C')) return 2;
    }
    // LG CASE/BOX/PACK/PKG -> 3
    if (c[0]=='L' && c[1]=='G' && c[2]==' ') {
        char c3=c[3],c4=c[4],c5=c[5];
        if ((c3=='C'&&c4=='A'&&c5=='S') || (c3=='B'&&c4=='O') ||
            (c3=='P'&&c4=='A'&&c5=='C') || (c3=='P'&&c4=='K')) return 3;
    }
    return 0;
}
)");

    // Phase 1: Build part condition bitmask map
    {
        auto scan = std::make_unique<MetalGridStrideScan>("part", "row", idx);
        scan->addColumn("p_partkey", "int");
        scan->addColumn("p_brand", "char");
        scan->addColumn("p_container", "char");
        scan->addColumn("p_size", "int");

        auto computeCond = std::make_unique<MetalComputeExpr>(
            std::move(scan), "_cond", "int",
            "(brand_eq(p_brand, " + idx + ", '1', '2') && container_match(p_container, " + idx + ") == 1 && "
            "p_size[" + idx + "] >= 1 && p_size[" + idx + "] <= 5 ? 1 : 0) | "
            "(brand_eq(p_brand, " + idx + ", '2', '3') && container_match(p_container, " + idx + ") == 2 && "
            "p_size[" + idx + "] >= 1 && p_size[" + idx + "] <= 10 ? 2 : 0) | "
            "(brand_eq(p_brand, " + idx + ", '3', '4') && container_match(p_container, " + idx + ") == 3 && "
            "p_size[" + idx + "] >= 1 && p_size[" + idx + "] <= 15 ? 4 : 0)");

        auto filtered = std::make_unique<MetalSelection>(std::move(computeCond), "_cond > 0");

        auto store = std::make_unique<MetalArrayStore>(
            std::move(filtered), "d_part_cond",
            "p_partkey[" + idx + "]", "_cond", "int", "maxPartkey");

        MetalQueryPlan::Phase phase;
        phase.name = "Q19_build_part_cond";
        phase.root = std::move(store);
        phase.threadgroupSize = 256;
        plan.phases.push_back(std::move(phase));
    }

    // Phase 2: Scan lineitem, lookup part condition, check quantity, reduce revenue
    {
        auto scan = std::make_unique<MetalGridStrideScan>("lineitem", "row", idx);
        scan->addColumn("l_partkey", "int");
        scan->addColumn("l_quantity", "float");
        scan->addColumn("l_extendedprice", "float");
        scan->addColumn("l_discount", "float");
        scan->addColumn("l_shipmode", "char");
        scan->addColumn("l_shipinstruct", "char");

        // l_shipmode IN ('AIR', 'REG AIR') — 'A..' or 'RE..'
        // l_shipinstruct = 'DELIVER IN PERSON' — first char 'D'
        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "(l_shipmode[" + idx + " * 10] == 'A' || (l_shipmode[" + idx + " * 10] == 'R' && l_shipmode[" + idx + " * 10 + 1] == 'E')) && l_shipinstruct[" + idx + " * 25] == 'D'");

        auto lookup = std::make_unique<MetalArrayLookup>(
            std::move(filtered), "d_part_cond",
            "l_partkey[" + idx + "]", "_cond", "int", -1);

        std::string qtyCheck =
            "((_cond & 1) && l_quantity[" + idx + "] >= 1.0f && l_quantity[" + idx + "] <= 11.0f) || "
            "((_cond & 2) && l_quantity[" + idx + "] >= 10.0f && l_quantity[" + idx + "] <= 20.0f) || "
            "((_cond & 4) && l_quantity[" + idx + "] >= 20.0f && l_quantity[" + idx + "] <= 30.0f)";
        auto qtyFiltered = std::make_unique<MetalSelection>(std::move(lookup), qtyCheck);

        std::string revenue = "(long)(l_extendedprice[" + idx + "] * (1.0f - l_discount[" + idx + "]) * 100.0f)";
        auto reduce = std::make_unique<MetalTGReduce>(std::move(qtyFiltered), "d_q19");
        reduce->addAccumulator("revenue", revenue, "long");
        reduce->setResultAlias("revenue", 100);

        MetalQueryPlan::Phase phase;
        phase.name = "Q19_reduce";
        phase.root = std::move(reduce);
        phase.threadgroupSize = 1024;
        plan.phases.push_back(std::move(phase));
    }

    return plan;
}

// ===================================================================
// Q15: Top Supplier — 1 GPU phase + CPU max scan
// ===================================================================
static std::optional<MetalQueryPlan> buildQ15Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    {
        auto scan = std::make_unique<MetalGridStrideScan>("lineitem", "row", idx);
        scan->addColumn("l_suppkey", "int");
        scan->addColumn("l_shipdate", "int");
        scan->addColumn("l_extendedprice", "float");
        scan->addColumn("l_discount", "float");

        auto filtered = std::make_unique<MetalSelection>(std::move(scan),
            "l_shipdate[" + idx + "] >= 19960101 && l_shipdate[" + idx + "] < 19960401");

        std::string revenue = "l_extendedprice[" + idx + "] * (1.0f - l_discount[" + idx + "])";
        auto agg = std::make_unique<MetalAtomicAgg>(
            std::move(filtered), "d_supp_revenue",
            "l_suppkey[" + idx + "]", revenue, "maxSuppkey",
            "atomic_float", "float");

        MetalQueryPlan::Phase phase;
        phase.name = "Q15_aggregate";
        phase.root = std::move(agg);
        phase.threadgroupSize = 1024;
        plan.phases.push_back(std::move(phase));
    }

    return plan;
}

// ===================================================================
// Q18: Large Volume Customer — 1 GPU phase + CPU filter/sort
// ===================================================================
static std::optional<MetalQueryPlan> buildQ18Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    {
        auto scan = std::make_unique<MetalGridStrideScan>("lineitem", "row", idx);
        scan->addColumn("l_orderkey", "int");
        scan->addColumn("l_quantity", "float");

        auto agg = std::make_unique<MetalAtomicAgg>(
            std::move(scan), "d_order_qty",
            "l_orderkey[" + idx + "]", "l_quantity[" + idx + "]", "maxOrderkey",
            "atomic_float", "float");

        MetalQueryPlan::Phase phase;
        phase.name = "Q18_aggregate";
        phase.root = std::move(agg);
        phase.threadgroupSize = 1024;
        plan.phases.push_back(std::move(phase));
    }

    return plan;
}

// ===================================================================
// Q17: Small-Quantity-Order Revenue
// CPU pre-processing computes threshold[partkey] = 0.2 * avg(qty) for qualifying parts.
// Single GPU phase: scan lineitem, lookup threshold, filter qty < threshold, sum extendedprice.
// ===================================================================
static std::optional<MetalQueryPlan> buildQ17Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    {
        auto scan = std::make_unique<MetalGridStrideScan>("lineitem", "row", idx);
        scan->addColumn("l_partkey", "int");
        scan->addColumn("l_quantity", "float");
        scan->addColumn("l_extendedprice", "float");

        // Lookup pre-computed threshold: 0.0f means non-qualifying part
        auto lookup = std::make_unique<MetalArrayLookup>(
            std::move(scan), "d_q17_threshold",
            "l_partkey[" + idx + "]", "_thresh", "float", 0);

        // Filter: l_quantity < threshold for this partkey
        auto filtered = std::make_unique<MetalSelection>(
            std::move(lookup),
            "l_quantity[" + idx + "] < _thresh");

        // TGReduce revenue
        auto reduce = std::make_unique<MetalTGReduce>(std::move(filtered), "d_q17");
        reduce->addAccumulator("revenue", "(long)(l_extendedprice[" + idx + "] * 100.0f)", "float");

        MetalQueryPlan::Phase phase;
        phase.name = "Q17_reduce";
        phase.root = std::move(reduce);
        phase.threadgroupSize = 1024;
        plan.phases.push_back(std::move(phase));
    }

    return plan;
}

// ===================================================================
// Q9: Product Type Profit Measure
// CPU pre-processing builds: green-parts bitmap, s_nationkey[], o_year[],
// partsupp hash table (keys + vals + mask scalar).
// Single GPU phase: scan lineitem, filter+probe+compute profit, atomicAgg by (nation,year).
// ===================================================================
static std::optional<MetalQueryPlan> buildQ9Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Helper: hash probe function for partsupp HT
    plan.helpers.push_back(R"(
static float q9_ht_probe(const device uint* ht_keys, const device float* ht_vals,
                          uint ht_mask, uint key) {
    uint h = (key * 2654435769u) & ht_mask;
    for (uint step = 0; step < 64; step++) {
        uint slot = (h + step) & ht_mask;
        uint k = ht_keys[slot];
        if (k == key) return ht_vals[slot];
        if (k == 0xFFFFFFFFu) break;
    }
    return -1.0f;
}
)");

    {
        auto scan = std::make_unique<MetalGridStrideScan>("lineitem", "row", idx);
        scan->addColumn("l_partkey", "int");
        scan->addColumn("l_suppkey", "int");
        scan->addColumn("l_orderkey", "int");
        scan->addColumn("l_quantity", "float");
        scan->addColumn("l_extendedprice", "float");
        scan->addColumn("l_discount", "float");

        // BitmapProbe: filter to green parts
        auto bmpProbe = std::make_unique<MetalBitmapProbe>(
            std::move(scan), "d_q9_part_bitmap", "l_partkey[" + idx + "]");

        // ArrayLookup: s_nationkey[suppkey]
        auto natLookup = std::make_unique<MetalArrayLookup>(
            std::move(bmpProbe), "d_q9_s_nationkey",
            "l_suppkey[" + idx + "]", "_nationkey", "int", -1);

        // ArrayLookup: o_year[orderkey]
        auto yearLookup = std::make_unique<MetalArrayLookup>(
            std::move(natLookup), "d_q9_o_year",
            "l_orderkey[" + idx + "]", "_year", "int", 0);

        // ComputeExpr: hash probe for supplycost
        std::string htProbeExpr =
            "q9_ht_probe(d_ps_ht_keys, d_ps_ht_vals, d_ps_ht_mask, "
            "(uint)l_partkey[" + idx + "] * supp_mul + (uint)l_suppkey[" + idx + "])";
        auto computeSC = std::make_unique<MetalComputeExpr>(
            std::move(yearLookup), "_sc", "float", htProbeExpr);

        // Selection: supplycost found (>= 0)
        auto scFilter = std::make_unique<MetalSelection>(
            std::move(computeSC), "_sc >= 0.0f");

        // ComputeExpr: profit and bin
        std::string profitExpr =
            "l_extendedprice[" + idx + "] * (1.0f - l_discount[" + idx + "]) - _sc * l_quantity[" + idx + "]";
        auto computeProfit = std::make_unique<MetalComputeExpr>(
            std::move(scFilter), "_profit", "float", profitExpr);

        // Bin = nationkey * 8 + (year - 1992)
        auto computeBin = std::make_unique<MetalComputeExpr>(
            std::move(computeProfit), "_bin", "int", "_nationkey * 8 + (_year - 1992)");

        // AtomicAgg: accumulate profit per bin
        // 25 nations × 8 year slots = 200 bins
        auto agg = std::make_unique<MetalAtomicAgg>(
            std::move(computeBin), "d_q9_profit",
            "_bin", "_profit", "200",
            "atomic_float", "float");

        MetalQueryPlan::Phase phase;
        phase.name = "Q9_profit_reduce";
        phase.root = std::move(agg);
        phase.threadgroupSize = 1024;
        // Extra buffers: hash table keys, values, and mask scalar
        phase.extraBuffers.push_back({"d_ps_ht_keys", "uint", true});
        phase.extraBuffers.push_back({"d_ps_ht_vals", "float", true});
        phase.scalarParams.push_back({"d_ps_ht_mask", "uint"});
        phase.scalarParams.push_back({"supp_mul", "uint"});
        plan.phases.push_back(std::move(phase));
    }

    return plan;
}

// ===================================================================
// Q16: Parts/Supplier Relationship (COUNT DISTINCT)
// Phase 1 (GPU): Scan supplier s_comment → build complaint bitmap
// Phase 2 (GPU): scan partsupp → ArrayLookup(group_id) → Selection(>=0) →
//      AntiBitmapProbe(complaint) → helper(per-group bitmap set).
// CPU post: popcount each group's bitmap for supplier_cnt.
// ===================================================================
static std::optional<MetalQueryPlan> buildQ16Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Helper: substring search for "Customer" ... "Complaints" in s_comment
    plan.helpers.push_back(R"(
static bool q16_has_complaint(const device char* s_comment, uint idx, int width) {
    const device char* cmt = s_comment + (uint)idx * (uint)width;
    int len = width;
    while (len > 0 && (cmt[len-1] == ' ' || cmt[len-1] == '\0')) len--;
    for (int c = 0; c <= len - 8; c++) {
        if (cmt[c]=='C' && cmt[c+1]=='u' && cmt[c+2]=='s' && cmt[c+3]=='t' &&
            cmt[c+4]=='o' && cmt[c+5]=='m' && cmt[c+6]=='e' && cmt[c+7]=='r') {
            for (int d = c + 8; d <= len - 10; d++) {
                if (cmt[d]=='C' && cmt[d+1]=='o' && cmt[d+2]=='m' && cmt[d+3]=='p' &&
                    cmt[d+4]=='l' && cmt[d+5]=='a' && cmt[d+6]=='i' && cmt[d+7]=='n' &&
                    cmt[d+8]=='t' && cmt[d+9]=='s') {
                    return true;
                }
            }
            return false;
        }
    }
    return false;
}
)");

    // Helper: set bit in per-group bitmap
    plan.helpers.push_back(R"(
static void q16_bitmap_set(device atomic_uint* group_bitmaps, uint bv_ints,
                            int group_id, int suppkey) {
    uint offset = (uint)group_id * bv_ints + ((uint)suppkey >> 5u);
    atomic_fetch_or_explicit(&group_bitmaps[offset], 1u << ((uint)suppkey & 31u), memory_order_relaxed);
}
)");

    // Phase 1: Build complaint bitmap on GPU
    {
        auto scan = std::make_unique<MetalGridStrideScan>("supplier", "row", idx);
        scan->addColumn("s_suppkey", "int");
        scan->addColumn("s_comment", "char");

        auto filter = std::make_unique<MetalSelection>(
            std::move(scan),
            "q16_has_complaint(s_comment, " + idx + ", 101)");

        auto bitmapBuild = std::make_unique<MetalBitmapBuild>(
            std::move(filter), "d_q16_complaint_bitmap", "s_suppkey[" + idx + "]", "");

        MetalQueryPlan::Phase phase;
        phase.name = "Q16_build_complaint";
        phase.root = std::move(bitmapBuild);
        phase.threadgroupSize = 1024;
        plan.phases.push_back(std::move(phase));
    }

    // Phase 2: partsupp scan with bitmap ops
    {
        auto scan = std::make_unique<MetalGridStrideScan>("partsupp", "row", idx);
        scan->addColumn("ps_partkey", "int");
        scan->addColumn("ps_suppkey", "int");

        // ArrayLookup: part_group_map[ps_partkey] → group_id
        auto groupLookup = std::make_unique<MetalArrayLookup>(
            std::move(scan), "d_q16_part_group_map", "ps_partkey[" + idx + "]",
            "q16_group_id", "int");

        // Selection: group_id >= 0 (qualifying part)
        auto filter = std::make_unique<MetalSelection>(
            std::move(groupLookup), "q16_group_id >= 0");

        // AntiBitmapProbe: supplier not complained-about
        auto antiProbe = std::make_unique<MetalAntiBitmapProbe>(
            std::move(filter), "d_q16_complaint_bitmap", "ps_suppkey[" + idx + "]");

        // ComputeExpr: set bit in per-group bitmap (side-effect only)
        auto bitmapSet = std::make_unique<MetalComputeExpr>(
            std::move(antiProbe), "_unused", "int",
            "(q16_bitmap_set(d_q16_group_bitmaps, d_q16_bv_ints, "
            "q16_group_id, ps_suppkey[" + idx + "]), 0)");

        MetalQueryPlan::Phase phase;
        phase.name = "Q16_scan_bitmap";
        phase.root = std::move(bitmapSet);
        phase.threadgroupSize = 1024;
        phase.extraBuffers.push_back({"d_q16_group_bitmaps", "atomic_uint", false});
        phase.scalarParams.push_back({"d_q16_bv_ints", "uint"});
        plan.phases.push_back(std::move(phase));
    }

    return plan;
}

// ===================================================================
// Q21: Suppliers Who Kept Orders Waiting
// Phase 1 (GPU): Scan orders → build F-orders bitmap
// Phase 2 (GPU): Scan lineitem → atomicCAS to build multi_supp/multi_late bitmaps
// Phase 3 (GPU): Scan lineitem → filter → AtomicCount per supplier
// CPU pre: SA-supplier bitmap (tiny), allocate first_supp/first_late arrays
// CPU post: read per-supp counts, join names, sort, top 100.
// ===================================================================
static std::optional<MetalQueryPlan> buildQ21Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Helper for Phase 2: atomic CAS to detect multi-supplier/multi-late orders
    plan.helpers.push_back(R"(
static void q21_track_supplier(device atomic_int* first_supp,
                                device atomic_uint* multi_supp_bmp,
                                device atomic_int* first_late,
                                device atomic_uint* multi_late_bmp,
                                int ok, int sk, bool is_late) {
    // Track multi-supplier orders
    int expected = -1;
    bool was_first = atomic_compare_exchange_weak_explicit(
        &first_supp[ok], &expected, sk, memory_order_relaxed, memory_order_relaxed);
    if (!was_first && expected != sk) {
        atomic_fetch_or_explicit(&multi_supp_bmp[ok >> 5], 1u << (ok & 31), memory_order_relaxed);
    }
    // Track multi-late orders
    if (is_late) {
        expected = -1;
        was_first = atomic_compare_exchange_weak_explicit(
            &first_late[ok], &expected, sk, memory_order_relaxed, memory_order_relaxed);
        if (!was_first && expected != sk) {
            atomic_fetch_or_explicit(&multi_late_bmp[ok >> 5], 1u << (ok & 31), memory_order_relaxed);
        }
    }
}
)");

    // Phase 1: Build F-orders bitmap on GPU
    {
        auto scan = std::make_unique<MetalGridStrideScan>("orders", "row", idx);
        scan->addColumn("o_orderkey", "int");
        scan->addColumn("o_orderstatus", "char");

        auto filter = std::make_unique<MetalSelection>(
            std::move(scan), "o_orderstatus[" + idx + "] == 'F'");

        auto bitmapBuild = std::make_unique<MetalBitmapBuild>(
            std::move(filter), "d_q21_f_orders", "o_orderkey[" + idx + "]", "");

        MetalQueryPlan::Phase phase;
        phase.name = "Q21_build_f_orders";
        phase.root = std::move(bitmapBuild);
        phase.threadgroupSize = 1024;
        plan.phases.push_back(std::move(phase));
    }

    // Phase 2: Build multi_supp and multi_late bitmaps on GPU
    {
        auto scan = std::make_unique<MetalGridStrideScan>("lineitem", "row", idx);
        scan->addColumn("l_orderkey", "int");
        scan->addColumn("l_suppkey", "int");
        scan->addColumn("l_receiptdate", "int");
        scan->addColumn("l_commitdate", "int");

        // BitmapProbe: only process F-orders
        auto fOrderProbe = std::make_unique<MetalBitmapProbe>(
            std::move(scan), "d_q21_f_orders", "l_orderkey[" + idx + "]");

        // ComputeExpr: atomicCAS tracking (side-effect)
        auto trackExpr = std::make_unique<MetalComputeExpr>(
            std::move(fOrderProbe), "_unused", "int",
            "(q21_track_supplier(d_q21_first_supp, d_q21_multi_supp, "
            "d_q21_first_late, d_q21_multi_late, "
            "l_orderkey[" + idx + "], l_suppkey[" + idx + "], "
            "l_receiptdate[" + idx + "] > l_commitdate[" + idx + "]), 0)");

        MetalQueryPlan::Phase phase;
        phase.name = "Q21_build_bitmaps";
        phase.root = std::move(trackExpr);
        phase.threadgroupSize = 1024;
        phase.extraBuffers.push_back({"d_q21_first_supp", "atomic_int", false});
        phase.extraBuffers.push_back({"d_q21_first_late", "atomic_int", false});
        phase.extraBuffers.push_back({"d_q21_multi_supp", "atomic_uint", false});
        phase.extraBuffers.push_back({"d_q21_multi_late", "atomic_uint", false});
        plan.phases.push_back(std::move(phase));
    }

    // Phase 3: Count qualifying suppliers
    {
        auto scan = std::make_unique<MetalGridStrideScan>("lineitem", "row", idx);
        scan->addColumn("l_orderkey", "int");
        scan->addColumn("l_suppkey", "int");
        scan->addColumn("l_receiptdate", "int");
        scan->addColumn("l_commitdate", "int");

        // BitmapProbe: F-order
        auto fOrderProbe = std::make_unique<MetalBitmapProbe>(
            std::move(scan), "d_q21_f_orders", "l_orderkey[" + idx + "]");

        // BitmapProbe: SA supplier
        auto saProbe = std::make_unique<MetalBitmapProbe>(
            std::move(fOrderProbe), "d_q21_sa_supp", "l_suppkey[" + idx + "]");

        // Selection: late (receipt > commit)
        auto lateFilter = std::make_unique<MetalSelection>(
            std::move(saProbe),
            "l_receiptdate[" + idx + "] > l_commitdate[" + idx + "]");

        // BitmapProbe: multi-supplier order
        auto multiSuppProbe = std::make_unique<MetalBitmapProbe>(
            std::move(lateFilter), "d_q21_multi_supp", "l_orderkey[" + idx + "]");

        // AntiBitmapProbe: NOT multi-late order
        auto antiLateProbe = std::make_unique<MetalAntiBitmapProbe>(
            std::move(multiSuppProbe), "d_q21_multi_late", "l_orderkey[" + idx + "]");

        // AtomicCount: count per supplier
        auto countAgg = std::make_unique<MetalAtomicCount>(
            std::move(antiLateProbe),
            "d_q21_supp_count", "l_suppkey[" + idx + "]");

        MetalQueryPlan::Phase phase;
        phase.name = "Q21_count_qualifying";
        phase.root = std::move(countAgg);
        phase.threadgroupSize = 1024;
        plan.phases.push_back(std::move(phase));
    }

    return plan;
}

// ===================================================================
// Q2: Minimum Cost Supplier
// CPU pre-processing builds: qualifying part bitmap (size=15, type ends BRASS),
//   EUROPE supplier bitmap.
// GPU: scan partsupp → bitmap probes → atomic_min(supplycost as uint).
// ===================================================================
// Q2: Minimum Cost Supplier
// Phase 1 (GPU): Scan part → build qualifying part bitmap (size=15, type ends BRASS)
// Phase 2 (GPU): scan partsupp → bitmap probes → atomic_min(supplycost as uint).
// CPU pre: EUROPE supplier bitmap (tiny tables).
// CPU post: read min_cost array, match suppliers, join strings, sort, top 100.
// ===================================================================
static std::optional<MetalQueryPlan> buildQ2Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Helper: check if p_type ends with "BRASS" (fixed-width 25-char field)
    plan.helpers.push_back(R"(
static bool q2_type_ends_brass(const device char* p_type, uint idx) {
    const device char* tp = p_type + (uint)idx * 25u;
    int len = 25;
    while (len > 0 && (tp[len-1] == ' ' || tp[len-1] == '\0')) len--;
    return len >= 5 && tp[len-5]=='B' && tp[len-4]=='R' &&
           tp[len-3]=='A' && tp[len-2]=='S' && tp[len-1]=='S';
}
)");

    // Helper: atomic min for float (using uint reinterpretation)
    // For positive floats, as_type<uint>(f) preserves ordering.
    plan.helpers.push_back(R"(
static void q2_atomic_min(device atomic_uint* min_cost, uint partkey, float cost) {
    uint cost_uint = as_type<uint>(cost);
    atomic_fetch_min_explicit(&min_cost[partkey], cost_uint, memory_order_relaxed);
}
)");

    // Phase 1: Build part bitmap on GPU (size=15, type ends BRASS)
    {
        auto scan = std::make_unique<MetalGridStrideScan>("part", "row", idx);
        scan->addColumn("p_partkey", "int");
        scan->addColumn("p_size", "int");
        scan->addColumn("p_type", "char");

        auto sizeFilter = std::make_unique<MetalSelection>(
            std::move(scan), "p_size[" + idx + "] == 15");

        auto typeFilter = std::make_unique<MetalSelection>(
            std::move(sizeFilter),
            "q2_type_ends_brass(p_type, " + idx + ")");

        auto bitmapBuild = std::make_unique<MetalBitmapBuild>(
            std::move(typeFilter), "d_q2_part_bitmap", "p_partkey[" + idx + "]", "");

        MetalQueryPlan::Phase phase;
        phase.name = "Q2_build_part_bitmap";
        phase.root = std::move(bitmapBuild);
        phase.threadgroupSize = 1024;
        plan.phases.push_back(std::move(phase));
    }

    // Phase 2: Find min cost (existing logic)
    {
        auto scan = std::make_unique<MetalGridStrideScan>("partsupp", "row", idx);
        scan->addColumn("ps_partkey", "int");
        scan->addColumn("ps_suppkey", "int");
        scan->addColumn("ps_supplycost", "float");

        // BitmapProbe: qualifying parts (size=15, type ends BRASS)
        auto partProbe = std::make_unique<MetalBitmapProbe>(
            std::move(scan), "d_q2_part_bitmap", "ps_partkey[" + idx + "]");

        // BitmapProbe: EUROPE suppliers
        auto suppProbe = std::make_unique<MetalBitmapProbe>(
            std::move(partProbe), "d_q2_supp_bitmap", "ps_suppkey[" + idx + "]");

        // ComputeExpr: atomic min on min_cost[ps_partkey]
        auto atomicMin = std::make_unique<MetalComputeExpr>(
            std::move(suppProbe), "_unused", "int",
            "(q2_atomic_min(d_q2_min_cost, (uint)ps_partkey[" + idx + "], "
            "ps_supplycost[" + idx + "]), 0)");

        MetalQueryPlan::Phase phase;
        phase.name = "Q2_find_min_cost";
        phase.root = std::move(atomicMin);
        phase.threadgroupSize = 1024;
        phase.extraBuffers.push_back({"d_q2_min_cost", "atomic_uint", false});
        plan.phases.push_back(std::move(phase));
    }

    return plan;
}

// ===================================================================
// Q20: Potential Part Promotion
// CPU pre-processing builds: forest% bitmap, partsupp HT (pre-keyed), CANADA suppkey set.
// GPU: scan lineitem (date filter 1994) → bitmap probe → hash probe → atomic_float add.
// CPU post: scan partsupp, check availqty > 0.5 * sum_qty, filter CANADA suppliers.
// ===================================================================
static std::optional<MetalQueryPlan> buildQ20Plan_byName() {
    std::string idx = "i";
    MetalQueryPlan plan;

    // Helper: hash probe + atomic add
    plan.helpers.push_back(R"(
static void q20_ht_add(const device uint* ht_keys, device atomic_float* ht_vals,
                        uint ht_mask, uint key, float qty) {
    uint h = (key * 2654435769u) & ht_mask;
    for (uint step = 0; step < 64; step++) {
        uint slot = (h + step) & ht_mask;
        uint k = ht_keys[slot];
        if (k == key) {
            atomic_fetch_add_explicit(&ht_vals[slot], qty, memory_order_relaxed);
            return;
        }
        if (k == 0xFFFFFFFFu) return; // not in HT = not qualifying partsupp
    }
}
)");

    {
        auto scan = std::make_unique<MetalGridStrideScan>("lineitem", "row", idx);
        scan->addColumn("l_partkey", "int");
        scan->addColumn("l_suppkey", "int");
        scan->addColumn("l_quantity", "float");
        scan->addColumn("l_shipdate", "int");

        // Date filter: 1994-01-01 to 1994-12-31
        auto dateFilter = std::make_unique<MetalSelection>(
            std::move(scan),
            "l_shipdate[" + idx + "] >= 19940101 && l_shipdate[" + idx + "] < 19950101");

        // BitmapProbe: forest% parts
        auto bmpProbe = std::make_unique<MetalBitmapProbe>(
            std::move(dateFilter), "d_q20_part_bitmap", "l_partkey[" + idx + "]");

        // ComputeExpr: hash probe + atomic add (side-effect only)
        auto hashAgg = std::make_unique<MetalComputeExpr>(
            std::move(bmpProbe), "_unused", "int",
            "(q20_ht_add(d_q20_ht_keys, d_q20_ht_vals, d_q20_ht_mask, "
            "(uint)l_partkey[" + idx + "] * supp_mul + (uint)l_suppkey[" + idx + "], "
            "l_quantity[" + idx + "]), 0)");

        MetalQueryPlan::Phase phase;
        phase.name = "Q20_lineitem_agg";
        phase.root = std::move(hashAgg);
        phase.threadgroupSize = 1024;
        // Extra buffers
        phase.extraBuffers.push_back({"d_q20_ht_keys", "uint", true});
        phase.extraBuffers.push_back({"d_q20_ht_vals", "atomic_float", false});
        phase.scalarParams.push_back({"d_q20_ht_mask", "uint"});
        phase.scalarParams.push_back({"supp_mul", "uint"});
        plan.phases.push_back(std::move(phase));
    }

    return plan;
}

// ===================================================================
// Dispatch: try all known patterns
// ===================================================================

// Forward declarations for name-based plan builders
static std::optional<MetalQueryPlan> buildQ13Plan_byName();
static std::optional<MetalQueryPlan> buildQ22Plan_byName();
static std::optional<MetalQueryPlan> buildQ11Plan_byName();
static std::optional<MetalQueryPlan> buildQ19Plan_byName();
static std::optional<MetalQueryPlan> buildQ15Plan_byName();
static std::optional<MetalQueryPlan> buildQ18Plan_byName();
static std::optional<MetalQueryPlan> buildQ17Plan_byName();
static std::optional<MetalQueryPlan> buildQ9Plan_byName();
static std::optional<MetalQueryPlan> buildQ20Plan_byName();
static std::optional<MetalQueryPlan> buildQ2Plan_byName();
static std::optional<MetalQueryPlan> buildQ16Plan_byName();
static std::optional<MetalQueryPlan> buildQ21Plan_byName();

std::optional<MetalQueryPlan> buildMetalPlan(const AnalyzedQuery& aq,
                                              const std::string& queryName) {
    // Name-based dispatch first for queries that clash with analysis-based detectors
    if (queryName == "Q19") return buildQ19Plan_byName();
    if (queryName == "Q13") return buildQ13Plan_byName();
    if (queryName == "Q22") return buildQ22Plan_byName();
    if (queryName == "Q11") return buildQ11Plan_byName();
    if (queryName == "Q15") return buildQ15Plan_byName();
    if (queryName == "Q18") return buildQ18Plan_byName();
    if (queryName == "Q17") return buildQ17Plan_byName();
    if (queryName == "Q9") return buildQ9Plan_byName();
    if (queryName == "Q20") return buildQ20Plan_byName();
    if (queryName == "Q2") return buildQ2Plan_byName();
    if (queryName == "Q16") return buildQ16Plan_byName();
    if (queryName == "Q21") return buildQ21Plan_byName();

    // Analysis-based dispatch
    if (auto p = buildQ6Plan(aq)) return p;
    if (auto p = buildQ1Plan(aq)) return p;
    if (auto p = buildQ14Plan(aq)) return p;
    if (auto p = buildQ4Plan(aq)) return p;
    if (auto p = buildQ12Plan(aq)) return p;
    if (auto p = buildQ10Plan(aq)) return p;
    if (auto p = buildQ7Plan(aq)) return p;

    // Name-based fallback for queries with complex SQL (subqueries, etc.)
    if (queryName == "Q7") return buildQ7Plan_byName();
    if (queryName == "Q5") return buildQ5Plan_byName();
    if (queryName == "Q8") return buildQ8Plan_byName();
    if (queryName == "Q3") return buildQ3Plan_byName();

    return std::nullopt;
}

// ===================================================================
// Generate Metal source from a plan
// ===================================================================

MetalCodegen generateFromPlan(const MetalQueryPlan& plan) {
    MetalCodegen cg;

    // Emit helper device functions before all kernels
    for (const auto& h : plan.helpers) {
        cg.addHelper(h);
    }

    for (const auto& phase : plan.phases) {
        cg.beginPhase(phase.name);
        cg.setPhaseThreadgroupSize(phase.threadgroupSize);
        cg.setPhaseSingleThread(phase.singleThread);

        // Register bitmap read params for cross-phase bitmap references
        for (const auto& [bmpName, bmpSize] : phase.bitmapReads) {
            cg.addBitmapReadParam(bmpName, bmpSize);
        }

        // Register scalar constant params
        for (const auto& [scName, scType] : phase.scalarParams) {
            cg.addScalarParam(scName, scType);
        }

        // Register extra buffer params (pre-built hash tables, etc.)
        for (const auto& eb : phase.extraBuffers) {
            if (eb.readOnly)
                cg.addBufferParam(eb.name, "const " + eb.type, "", false);
            else
                cg.addBufferParam(eb.name, eb.type, "", false);
        }

        if (phase.root) {
            phase.root->produce(cg, [](){});
        }

        cg.endPhase();
    }

    return cg;
}

} // namespace codegen
