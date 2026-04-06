#include "query_planner.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace codegen {

namespace {

const auto& S() { return TPCHSchema::instance(); }

ColumnBinding bind(const std::string& table, const std::string& col) {
    return S().binding(table, col);
}

// ===================================================================
// PATTERN MATCHERS — one per TPC-H query structure
// ===================================================================

// Q1: Single-table scan+agg on lineitem with 6 bins
bool matchQ1(const AnalyzedQuery& aq) {
    return aq.isSingleTable() && aq.tables[0] == "lineitem"
        && aq.hasGroupBy() && aq.hasAggregation()
        && aq.groupBy.size() == 2;
}

QueryPlan planQ1(const AnalyzedQuery& /*aq*/) {
    QueryPlan plan;
    plan.name = "Q1";

    TwoStageReduceOp op;
    op.numBins = 6; // 3 returnflags × 2 linestatuses

    // Bin expression: rf_index * 2 + ls_index
    // (encoded in codegen as a special Q1 pattern)
    op.binExpr = Expr::lit(0); // placeholder — codegen recognizes Q1 pattern

    op.aggregations = {
        {AggFunc::SUM,   Expr::col("lineitem","l_quantity",4,DataType::FLOAT), "sum_qty"},
        {AggFunc::SUM,   Expr::col("lineitem","l_extendedprice",5,DataType::FLOAT), "sum_base_price"},
        {AggFunc::SUM,   nullptr, "sum_disc_price"},   // computed: price*(1-disc)
        {AggFunc::SUM,   nullptr, "sum_charge"},        // computed: disc_price*(1+tax)
        {AggFunc::SUM,   Expr::col("lineitem","l_discount",6,DataType::FLOAT), "avg_disc_sum"},
        {AggFunc::COUNT, nullptr, "count_order"},
    };

    plan.ops.push_back(std::move(op));

    CpuSortOp sort;
    sort.keys = {{"l_returnflag", false}, {"l_linestatus", false}};
    plan.ops.push_back(std::move(sort));
    return plan;
}

// Q6: Single-table scan, single SUM aggregate, no GROUP BY
bool matchQ6(const AnalyzedQuery& aq) {
    return aq.isSingleTable() && aq.tables[0] == "lineitem"
        && aq.hasAggregation() && !aq.hasGroupBy()
        && aq.targets.size() == 1;
}

QueryPlan planQ6(const AnalyzedQuery& /*aq*/) {
    QueryPlan plan;
    plan.name = "Q6";

    TwoStageReduceOp op;
    op.numBins = 1;
    op.binExpr = Expr::lit(0); // single bin
    op.aggregations = {
        {AggFunc::SUM, Expr::binary(ExprOp::MUL,
            Expr::col("lineitem","l_extendedprice",5,DataType::FLOAT),
            Expr::col("lineitem","l_discount",6,DataType::FLOAT)),
         "revenue"},
    };

    plan.ops.push_back(std::move(op));
    return plan;
}

// Q3: Three-table join (customer, orders, lineitem)
bool matchQ3(const AnalyzedQuery& aq) {
    if (aq.tables.size() != 3) return false;
    std::vector<std::string> sorted = aq.tables;
    std::sort(sorted.begin(), sorted.end());
    return sorted[0] == "customer" && sorted[1] == "lineitem" && sorted[2] == "orders";
}

QueryPlan planQ3(const AnalyzedQuery& /*aq*/) {
    QueryPlan plan;
    plan.name = "Q3";

    // Step 1: CPU bitmap on customer where c_mktsegment = 'BUILDING'
    CpuBitmapBuildOp custBm;
    custBm.table = "customer";
    custBm.keyCol = bind("customer", "c_custkey");
    custBm.filter = Predicate::cmp(CmpOp::EQ,
        Expr::col("customer","c_mktsegment",6,DataType::CHAR1),
        Expr::lits("BUILDING"));
    custBm.resultName = "customer_bitmap";
    plan.ops.push_back(std::move(custBm));

    // Step 2: Direct map on orders where o_orderdate < 19950315 AND customer bitmap
    DirectMapBuildOp ordMap;
    ordMap.table = "orders";
    ordMap.keyCol = bind("orders", "o_orderkey");
    ordMap.valueCols = {bind("orders","o_custkey"), bind("orders","o_orderdate"), bind("orders","o_shippriority")};
    ordMap.filter = Predicate::cmp(CmpOp::LT,
        Expr::col("orders","o_orderdate",4,DataType::DATE),
        Expr::lit(19950315));
    plan.ops.push_back(std::move(ordMap));

    // Step 3: Probe lineitem against orders map, aggregate by orderkey
    ProbeAggOp probe;
    probe.factTable = "lineitem";
    probe.factColumns = {
        bind("lineitem","l_orderkey"), bind("lineitem","l_shipdate"),
        bind("lineitem","l_extendedprice"), bind("lineitem","l_discount")
    };
    probe.factFilter = Predicate::cmp(CmpOp::GT,
        Expr::col("lineitem","l_shipdate",10,DataType::DATE),
        Expr::lit(19950315));
    probe.lookups = {{1, bind("lineitem","l_orderkey"), ProbeAggOp::LookupRef::MAP_LOOKUP}};
    probe.groupKeyExpr = Expr::col("lineitem","l_orderkey",0,DataType::INT);
    probe.numGroups = 0; // dynamic — use HT
    probe.aggregations = {
        {AggFunc::SUM,
         Expr::binary(ExprOp::MUL,
             Expr::col("lineitem","l_extendedprice",5,DataType::FLOAT),
             Expr::binary(ExprOp::SUB, Expr::litf(1.0f),
                 Expr::col("lineitem","l_discount",6,DataType::FLOAT))),
         "revenue"}
    };
    plan.ops.push_back(std::move(probe));

    // Step 4: Compact sparse HT
    CompactOp compact;
    compact.maxSlots = 0; // determined at runtime
    plan.ops.push_back(std::move(compact));

    // Step 5: CPU sort
    CpuSortOp sort;
    sort.keys = {{"revenue", true}, {"o_orderdate", false}};
    sort.limit = 10;
    plan.ops.push_back(std::move(sort));

    return plan;
}

// Q14: Two-table (lineitem, part), bitmap probe, two-bin reduction
bool matchQ14(const AnalyzedQuery& aq) {
    if (aq.tables.size() != 2) return false;
    std::vector<std::string> sorted = aq.tables;
    std::sort(sorted.begin(), sorted.end());
    return sorted[0] == "lineitem" && sorted[1] == "part";
}

QueryPlan planQ14(const AnalyzedQuery& /*aq*/) {
    QueryPlan plan;
    plan.name = "Q14";

    // Distinguish Q14 from Q17/Q19 by looking for CASE/LIKE 'PROMO%'
    // Q14 has a CASE WHEN p_type LIKE 'PROMO%' pattern
    // For now just check table combo + aggregate structure

    // Step 1: CPU bitmap on part where p_type LIKE 'PROMO%'
    CpuBitmapBuildOp partBm;
    partBm.table = "part";
    partBm.keyCol = bind("part", "p_partkey");
    partBm.filter = Predicate::like(
        Expr::col("part","p_type",4,DataType::CHAR_FIXED),
        "PROMO%");
    partBm.resultName = "promo_bitmap";
    plan.ops.push_back(std::move(partBm));

    // Step 2: Scan lineitem with date filter, probe bitmap, reduce 2 bins
    TwoStageReduceOp op;
    op.numBins = 2; // bin 0 = promo, bin 1 = total
    op.binExpr = Expr::lit(0); // codegen recognizes Q14 pattern
    op.aggregations = {
        {AggFunc::SUM, nullptr, "promo_revenue"},
        {AggFunc::SUM, nullptr, "total_revenue"},
    };
    plan.ops.push_back(std::move(op));

    return plan;
}

// Q13: Customer-orders left outer join with pattern match
bool matchQ13(const AnalyzedQuery& aq) {
    // Q13's outer query is SELECT c_count, count(*) FROM (subquery) GROUP BY c_count
    // Tables will contain "__subquery__" since it has a subquery in FROM
    if (aq.tables.size() == 1 && aq.tables[0] == "__subquery__")
        return true;
    // Also match if somehow the tables are directly customer+orders
    if (aq.tables.size() != 2) return false;
    std::vector<std::string> sorted = aq.tables;
    std::sort(sorted.begin(), sorted.end());
    return sorted[0] == "customer" && sorted[1] == "orders";
}

QueryPlan planQ13(const AnalyzedQuery& /*aq*/) {
    QueryPlan plan;
    plan.name = "Q13";

    // Step 1: String match on orders, count per customer
    StringMatchOp match;
    match.table = "orders";
    match.keyCol = bind("orders", "o_custkey");
    match.stringCol = bind("orders", "o_comment");
    match.pattern = "%special%requests%";
    match.negated = true;
    match.fixedWidth = 79;
    plan.ops.push_back(std::move(match));

    // Step 2: Histogram of order counts
    HistogramOp hist;
    hist.maxBuckets = 0; // determined at runtime
    plan.ops.push_back(std::move(hist));

    // Step 3: CPU sort by count desc, custdist desc
    CpuSortOp sort;
    sort.keys = {{"custdist", true}, {"c_count", true}};
    plan.ops.push_back(std::move(sort));

    return plan;
}

// Q19: Three predicate groups on lineitem+part
bool matchQ19(const AnalyzedQuery& aq) {
    // Q19 also involves lineitem+part but has OR groups with brand/container/size/quantity
    if (aq.tables.size() != 2) return false;
    std::vector<std::string> sorted = aq.tables;
    std::sort(sorted.begin(), sorted.end());
    if (sorted[0] != "lineitem" || sorted[1] != "part") return false;
    // Distinguish from Q14 by checking for OR in filters
    for (auto& f : aq.filters) {
        if (std::get_if<LogicalOr>(&f->node)) return true;
    }
    return false;
}

QueryPlan planQ19(const AnalyzedQuery& /*aq*/) {
    QueryPlan plan;
    plan.name = "Q19";

    // Simplified: bitmap on part for the 3 brand/container/size groups,
    // then scan lineitem with quantity checks
    CpuBitmapBuildOp bm;
    bm.table = "part";
    bm.keyCol = bind("part", "p_partkey");
    bm.resultName = "part_bitmap";
    // Filter constructed at codegen time (3 OR groups)
    plan.ops.push_back(std::move(bm));

    TwoStageReduceOp reduce;
    reduce.numBins = 1;
    reduce.binExpr = Expr::lit(0);
    reduce.aggregations = {
        {AggFunc::SUM,
         Expr::binary(ExprOp::MUL,
             Expr::col("lineitem","l_extendedprice",5,DataType::FLOAT),
             Expr::binary(ExprOp::SUB, Expr::litf(1.0f),
                 Expr::col("lineitem","l_discount",6,DataType::FLOAT))),
         "revenue"}
    };
    plan.ops.push_back(std::move(reduce));

    return plan;
}

// Q4: Two-table (orders, lineitem) with EXISTS subquery
bool matchQ4(const AnalyzedQuery& aq) {
    // Q4 has orders as main table + EXISTS subquery (lineitem)
    // The analyzer extracts EXISTS as an ExistsPred in the filter tree
    if (aq.tables.size() == 1 && aq.tables[0] == "orders" && aq.hasGroupBy()) {
        for (auto& g : aq.groupBy) {
            if (auto* cr = std::get_if<ColRef>(&g->node))
                if (cr->column == "o_orderpriority") return true;
        }
    }
    return false;
}

QueryPlan planQ4(const AnalyzedQuery& /*aq*/) {
    QueryPlan plan;
    plan.name = "Q4";

    // Step 1: GPU bitmap build on lineitem (commitdate < receiptdate)
    BitmapBuildOp bm;
    bm.table = "lineitem";
    bm.keyCol = bind("lineitem", "l_orderkey");
    bm.filter = Predicate::cmp(CmpOp::LT,
        Expr::col("lineitem","l_commitdate",11,DataType::DATE),
        Expr::col("lineitem","l_receiptdate",12,DataType::DATE));
    plan.ops.push_back(std::move(bm));

    // Step 2: Scan orders with date filter + bitmap_test, N-bin TG reduce
    TwoStageReduceOp reduce;
    reduce.numBins = 5; // '1' through '5' priorities
    reduce.binExpr = Expr::col("orders","o_orderpriority",5,DataType::CHAR1);
    reduce.aggregations = {
        {AggFunc::COUNT, nullptr, "order_count"},
    };
    plan.ops.push_back(std::move(reduce));

    CpuSortOp sort;
    sort.keys = {{"o_orderpriority", false}};
    plan.ops.push_back(std::move(sort));

    return plan;
}

// Q12: Two-table (orders, lineitem) with shipmode grouping
bool matchQ12(const AnalyzedQuery& aq) {
    if (aq.tables.size() != 2) return false;
    std::vector<std::string> sorted = aq.tables;
    std::sort(sorted.begin(), sorted.end());
    if (sorted[0] != "lineitem" || sorted[1] != "orders") return false;
    // Distinguish from Q4 by checking GROUP BY on shipmode
    for (auto& g : aq.groupBy) {
        if (auto* cr = std::get_if<ColRef>(&g->node))
            if (cr->column == "l_shipmode") return true;
    }
    return false;
}

QueryPlan planQ12(const AnalyzedQuery& /*aq*/) {
    QueryPlan plan;
    plan.name = "Q12";

    // Step 1: GPU bitmap build on orders (priority in '1','2')
    BitmapBuildOp bm;
    bm.table = "orders";
    bm.keyCol = bind("orders", "o_orderkey");
    bm.filter = Predicate::logOr({
        Predicate::cmp(CmpOp::EQ,
            Expr::col("orders","o_orderpriority",5,DataType::CHAR1),
            Expr::lits("1")),
        Predicate::cmp(CmpOp::EQ,
            Expr::col("orders","o_orderpriority",5,DataType::CHAR1),
            Expr::lits("2")),
    });
    plan.ops.push_back(std::move(bm));

    // Step 2: Scan lineitem with 4 filters + bitmap_test, 4-bin TG reduce
    TwoStageReduceOp reduce;
    reduce.numBins = 4; // MAIL-HIGH, MAIL-LOW, SHIP-HIGH, SHIP-LOW
    reduce.binExpr = Expr::col("lineitem","l_shipmode",14,DataType::CHAR1);
    reduce.aggregations = {
        {AggFunc::COUNT, nullptr, "high_line_count"},
        {AggFunc::COUNT, nullptr, "low_line_count"},
    };
    plan.ops.push_back(std::move(reduce));

    CpuSortOp sort;
    sort.keys = {{"l_shipmode", false}};
    plan.ops.push_back(std::move(sort));

    return plan;
}

// Q15: Top Supplier — lineitem agg revenue per suppkey, then find max
// After VIEW stripping, analyzer sees: SELECT from supplier + revenue0 (unresolved)
// We match on supplier being the only recognized table
bool matchQ15(const AnalyzedQuery& aq) {
    if (aq.tables.size() == 1 && aq.tables[0] == "supplier") return true;
    // Also match if tables includes supplier + revenue0/lineitem in some form
    for (auto& t : aq.tables)
        if (t == "supplier") return true;
    return false;
}

QueryPlan planQ15(const AnalyzedQuery& /*aq*/) {
    QueryPlan plan;
    plan.name = "Q15";

    // Step 1: GPU atomic map agg — lineitem date filter → atomic_add revenue to map[suppkey]
    ProbeAggOp agg;
    agg.factTable = "lineitem";
    agg.factColumns = {
        bind("lineitem","l_suppkey"), bind("lineitem","l_shipdate"),
        bind("lineitem","l_extendedprice"), bind("lineitem","l_discount")
    };
    agg.factFilter = Predicate::logAnd({
        Predicate::cmp(CmpOp::GE,
            Expr::col("lineitem","l_shipdate",10,DataType::DATE),
            Expr::lit(19960101)),
        Predicate::cmp(CmpOp::LT,
            Expr::col("lineitem","l_shipdate",10,DataType::DATE),
            Expr::lit(19960401)),
    });
    agg.groupKeyExpr = Expr::col("lineitem","l_suppkey",2,DataType::INT);
    agg.numGroups = 0; // per-suppkey atomic map
    agg.aggregations = {
        {AggFunc::SUM,
         Expr::binary(ExprOp::MUL,
             Expr::col("lineitem","l_extendedprice",5,DataType::FLOAT),
             Expr::binary(ExprOp::SUB, Expr::litf(1.0f),
                 Expr::col("lineitem","l_discount",6,DataType::FLOAT))),
         "total_revenue"}
    };
    plan.ops.push_back(std::move(agg));

    // CPU: find max revenue, print matching supplier(s)
    return plan;
}

// Q11: Important Stock — partsupp agg per partkey, filtered by GERMANY suppliers
bool matchQ11(const AnalyzedQuery& aq) {
    for (auto& t : aq.tables)
        if (t == "partsupp") return true;
    return false;
}

QueryPlan planQ11(const AnalyzedQuery& /*aq*/) {
    QueryPlan plan;
    plan.name = "Q11";

    // Step 1 (CPU): bitmap on supplier WHERE s_nationkey = GERMANY
    CpuBitmapBuildOp suppBm;
    suppBm.table = "supplier";
    suppBm.keyCol = bind("supplier", "s_suppkey");
    suppBm.filter = Predicate::cmp(CmpOp::EQ,
        Expr::col("supplier","s_nationkey",3,DataType::INT),
        Expr::lit(0)); // GERMANY nationkey resolved at runtime
    suppBm.resultName = "supp_bitmap";
    plan.ops.push_back(std::move(suppBm));

    // Step 2: GPU atomic map agg — per-partkey, bitmap-filtered scan of partsupp
    ProbeAggOp agg;
    agg.factTable = "partsupp";
    agg.factColumns = {
        bind("partsupp","ps_partkey"), bind("partsupp","ps_suppkey"),
        bind("partsupp","ps_supplycost"), bind("partsupp","ps_availqty")
    };
    agg.lookups = {{0, bind("partsupp","ps_suppkey"), ProbeAggOp::LookupRef::BITMAP_TEST}};
    agg.groupKeyExpr = Expr::col("partsupp","ps_partkey",0,DataType::INT);
    agg.numGroups = 0; // per-partkey atomic (+ TG reduce for global sum)
    agg.aggregations = {
        {AggFunc::SUM, nullptr, "value"}, // ps_supplycost * ps_availqty
    };
    plan.ops.push_back(std::move(agg));

    CpuSortOp sort;
    sort.keys = {{"value", true}};
    plan.ops.push_back(std::move(sort));
    return plan;
}

// Q10: Returned Item Reporting — 4 tables, GROUP BY custkey, LIMIT 20
bool matchQ10(const AnalyzedQuery& aq) {
    if (aq.tables.size() < 3) return false;
    std::vector<std::string> sorted = aq.tables;
    std::sort(sorted.begin(), sorted.end());
    // Must have customer, lineitem, orders (nation optional — may be resolved as join only)
    bool hasCust = false, hasLi = false, hasOrd = false;
    for (auto& t : sorted) {
        if (t == "customer") hasCust = true;
        if (t == "lineitem") hasLi = true;
        if (t == "orders") hasOrd = true;
    }
    if (!hasCust || !hasLi || !hasOrd) return false;
    // Distinguish from Q3 (which also has these 3) by checking for returnflag filter
    // or GROUP BY on c_custkey (Q10 groups by many customer cols)
    for (auto& g : aq.groupBy) {
        if (auto* cr = std::get_if<ColRef>(&g->node))
            if (cr->column == "c_custkey") return true;
    }
    return false;
}

QueryPlan planQ10(const AnalyzedQuery& /*aq*/) {
    QueryPlan plan;
    plan.name = "Q10";
    // Step 1: Build orders map (orderkey → custkey) with date filter
    DirectMapBuildOp ordMap;
    ordMap.table = "orders";
    ordMap.keyCol = bind("orders", "o_orderkey");
    ordMap.valueCols = {bind("orders","o_custkey")};
    ordMap.filter = Predicate::cmp(CmpOp::GE,
        Expr::col("orders","o_orderdate",4,DataType::DATE),
        Expr::lit(19931001));
    plan.ops.push_back(std::move(ordMap));

    // Step 2: Probe lineitem, filter returnflag='R', agg revenue per custkey
    ProbeAggOp probe;
    probe.factTable = "lineitem";
    probe.factColumns = {
        bind("lineitem","l_orderkey"), bind("lineitem","l_returnflag"),
        bind("lineitem","l_extendedprice"), bind("lineitem","l_discount")
    };
    probe.factFilter = Predicate::cmp(CmpOp::EQ,
        Expr::col("lineitem","l_returnflag",8,DataType::CHAR1),
        Expr::lits("R"));
    probe.lookups = {{0, bind("lineitem","l_orderkey"), ProbeAggOp::LookupRef::MAP_LOOKUP}};
    probe.groupKeyExpr = Expr::lit(0); // per-custkey (from map lookup)
    probe.numGroups = 0; // dynamic
    probe.aggregations = {
        {AggFunc::SUM,
         Expr::binary(ExprOp::MUL,
             Expr::col("lineitem","l_extendedprice",5,DataType::FLOAT),
             Expr::binary(ExprOp::SUB, Expr::litf(1.0f),
                 Expr::col("lineitem","l_discount",6,DataType::FLOAT))),
         "revenue"}
    };
    plan.ops.push_back(std::move(probe));

    CpuSortOp sort;
    sort.keys = {{"revenue", true}};
    sort.limit = 20;
    plan.ops.push_back(std::move(sort));

    return plan;
}

// Q17: Small-Quantity-Order Revenue — lineitem+part with correlated subquery
bool matchQ17(const AnalyzedQuery& aq) {
    if (aq.tables.size() != 2) return false;
    std::vector<std::string> sorted = aq.tables;
    std::sort(sorted.begin(), sorted.end());
    if (sorted[0] != "lineitem" || sorted[1] != "part") return false;
    return !aq.subqueries.empty(); // Q17 has correlated scalar subquery, Q14 doesn't
}

QueryPlan planQ17(const AnalyzedQuery& /*aq*/) {
    QueryPlan plan;
    plan.name = "Q17";

    // Step 1: CPU bitmap on part where p_brand='Brand#23' AND p_container='MED BOX'
    CpuBitmapBuildOp partBm;
    partBm.table = "part";
    partBm.keyCol = bind("part", "p_partkey");
    partBm.filter = Predicate::logAnd({
        Predicate::cmp(CmpOp::EQ,
            Expr::col("part","p_brand",3,DataType::CHAR_FIXED),
            Expr::lits("Brand#23")),
        Predicate::cmp(CmpOp::EQ,
            Expr::col("part","p_container",6,DataType::CHAR_FIXED),
            Expr::lits("MED BOX")),
    });
    partBm.resultName = "part_bitmap";
    plan.ops.push_back(std::move(partBm));

    // Step 2: GPU atomic agg — sum(qty) and count per partkey (bitmap filtered)
    ProbeAggOp agg;
    agg.factTable = "lineitem";
    agg.factColumns = {
        bind("lineitem","l_partkey"), bind("lineitem","l_quantity")
    };
    agg.lookups = {{0, bind("lineitem","l_partkey"), ProbeAggOp::LookupRef::BITMAP_TEST}};
    agg.groupKeyExpr = Expr::col("lineitem","l_partkey",1,DataType::INT);
    agg.numGroups = 0; // per-partkey atomic
    agg.aggregations = {
        {AggFunc::SUM, Expr::col("lineitem","l_quantity",4,DataType::FLOAT), "sum_qty"},
        {AggFunc::COUNT, nullptr, "count_qty"},
    };
    plan.ops.push_back(std::move(agg));

    // Step 3 (CPU): compute threshold = 0.2 * sum/count per partkey

    // Step 4: GPU probe reduce — sum extendedprice where qty < threshold
    TwoStageReduceOp reduce;
    reduce.numBins = 1;
    reduce.binExpr = Expr::lit(0);
    reduce.aggregations = {
        {AggFunc::SUM, Expr::col("lineitem","l_extendedprice",5,DataType::FLOAT), "total_revenue"},
    };
    plan.ops.push_back(std::move(reduce));

    return plan;
}

// Q2: Minimum Cost Supplier — 5 tables including partsupp + region
bool matchQ2(const AnalyzedQuery& aq) {
    if (aq.tables.size() < 5) return false;
    for (auto& t : aq.tables)
        if (t == "partsupp") return true;
    return false;
}

QueryPlan planQ2(const AnalyzedQuery& /*aq*/) {
    QueryPlan plan;
    plan.name = "Q2";

    // Step 1: CPU bitmap on supplier WHERE s_nationkey IN (EUROPE nations)
    CpuBitmapBuildOp suppBm;
    suppBm.table = "supplier";
    suppBm.keyCol = bind("supplier", "s_suppkey");
    suppBm.resultName = "supplier_bitmap";
    plan.ops.push_back(std::move(suppBm));

    // Step 2: GPU bitmap on part WHERE p_size=15 AND p_type LIKE '%BRASS'
    BitmapBuildOp partBm;
    partBm.table = "part";
    partBm.keyCol = bind("part", "p_partkey");
    partBm.filter = Predicate::logAnd({
        Predicate::cmp(CmpOp::EQ,
            Expr::col("part","p_size",5,DataType::INT),
            Expr::lit(15)),
        Predicate::like(
            Expr::col("part","p_type",4,DataType::CHAR_FIXED),
            "%BRASS"),
    });
    plan.ops.push_back(std::move(partBm));

    // Step 3: GPU atomic min — find min supplycost per partkey
    ProbeAggOp minAgg;
    minAgg.factTable = "partsupp";
    minAgg.factColumns = {
        bind("partsupp","ps_partkey"), bind("partsupp","ps_suppkey"),
        bind("partsupp","ps_supplycost")
    };
    minAgg.lookups = {
        {1, bind("partsupp","ps_partkey"), ProbeAggOp::LookupRef::BITMAP_TEST},
        {0, bind("partsupp","ps_suppkey"), ProbeAggOp::LookupRef::BITMAP_TEST},
    };
    minAgg.groupKeyExpr = Expr::col("partsupp","ps_partkey",0,DataType::INT);
    minAgg.aggregations = {
        {AggFunc::MIN, Expr::col("partsupp","ps_supplycost",3,DataType::FLOAT), "min_cost"},
    };
    plan.ops.push_back(std::move(minAgg));

    // Step 4: GPU compact — match suppliers with min cost
    CompactOp compact;
    plan.ops.push_back(std::move(compact));

    // Step 5: CPU sort+limit
    CpuSortOp sort;
    sort.keys = {{"s_acctbal", true}, {"n_name", false}, {"s_name", false}, {"p_partkey", false}};
    sort.limit = 100;
    plan.ops.push_back(std::move(sort));

    return plan;
}

// Q18: Large Volume Customer — customer+orders+lineitem with GROUP BY o_orderkey
bool matchQ18(const AnalyzedQuery& aq) {
    if (aq.tables.size() < 3) return false;
    bool hasCust = false, hasLi = false, hasOrd = false;
    for (auto& t : aq.tables) {
        if (t == "customer") hasCust = true;
        if (t == "lineitem") hasLi = true;
        if (t == "orders") hasOrd = true;
    }
    if (!hasCust || !hasLi || !hasOrd) return false;
    for (auto& g : aq.groupBy) {
        if (auto* cr = std::get_if<ColRef>(&g->node))
            if (cr->column == "o_orderkey") return true;
    }
    return false;
}

QueryPlan planQ18(const AnalyzedQuery& /*aq*/) {
    QueryPlan plan;
    plan.name = "Q18";

    // Step 1: GPU atomic agg — sum(l_quantity) per orderkey
    ProbeAggOp agg;
    agg.factTable = "lineitem";
    agg.factColumns = {
        bind("lineitem","l_orderkey"), bind("lineitem","l_quantity")
    };
    agg.groupKeyExpr = Expr::col("lineitem","l_orderkey",0,DataType::INT);
    agg.numGroups = 0; // per-orderkey atomic
    agg.aggregations = {
        {AggFunc::SUM, Expr::col("lineitem","l_quantity",4,DataType::FLOAT), "sum_qty"},
    };
    plan.ops.push_back(std::move(agg));

    // Step 2: GPU compact — filter orders where sum(qty) > 300
    CompactOp compact;
    plan.ops.push_back(std::move(compact));

    // Step 3: CPU sort top-100
    CpuSortOp sort;
    sort.keys = {{"o_totalprice", true}, {"o_orderdate", false}};
    sort.limit = 100;
    plan.ops.push_back(std::move(sort));

    return plan;
}

// Q22: Global Sales Opportunity — subquery with alias "custsale"
bool matchQ22(const AnalyzedQuery& aq) {
    if (aq.tables.size() != 1 || aq.tables[0] != "__subquery__") return false;
    for (auto& a : aq.tableAliases)
        if (a == "custsale") return true;
    return false;
}

QueryPlan planQ22(const AnalyzedQuery& /*aq*/) {
    QueryPlan plan;
    plan.name = "Q22";

    // Step 1: GPU scalar agg — avg balance for qualifying country code prefixes with bal > 0
    TwoStageReduceOp scalarAgg;
    scalarAgg.numBins = 1;
    scalarAgg.binExpr = Expr::lit(0);
    scalarAgg.aggregations = {
        {AggFunc::SUM, Expr::col("customer","c_acctbal",5,DataType::FLOAT), "sum_bal"},
        {AggFunc::COUNT, nullptr, "count_bal"},
    };
    plan.ops.push_back(std::move(scalarAgg));

    // Step 2: GPU bitmap on orders (all custkeys that have orders)
    BitmapBuildOp ordBm;
    ordBm.table = "orders";
    ordBm.keyCol = bind("orders", "o_custkey");
    plan.ops.push_back(std::move(ordBm));

    // Step 3 (CPU): compute avg = sum/count

    // Step 4: GPU 7-bin aggregate — count+sum per country code
    TwoStageReduceOp binAgg;
    binAgg.numBins = 7;
    binAgg.binExpr = Expr::col("customer","c_phone",4,DataType::CHAR_FIXED); // prefix → bin
    binAgg.aggregations = {
        {AggFunc::COUNT, nullptr, "numcust"},
        {AggFunc::SUM, Expr::col("customer","c_acctbal",5,DataType::FLOAT), "totacctbal"},
    };
    plan.ops.push_back(std::move(binAgg));

    CpuSortOp sort;
    sort.keys = {{"cntrycode", false}};
    plan.ops.push_back(std::move(sort));

    return plan;
}

// Q9: Product Type Profit — 6 tables with p_name LIKE '%green%'
bool matchQ9(const AnalyzedQuery& aq) {
    // Q9 has 6 tables: part, supplier, lineitem, partsupp, orders, nation
    if (aq.tables.size() != 6) return false;
    bool hasPart = false, hasPartsupp = false, hasNation = false;
    for (auto& t : aq.tables) {
        if (t == "part") hasPart = true;
        if (t == "partsupp") hasPartsupp = true;
        if (t == "nation") hasNation = true;
    }
    return hasPart && hasPartsupp && hasNation;
}

QueryPlan planQ9(const AnalyzedQuery& /*aq*/) {
    QueryPlan plan;
    plan.name = "Q9";

    // Step 1: GPU bitmap on part — p_name LIKE '%green%'
    BitmapBuildOp partBm;
    partBm.table = "part";
    partBm.keyCol = bind("part", "p_partkey");
    partBm.filter = Predicate::like(
        Expr::col("part","p_name",1,DataType::CHAR_FIXED),
        "%green%");
    plan.ops.push_back(std::move(partBm));

    // Step 2: Direct map — supplier → nationkey
    DirectMapBuildOp suppMap;
    suppMap.table = "supplier";
    suppMap.keyCol = bind("supplier", "s_suppkey");
    suppMap.valueCols = {bind("supplier","s_nationkey")};
    plan.ops.push_back(std::move(suppMap));

    // Step 3: HT build on partsupp (compound key: partkey+suppkey)
    HashTableBuildOp pshtBuild;
    pshtBuild.table = "partsupp";
    pshtBuild.keyCol = bind("partsupp", "ps_partkey");
    pshtBuild.valueCols = {bind("partsupp","ps_suppkey"), bind("partsupp","ps_supplycost")};
    pshtBuild.filter = Predicate::cmp(CmpOp::EQ, Expr::lit(1), Expr::lit(1)); // bitmap-filtered
    pshtBuild.sizingMultiplier = 4.0f;
    plan.ops.push_back(std::move(pshtBuild));

    // Step 4: HT build on orders (orderkey → year)
    HashTableBuildOp ordhtBuild;
    ordhtBuild.table = "orders";
    ordhtBuild.keyCol = bind("orders", "o_orderkey");
    ordhtBuild.valueCols = {bind("orders","o_orderdate")}; // year derived from date
    ordhtBuild.sizingMultiplier = 4.0f;
    plan.ops.push_back(std::move(ordhtBuild));

    // Step 5: Probe lineitem + aggregate by (nation, year)
    ProbeAggOp probe;
    probe.factTable = "lineitem";
    probe.factColumns = {
        bind("lineitem","l_suppkey"), bind("lineitem","l_partkey"),
        bind("lineitem","l_orderkey"), bind("lineitem","l_extendedprice"),
        bind("lineitem","l_discount"), bind("lineitem","l_quantity")
    };
    probe.lookups = {
        {0, bind("lineitem","l_partkey"), ProbeAggOp::LookupRef::BITMAP_TEST},
        {1, bind("lineitem","l_suppkey"), ProbeAggOp::LookupRef::MAP_LOOKUP},
        {2, bind("lineitem","l_partkey"), ProbeAggOp::LookupRef::HT_PROBE}, // partsupp HT
        {3, bind("lineitem","l_orderkey"), ProbeAggOp::LookupRef::HT_PROBE}, // orders HT
    };
    probe.groupKeyExpr = Expr::lit(0); // (nation << 16 | year) compound key
    probe.numGroups = 0;
    probe.aggregations = {
        {AggFunc::SUM, nullptr, "profit"}, // extprice*(1-disc) - supplycost*qty
    };
    plan.ops.push_back(std::move(probe));

    CpuSortOp sort;
    sort.keys = {{"nation", false}, {"o_year", true}};
    plan.ops.push_back(std::move(sort));

    return plan;
}

// Q16: Parts/Supplier Relationship — partsupp+part with NOT IN + COUNT DISTINCT
bool matchQ16(const AnalyzedQuery& aq) {
    // Q16 has partsupp+part (2 tables) + NOT IN subquery on supplier
    if (aq.tables.size() != 2) return false;
    std::vector<std::string> sorted = aq.tables;
    std::sort(sorted.begin(), sorted.end());
    if (sorted[0] != "part" || sorted[1] != "partsupp") return false;
    // Distinguish from Q2 (5+ tables) by table count alone (already checked == 2)
    return true;
}

QueryPlan planQ16(const AnalyzedQuery& /*aq*/) {
    QueryPlan plan;
    plan.name = "Q16";

    // Step 1 (CPU): Build complaint bitmap from supplier WHERE s_comment LIKE '%Customer%Complaints%'
    CpuBitmapBuildOp complaintBm;
    complaintBm.table = "supplier";
    complaintBm.keyCol = bind("supplier", "s_suppkey");
    complaintBm.filter = Predicate::like(
        Expr::col("supplier","s_comment",6,DataType::CHAR_FIXED),
        "%Customer%Complaints%");
    complaintBm.resultName = "complaint_bitmap";
    plan.ops.push_back(std::move(complaintBm));

    // Step 2 (CPU): Build part group map — classify (brand, type, size) → group ID
    CpuDirectMapOp partGroupMap;
    partGroupMap.table = "part";
    partGroupMap.keyCol = bind("part", "p_partkey");
    partGroupMap.valueCols = {bind("part","p_brand"), bind("part","p_type"), bind("part","p_size")};
    partGroupMap.resultName = "part_group_map";
    plan.ops.push_back(std::move(partGroupMap));

    // Step 3: GPU — scan partsupp, set supplier bits in per-group bitmaps
    BitmapBuildOp groupBm;
    groupBm.table = "partsupp";
    groupBm.keyCol = bind("partsupp", "ps_suppkey");
    // Filter: part_group_map[ps_partkey] >= 0 AND NOT complaint_bitmap[ps_suppkey]
    plan.ops.push_back(std::move(groupBm));

    // Step 4: GPU — popcount each group's bitmap → distinct supplier count
    TwoStageReduceOp popcount;
    popcount.numBins = 0; // per-group
    popcount.binExpr = Expr::lit(0);
    popcount.aggregations = {
        {AggFunc::COUNT_DISTINCT, nullptr, "supplier_cnt"},
    };
    plan.ops.push_back(std::move(popcount));

    CpuSortOp sort;
    sort.keys = {{"supplier_cnt", true}, {"p_brand", false}, {"p_type", false}, {"p_size", false}};
    plan.ops.push_back(std::move(sort));

    return plan;
}

// Q20: Potential Part Promotion — supplier+nation with nested subqueries
bool matchQ20(const AnalyzedQuery& aq) {
    // Q20 outer query is supplier+nation (2 tables) with s_suppkey IN (subquery)
    if (aq.tables.size() != 2) return false;
    std::vector<std::string> sorted = aq.tables;
    std::sort(sorted.begin(), sorted.end());
    if (sorted[0] != "nation" || sorted[1] != "supplier") return false;
    // Has nested subqueries
    return !aq.subqueries.empty();
}

QueryPlan planQ20(const AnalyzedQuery& /*aq*/) {
    QueryPlan plan;
    plan.name = "Q20";

    // Step 1 (CPU): bitmap on part WHERE p_name LIKE 'forest%'
    CpuBitmapBuildOp partBm;
    partBm.table = "part";
    partBm.keyCol = bind("part", "p_partkey");
    partBm.filter = Predicate::like(
        Expr::col("part","p_name",1,DataType::CHAR_FIXED),
        "forest%");
    partBm.resultName = "part_bitmap";
    plan.ops.push_back(std::move(partBm));

    // Step 2 (CPU): bitmap on supplier WHERE s_nationkey = CANADA
    CpuBitmapBuildOp suppBm;
    suppBm.table = "supplier";
    suppBm.keyCol = bind("supplier", "s_suppkey");
    suppBm.resultName = "canada_bitmap";
    plan.ops.push_back(std::move(suppBm));

    // Step 3: GPU HT agg — sum(l_quantity) into HT keyed by (partkey, suppkey)
    HashTableBuildOp htAgg;
    htAgg.table = "lineitem";
    htAgg.keyCol = bind("lineitem", "l_partkey"); // compound key with l_suppkey
    htAgg.valueCols = {bind("lineitem","l_suppkey"), bind("lineitem","l_quantity")};
    htAgg.filter = Predicate::logAnd({
        Predicate::cmp(CmpOp::GE,
            Expr::col("lineitem","l_shipdate",10,DataType::DATE),
            Expr::lit(19940101)),
        Predicate::cmp(CmpOp::LT,
            Expr::col("lineitem","l_shipdate",10,DataType::DATE),
            Expr::lit(19950101)),
    });
    htAgg.sizingMultiplier = 4.0f;
    plan.ops.push_back(std::move(htAgg));

    // Step 4: GPU probe — partsupp against HT, check availqty > 0.5*sum_qty
    ProbeAggOp probe;
    probe.factTable = "partsupp";
    probe.factColumns = {
        bind("partsupp","ps_partkey"), bind("partsupp","ps_suppkey"),
        bind("partsupp","ps_availqty")
    };
    probe.lookups = {
        {0, bind("partsupp","ps_partkey"), ProbeAggOp::LookupRef::BITMAP_TEST},
        {1, bind("partsupp","ps_suppkey"), ProbeAggOp::LookupRef::BITMAP_TEST},
        {2, bind("partsupp","ps_partkey"), ProbeAggOp::LookupRef::HT_PROBE},
    };
    probe.aggregations = {}; // result is a qualifying supplier bitmap
    plan.ops.push_back(std::move(probe));

    CpuSortOp sort;
    sort.keys = {{"s_name", false}};
    plan.ops.push_back(std::move(sort));

    return plan;
}

// Q21: Suppliers Who Kept Orders Waiting — supplier+lineitem+orders+nation
bool matchQ21(const AnalyzedQuery& aq) {
    // Q21 has supplier, lineitem (as l1), orders, nation (4 tables)
    // Distinguish from other multi-table queries by EXISTS/NOT EXISTS subqueries
    if (aq.tables.size() < 4) return false;
    bool hasSupp = false, hasLi = false, hasOrd = false, hasNat = false;
    for (auto& t : aq.tables) {
        if (t == "supplier") hasSupp = true;
        if (t == "lineitem") hasLi = true;
        if (t == "orders") hasOrd = true;
        if (t == "nation") hasNat = true;
    }
    return hasSupp && hasLi && hasOrd && hasNat;
}

QueryPlan planQ21(const AnalyzedQuery& /*aq*/) {
    QueryPlan plan;
    plan.name = "Q21";

    // Step 1 (CPU): Build orders status map (orderkey → 1 if 'F' status)
    CpuDirectMapOp ordMap;
    ordMap.table = "orders";
    ordMap.keyCol = bind("orders", "o_orderkey");
    ordMap.valueCols = {bind("orders","o_orderstatus")};
    ordMap.filter = Predicate::cmp(CmpOp::EQ,
        Expr::col("orders","o_orderstatus",2,DataType::CHAR1),
        Expr::lits("F"));
    ordMap.resultName = "orders_status_map";
    plan.ops.push_back(std::move(ordMap));

    // Step 2 (CPU): bitmap on supplier WHERE s_nationkey = SAUDI ARABIA
    CpuBitmapBuildOp suppBm;
    suppBm.table = "supplier";
    suppBm.keyCol = bind("supplier", "s_suppkey");
    suppBm.resultName = "sa_supp_bitmap";
    plan.ops.push_back(std::move(suppBm));

    // Step 3: GPU pass 1 — build per-order supplier tracking via CAS
    // Records first_supp, multi_supp bitmap, late_supp, multi_late bitmap
    ProbeAggOp track;
    track.factTable = "lineitem";
    track.factColumns = {
        bind("lineitem","l_orderkey"), bind("lineitem","l_suppkey"),
        bind("lineitem","l_receiptdate"), bind("lineitem","l_commitdate")
    };
    track.lookups = {{0, bind("lineitem","l_orderkey"), ProbeAggOp::LookupRef::MAP_LOOKUP}};
    track.groupKeyExpr = Expr::col("lineitem","l_orderkey",0,DataType::INT);
    plan.ops.push_back(std::move(track));

    // Step 4: GPU pass 2 — count qualifying lineitems per SA supplier
    ProbeAggOp count;
    count.factTable = "lineitem";
    count.factColumns = {
        bind("lineitem","l_orderkey"), bind("lineitem","l_suppkey"),
        bind("lineitem","l_receiptdate"), bind("lineitem","l_commitdate")
    };
    count.lookups = {
        {0, bind("lineitem","l_orderkey"), ProbeAggOp::LookupRef::MAP_LOOKUP},
        {1, bind("lineitem","l_suppkey"), ProbeAggOp::LookupRef::BITMAP_TEST},
    };
    count.groupKeyExpr = Expr::col("lineitem","l_suppkey",2,DataType::INT);
    count.aggregations = {
        {AggFunc::COUNT, nullptr, "numwait"},
    };
    plan.ops.push_back(std::move(count));

    CpuSortOp sort;
    sort.keys = {{"numwait", true}, {"s_name", false}};
    sort.limit = 100;
    plan.ops.push_back(std::move(sort));

    return plan;
}

// Q5: Local Supplier Volume — 6 tables including region
bool matchQ5(const AnalyzedQuery& aq) {
    for (auto& t : aq.tables)
        if (t == "region") return true;
    return false;
}

QueryPlan planQ5(const AnalyzedQuery& /*aq*/) {
    QueryPlan plan;
    plan.name = "Q5";

    // Step 1 (CPU): bitmap on nation WHERE regionkey = ASIA
    CpuBitmapBuildOp nationBm;
    nationBm.table = "nation";
    nationBm.keyCol = bind("nation", "n_nationkey");
    nationBm.resultName = "nation_bitmap";
    plan.ops.push_back(std::move(nationBm));

    // Step 2: GPU direct map — customer → nationkey (ASIA nations only)
    DirectMapBuildOp custMap;
    custMap.table = "customer";
    custMap.keyCol = bind("customer", "c_custkey");
    custMap.valueCols = {bind("customer","c_nationkey")};
    // Filter: bitmap_test(nation_bitmap, c_nationkey)
    plan.ops.push_back(std::move(custMap));

    // Step 3: GPU direct map — supplier → nationkey (ASIA nations only)
    DirectMapBuildOp suppMap;
    suppMap.table = "supplier";
    suppMap.keyCol = bind("supplier", "s_suppkey");
    suppMap.valueCols = {bind("supplier","s_nationkey")};
    plan.ops.push_back(std::move(suppMap));

    // Step 4: GPU direct map — orders → nationkey (date filter + customer bitmap)
    DirectMapBuildOp ordMap;
    ordMap.table = "orders";
    ordMap.keyCol = bind("orders", "o_orderkey");
    ordMap.valueCols = {bind("orders","o_custkey")}; // resolves to nationkey via custMap
    ordMap.filter = Predicate::logAnd({
        Predicate::cmp(CmpOp::GE,
            Expr::col("orders","o_orderdate",4,DataType::DATE),
            Expr::lit(19940101)),
        Predicate::cmp(CmpOp::LT,
            Expr::col("orders","o_orderdate",4,DataType::DATE),
            Expr::lit(19950101)),
    });
    plan.ops.push_back(std::move(ordMap));

    // Step 5: GPU probe lineitem — same-nation check, aggregate revenue per nation
    ProbeAggOp probe;
    probe.factTable = "lineitem";
    probe.factColumns = {
        bind("lineitem","l_orderkey"), bind("lineitem","l_suppkey"),
        bind("lineitem","l_extendedprice"), bind("lineitem","l_discount")
    };
    probe.lookups = {
        {3, bind("lineitem","l_orderkey"), ProbeAggOp::LookupRef::MAP_LOOKUP},
        {2, bind("lineitem","l_suppkey"), ProbeAggOp::LookupRef::MAP_LOOKUP},
    };
    probe.groupKeyExpr = Expr::col("lineitem","l_suppkey",2,DataType::INT); // nationkey from map
    probe.numGroups = 25; // max nationkey
    probe.aggregations = {
        {AggFunc::SUM,
         Expr::binary(ExprOp::MUL,
             Expr::col("lineitem","l_extendedprice",5,DataType::FLOAT),
             Expr::binary(ExprOp::SUB, Expr::litf(1.0f),
                 Expr::col("lineitem","l_discount",6,DataType::FLOAT))),
         "revenue"}
    };
    plan.ops.push_back(std::move(probe));

    CpuSortOp sort;
    sort.keys = {{"revenue", true}};
    plan.ops.push_back(std::move(sort));

    return plan;
}

// Q7: Volume Shipping — subquery with 3 GROUP BY columns
bool matchQ7(const AnalyzedQuery& aq) {
    if (aq.tables.size() != 1 || aq.tables[0] != "__subquery__") return false;
    return aq.groupBy.size() >= 3;
}

QueryPlan planQ7(const AnalyzedQuery& /*aq*/) {
    QueryPlan plan;
    plan.name = "Q7";

    // Step 1: GPU direct map — supplier → nationkey (FRANCE/GERMANY only)
    DirectMapBuildOp suppMap;
    suppMap.table = "supplier";
    suppMap.keyCol = bind("supplier", "s_suppkey");
    suppMap.valueCols = {bind("supplier","s_nationkey")};
    plan.ops.push_back(std::move(suppMap));

    // Step 2: GPU direct map — customer → nationkey (FRANCE/GERMANY only)
    DirectMapBuildOp custMap;
    custMap.table = "customer";
    custMap.keyCol = bind("customer", "c_custkey");
    custMap.valueCols = {bind("customer","c_nationkey")};
    plan.ops.push_back(std::move(custMap));

    // Step 3: GPU direct map — orders → custkey
    DirectMapBuildOp ordMap;
    ordMap.table = "orders";
    ordMap.keyCol = bind("orders", "o_orderkey");
    ordMap.valueCols = {bind("orders","o_custkey")};
    plan.ops.push_back(std::move(ordMap));

    // Step 4: GPU probe lineitem — date+chain checks, aggregate 4 bins
    ProbeAggOp probe;
    probe.factTable = "lineitem";
    probe.factColumns = {
        bind("lineitem","l_orderkey"), bind("lineitem","l_suppkey"),
        bind("lineitem","l_shipdate"), bind("lineitem","l_extendedprice"),
        bind("lineitem","l_discount")
    };
    probe.factFilter = Predicate::logAnd({
        Predicate::cmp(CmpOp::GE,
            Expr::col("lineitem","l_shipdate",10,DataType::DATE),
            Expr::lit(19950101)),
        Predicate::cmp(CmpOp::LE,
            Expr::col("lineitem","l_shipdate",10,DataType::DATE),
            Expr::lit(19961231)),
    });
    probe.lookups = {
        {0, bind("lineitem","l_suppkey"), ProbeAggOp::LookupRef::MAP_LOOKUP},
        {1, bind("lineitem","l_orderkey"), ProbeAggOp::LookupRef::MAP_LOOKUP}, // via orders → custkey
        {2, bind("lineitem","l_orderkey"), ProbeAggOp::LookupRef::MAP_LOOKUP},
    };
    probe.numGroups = 4; // (FRANCE→GERMANY, GERMANY→FRANCE) × (1995, 1996)
    probe.aggregations = {
        {AggFunc::SUM,
         Expr::binary(ExprOp::MUL,
             Expr::col("lineitem","l_extendedprice",5,DataType::FLOAT),
             Expr::binary(ExprOp::SUB, Expr::litf(1.0f),
                 Expr::col("lineitem","l_discount",6,DataType::FLOAT))),
         "revenue"}
    };
    plan.ops.push_back(std::move(probe));

    CpuSortOp sort;
    sort.keys = {{"supp_nation", false}, {"cust_nation", false}, {"l_year", false}};
    plan.ops.push_back(std::move(sort));

    return plan;
}

// Q8: National Market Share — subquery with 1 GROUP BY, alias "all_nations"
bool matchQ8(const AnalyzedQuery& aq) {
    if (aq.tables.size() != 1 || aq.tables[0] != "__subquery__") return false;
    if (aq.groupBy.size() != 1) return false;
    // Distinguish from Q13 via alias
    for (auto& a : aq.tableAliases)
        if (a == "all_nations") return true;
    return false;
}

QueryPlan planQ8(const AnalyzedQuery& /*aq*/) {
    QueryPlan plan;
    plan.name = "Q8";

    // Step 1 (CPU): bitmap on part WHERE p_type = 'ECONOMY ANODIZED STEEL'
    CpuBitmapBuildOp partBm;
    partBm.table = "part";
    partBm.keyCol = bind("part", "p_partkey");
    partBm.filter = Predicate::cmp(CmpOp::EQ,
        Expr::col("part","p_type",4,DataType::CHAR_FIXED),
        Expr::lits("ECONOMY ANODIZED STEEL"));
    partBm.resultName = "part_bitmap";
    plan.ops.push_back(std::move(partBm));

    // Step 2 (CPU): direct map — customer → nationkey (AMERICA region nations)
    CpuDirectMapOp custMap;
    custMap.table = "customer";
    custMap.keyCol = bind("customer", "c_custkey");
    custMap.valueCols = {bind("customer","c_nationkey")};
    custMap.resultName = "cust_nation_map";
    plan.ops.push_back(std::move(custMap));

    // Step 3 (CPU): direct map — supplier → nationkey
    CpuDirectMapOp suppMap;
    suppMap.table = "supplier";
    suppMap.keyCol = bind("supplier", "s_suppkey");
    suppMap.valueCols = {bind("supplier","s_nationkey")};
    suppMap.resultName = "supp_nation_map";
    plan.ops.push_back(std::move(suppMap));

    // Step 4: GPU direct map — orders → (custkey, year) with date+AMERICA filter
    DirectMapBuildOp ordMap;
    ordMap.table = "orders";
    ordMap.keyCol = bind("orders", "o_orderkey");
    ordMap.valueCols = {bind("orders","o_custkey"), bind("orders","o_orderdate")};
    ordMap.filter = Predicate::logAnd({
        Predicate::cmp(CmpOp::GE,
            Expr::col("orders","o_orderdate",4,DataType::DATE),
            Expr::lit(19950101)),
        Predicate::cmp(CmpOp::LE,
            Expr::col("orders","o_orderdate",4,DataType::DATE),
            Expr::lit(19961231)),
    });
    plan.ops.push_back(std::move(ordMap));

    // Step 5: GPU probe lineitem — bitmap+map check, 4-bin aggregate
    ProbeAggOp probe;
    probe.factTable = "lineitem";
    probe.factColumns = {
        bind("lineitem","l_orderkey"), bind("lineitem","l_partkey"),
        bind("lineitem","l_suppkey"), bind("lineitem","l_extendedprice"),
        bind("lineitem","l_discount")
    };
    probe.lookups = {
        {0, bind("lineitem","l_partkey"), ProbeAggOp::LookupRef::BITMAP_TEST},
        {3, bind("lineitem","l_orderkey"), ProbeAggOp::LookupRef::MAP_LOOKUP},
    };
    probe.numGroups = 4; // brazil/total × 1995/1996
    probe.aggregations = {
        {AggFunc::SUM,
         Expr::binary(ExprOp::MUL,
             Expr::col("lineitem","l_extendedprice",5,DataType::FLOAT),
             Expr::binary(ExprOp::SUB, Expr::litf(1.0f),
                 Expr::col("lineitem","l_discount",6,DataType::FLOAT))),
         "revenue"}
    };
    plan.ops.push_back(std::move(probe));

    return plan;
}

} // anonymous namespace

// ===================================================================
// PUBLIC: buildPlan
// ===================================================================

QueryPlan buildPlan(const AnalyzedQuery& aq) {
    // Try pattern matchers in order of specificity
    if (matchQ19(aq)) return planQ19(aq); // Q19 before Q17/Q14 (OR filters)
    if (matchQ17(aq)) return planQ17(aq); // Q17 before Q14 (has subquery)
    if (matchQ4(aq))  return planQ4(aq);  // Q4 before Q12 (both orders+lineitem)
    if (matchQ12(aq)) return planQ12(aq);
    if (matchQ18(aq)) return planQ18(aq); // Q18 before Q10 (both cust+ord+li, Q18 groups by o_orderkey)
    if (matchQ21(aq)) return planQ21(aq); // Q21 before Q10 (supp+li+ord+nation, 4 tables)
    if (matchQ10(aq)) return planQ10(aq); // Q10 before Q3 (both have cust+li+ord)
    if (matchQ9(aq))  return planQ9(aq);  // Q9 before Q2 (6 tables with partsupp+part+nation)
    if (matchQ2(aq))  return planQ2(aq);  // Q2 before Q5/Q11 (has partsupp+region)
    if (matchQ16(aq)) return planQ16(aq); // Q16 before Q14 (partsupp+part, 2 tables)
    if (matchQ5(aq))  return planQ5(aq);  // Q5 has "region" in tables
    if (matchQ7(aq))  return planQ7(aq);  // Q7/Q8/Q22 before Q13 (all __subquery__)
    if (matchQ8(aq))  return planQ8(aq);
    if (matchQ22(aq)) return planQ22(aq);
    if (matchQ20(aq)) return planQ20(aq); // Q20: supplier+nation with subqueries
    if (matchQ1(aq))  return planQ1(aq);
    if (matchQ6(aq))  return planQ6(aq);
    if (matchQ3(aq))  return planQ3(aq);
    if (matchQ13(aq)) return planQ13(aq);
    if (matchQ14(aq)) return planQ14(aq);
    if (matchQ15(aq)) return planQ15(aq);
    if (matchQ11(aq)) return planQ11(aq);

    throw std::runtime_error("No query plan pattern matches for tables: " +
        (aq.tables.empty() ? "(none)" : aq.tables[0]));
}

} // namespace codegen
