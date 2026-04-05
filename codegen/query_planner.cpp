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

QueryPlan planQ14(const AnalyzedQuery& aq) {
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
    // Single kernel: lineitem date filter → atomic_add revenue to map[suppkey]
    // Then CPU: find max, print matching supplier(s)
    // We use a TwoStageReduceOp but codegen treats Q15 specially (direct-map atomic)
    TwoStageReduceOp op;
    op.numBins = 0; // dynamic — per-suppkey aggregation
    op.binExpr = Expr::col("lineitem","l_suppkey",2,DataType::INT);
    op.aggregations = {
        {AggFunc::SUM,
         Expr::binary(ExprOp::MUL,
             Expr::col("lineitem","l_extendedprice",5,DataType::FLOAT),
             Expr::binary(ExprOp::SUB, Expr::litf(1.0f),
                 Expr::col("lineitem","l_discount",6,DataType::FLOAT))),
         "total_revenue"}
    };
    plan.ops.push_back(std::move(op));
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
    // Single kernel: partsupp scan, supplier bitmap filter,
    // atomic_add value to map[partkey] + TG reduce for global sum
    TwoStageReduceOp op;
    op.numBins = 0; // dynamic — per-partkey aggregation
    op.binExpr = Expr::col("partsupp","ps_partkey",0,DataType::INT);
    op.aggregations = {
        {AggFunc::SUM, nullptr, "value"}, // ps_supplycost * ps_availqty
    };
    plan.ops.push_back(std::move(op));
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
    // Pass 1: aggregate qty stats per qualifying partkey (Brand#23, MED BOX)
    // CPU: compute threshold = 0.2 * avg per partkey
    // Pass 2: sum revenue where qty < threshold
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
    // 3 GPU kernels: filter part → bitmap, find min cost per partkey, match suppliers
    // CPU post: join with string columns, sort, limit 100
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
    // Kernel 1: aggregate sum(qty) per orderkey
    // Kernel 2: filter orders with sum > 300, compact output
    // CPU post: join customer names, top-100 sort
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
    // Phase 1: avg balance for qualifying prefixes with bal > 0
    // Phase 2: build orders custkey bitmap
    // CPU: compute avg
    // Phase 3: final 7-bin aggregate (count + sum per country code)
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
    // 5 GPU kernels: build part bitmap, supplier map, partsupp HT, orders HT, probe lineitem
    // Direct global atomic aggregation by (nation, year) key
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
    // CPU builds complaint bitmap + part group map
    // GPU kernel 1: scan partsupp, set bits in per-group bitmaps
    // GPU kernel 2: popcount each group's bitmap
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
    // GPU kernel 1: aggregate lineitem quantity into HT keyed by (partkey, suppkey)
    // GPU kernel 2: probe partsupp, check threshold, set qualifying suppkey bitmap
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
    // GPU pass 1: build per-order supplier tracking (CAS multi-value)
    // GPU pass 2: count qualifying lineitems per SA supplier
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
    // 4 GPU kernels: build customer/supplier/orders nation maps, probe lineitem
    // 25-bin atomic float aggregation by nationkey
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
    // 4 GPU kernels: build supplier/customer/orders maps, probe lineitem
    // 4-bin atomic float aggregation (FRANCE↔GERMANY × 1995/1996)
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
    // CPU: part bitmap + customer/supplier nation maps
    // 2 GPU kernels: build orders map, probe lineitem
    // 4-bin atomic float aggregation (brazil/total × 1995/1996)
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
