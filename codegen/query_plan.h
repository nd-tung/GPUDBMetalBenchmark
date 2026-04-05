#pragma once
#include <string>
#include <vector>
#include <variant>
#include <optional>
#include <memory>
#include <cstdint>

namespace codegen {

// ===================================================================
// DATA TYPES
// ===================================================================

enum class DataType { INT, FLOAT, DATE, CHAR1, CHAR_FIXED };

struct TypeInfo {
    DataType type;
    int fixedWidth = 0; // only for CHAR_FIXED
};

// ===================================================================
// EXPRESSIONS
// ===================================================================

enum class ExprOp { ADD, SUB, MUL, DIV };
enum class AggFunc { SUM, COUNT, AVG, MIN, MAX, COUNT_DISTINCT };
enum class CmpOp { EQ, NE, LT, LE, GT, GE };

struct ColRef {
    std::string table;    // e.g. "lineitem"
    std::string column;   // e.g. "l_shipdate"
    int         colIndex; // resolved TPC-H column index
    DataType    dataType;
};

struct Literal {
    std::variant<int, float, std::string> value;
};

struct Expr;
using ExprPtr = std::shared_ptr<Expr>;

struct BinaryExpr {
    ExprOp op;
    ExprPtr left, right;
};

struct CaseWhen {
    struct Branch { ExprPtr condition, result; };
    std::vector<Branch> branches;
    ExprPtr elseResult;
};

struct FuncCall {
    std::string name; // EXTRACT, SUBSTRING, etc.
    std::vector<ExprPtr> args;
};

struct Expr {
    std::variant<ColRef, Literal, BinaryExpr, CaseWhen, FuncCall> node;

    static ExprPtr col(const std::string& table, const std::string& col, int idx, DataType dt) {
        auto e = std::make_shared<Expr>();
        e->node = ColRef{table, col, idx, dt};
        return e;
    }
    static ExprPtr lit(int v) {
        auto e = std::make_shared<Expr>();
        e->node = Literal{v};
        return e;
    }
    static ExprPtr litf(float v) {
        auto e = std::make_shared<Expr>();
        e->node = Literal{v};
        return e;
    }
    static ExprPtr lits(const std::string& v) {
        auto e = std::make_shared<Expr>();
        e->node = Literal{v};
        return e;
    }
    static ExprPtr binary(ExprOp op, ExprPtr l, ExprPtr r) {
        auto e = std::make_shared<Expr>();
        e->node = BinaryExpr{op, l, r};
        return e;
    }
};

// ===================================================================
// PREDICATES
// ===================================================================

struct Predicate;
using PredPtr = std::shared_ptr<Predicate>;

struct Comparison {
    CmpOp op;
    ExprPtr left, right;
};

struct Between {
    ExprPtr expr, low, high;
};

struct InList {
    ExprPtr expr;
    std::vector<ExprPtr> values;
};

struct Like {
    ExprPtr expr;
    std::string pattern;
    bool negated = false;
};

struct LogicalAnd { std::vector<PredPtr> children; };
struct LogicalOr  { std::vector<PredPtr> children; };
struct LogicalNot { PredPtr child; };

struct ExistsPred {
    bool negated = false;
    // child plan index in QueryPlan::subqueries
    int subqueryIdx = -1;
};

struct Predicate {
    std::variant<Comparison, Between, InList, Like,
                 LogicalAnd, LogicalOr, LogicalNot, ExistsPred> node;

    static PredPtr cmp(CmpOp op, ExprPtr l, ExprPtr r) {
        auto p = std::make_shared<Predicate>();
        p->node = Comparison{op, l, r};
        return p;
    }
    static PredPtr between(ExprPtr e, ExprPtr lo, ExprPtr hi) {
        auto p = std::make_shared<Predicate>();
        p->node = Between{e, lo, hi};
        return p;
    }
    static PredPtr inList(ExprPtr e, std::vector<ExprPtr> vals) {
        auto p = std::make_shared<Predicate>();
        p->node = InList{e, std::move(vals)};
        return p;
    }
    static PredPtr like(ExprPtr e, const std::string& pat, bool neg = false) {
        auto p = std::make_shared<Predicate>();
        p->node = Like{e, pat, neg};
        return p;
    }
    static PredPtr logAnd(std::vector<PredPtr> ch) {
        auto p = std::make_shared<Predicate>();
        p->node = LogicalAnd{std::move(ch)};
        return p;
    }
    static PredPtr logOr(std::vector<PredPtr> ch) {
        auto p = std::make_shared<Predicate>();
        p->node = LogicalOr{std::move(ch)};
        return p;
    }
    static PredPtr logNot(PredPtr ch) {
        auto p = std::make_shared<Predicate>();
        p->node = LogicalNot{ch};
        return p;
    }
};

// ===================================================================
// AGGREGATION SPEC
// ===================================================================

struct AggSpec {
    AggFunc   func;
    ExprPtr   expr;       // null for COUNT(*)
    std::string alias;
};

// ===================================================================
// PLAN OPERATORS
// ===================================================================

enum class OpType {
    TABLE_SCAN,         // single-table scan with filter
    BITMAP_BUILD,       // build a bitmap from dimension table
    DIRECT_MAP_BUILD,   // build a direct-address map
    HT_BUILD,           // build a hash table (open-addressing CAS)
    PROBE_AGG,          // probe lookups + aggregation
    TWO_STAGE_REDUCE,   // low-cardinality TG-reduce + global atomic
    COMPACT,            // sparse HT → dense output
    CPU_SORT,           // CPU-side sort + top-K
    SUBQUERY,           // child query plan
    HISTOGRAM,          // count distribution (Q13 pattern)
    STRING_MATCH,       // LIKE pattern match (Q13 orders)
    CPU_BITMAP_BUILD,   // build bitmap on CPU (small dimension table)
    CPU_DIRECT_MAP,     // build direct map on CPU
};

struct ColumnBinding {
    std::string table;
    std::string column;
    int         colIndex;
    DataType    dataType;
    int         fixedWidth = 0;
};

// --- TableScanOp ---
struct TableScanOp {
    std::string table;
    std::vector<ColumnBinding> columns;
    PredPtr filter;
};

// --- BitmapBuildOp (GPU) ---
struct BitmapBuildOp {
    std::string table;
    ColumnBinding keyCol;
    PredPtr filter;
    int estimatedMaxKey = 0;
};

// --- CpuBitmapBuildOp ---
struct CpuBitmapBuildOp {
    std::string table;
    ColumnBinding keyCol;
    PredPtr filter;
    std::string resultName; // buffer name for reference
};

// --- DirectMapBuildOp ---
struct DirectMapBuildOp {
    std::string table;
    ColumnBinding keyCol;
    std::vector<ColumnBinding> valueCols;
    PredPtr filter;
    int estimatedMaxKey = 0;
};

// --- CpuDirectMapOp ---
struct CpuDirectMapOp {
    std::string table;
    ColumnBinding keyCol;
    std::vector<ColumnBinding> valueCols;
    PredPtr filter;
    std::string resultName;
};

// --- HashTableBuildOp ---
struct HashTableBuildOp {
    std::string table;
    ColumnBinding keyCol;
    std::vector<ColumnBinding> valueCols;
    PredPtr filter;
    float sizingMultiplier = 4.0f;
};

// --- ProbeAggOp ---
struct ProbeAggOp {
    std::string factTable;
    std::vector<ColumnBinding> factColumns;
    PredPtr factFilter;

    // What to probe against (references earlier build ops by index)
    struct LookupRef {
        int buildOpIndex;
        ColumnBinding probeKey;
        enum Type { BITMAP_TEST, MAP_LOOKUP, HT_PROBE } type;
    };
    std::vector<LookupRef> lookups;

    // Grouping and aggregation
    ExprPtr groupKeyExpr;
    int numGroups = 0;
    std::vector<AggSpec> aggregations;
    ExprPtr computeExpr; // expression computed per qualifying row
};

// --- TwoStageReduceOp ---
struct TwoStageReduceOp {
    int numBins;
    ExprPtr binExpr; // maps row → bin index
    std::vector<AggSpec> aggregations;
    // column buffers come from the scan in the same plan
};

// --- CompactOp ---
struct CompactOp {
    int maxSlots;
    // reads from previous HT output
};

// --- CpuSortOp ---
struct CpuSortOp {
    struct SortKey { std::string column; bool descending; };
    std::vector<SortKey> keys;
    int limit = -1; // -1 = no limit
};

// --- SubqueryOp ---
struct SubqueryOp {
    int childPlanIndex;
    enum ResultType { SCALAR, BITMAP, HASH_TABLE, COLUMN } resultType;
};

// --- HistogramOp ---
struct HistogramOp {
    int maxBuckets;
    // reads count-per-key from previous op, builds count distribution
};

// --- StringMatchOp ---
struct StringMatchOp {
    std::string table;
    ColumnBinding keyCol;
    ColumnBinding stringCol;
    std::string pattern;
    bool negated = false;
    int fixedWidth;
};

// ===================================================================
// PLAN OPERATOR (VARIANT)
// ===================================================================

using PlanOp = std::variant<
    TableScanOp, BitmapBuildOp, CpuBitmapBuildOp,
    DirectMapBuildOp, CpuDirectMapOp,
    HashTableBuildOp, ProbeAggOp, TwoStageReduceOp,
    CompactOp, CpuSortOp, SubqueryOp,
    HistogramOp, StringMatchOp
>;

// ===================================================================
// QUERY PLAN
// ===================================================================

struct QueryPlan {
    std::string name;             // e.g. "Q1", "Q6"
    std::vector<PlanOp> ops;      // executed in order
    std::vector<QueryPlan> subqueries; // referenced by SubqueryOp

    // Output format description
    struct OutputCol {
        std::string name;
        DataType type;
    };
    std::vector<OutputCol> outputSchema;
};

} // namespace codegen
