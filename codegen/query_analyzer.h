#pragma once
#include "query_plan.h"
#include "tpch_schema.h"
#include <string>
#include <vector>
#include <optional>

namespace codegen {

// ===================================================================
// ANALYZED QUERY — Extracted from SQL AST
// ===================================================================

struct JoinClause {
    std::string leftTable, rightTable;
    std::string leftCol, rightCol;
};

struct AggTarget {
    AggFunc func;
    ExprPtr innerExpr;  // the expression inside the aggregate
    std::string alias;
    bool isStar = false; // COUNT(*)
};

struct SelectTarget {
    ExprPtr expr;
    std::string alias;
    bool isAgg = false;
    std::optional<AggTarget> agg;
};

struct OrderByItem {
    ExprPtr expr;
    bool descending = false;
};

struct AnalyzedQuery {
    // FROM clause
    std::vector<std::string> tables;
    std::vector<std::string> tableAliases;

    // JOIN conditions (equi-joins extracted from WHERE or explicit JOIN ON)
    std::vector<JoinClause> joins;

    // WHERE predicates (non-join predicates, per-table filters)
    std::vector<PredPtr> filters;

    // SELECT target list
    std::vector<SelectTarget> targets;

    // GROUP BY
    std::vector<ExprPtr> groupBy;

    // HAVING
    PredPtr having;

    // ORDER BY
    std::vector<OrderByItem> orderBy;

    // LIMIT
    int limit = -1;

    // Subqueries (IN, EXISTS, scalar)
    struct Subquery {
        enum Type { IN_SUBQUERY, EXISTS_SUBQUERY, NOT_EXISTS_SUBQUERY, SCALAR_SUBQUERY };
        Type type;
        std::string sql; // raw SQL for re-parsing
        ExprPtr outerExpr; // for IN: the outer expression being tested
        AnalyzedQuery* analyzed = nullptr; // filled later
    };
    std::vector<Subquery> subqueries;

    // Helpers
    bool isSingleTable() const { return tables.size() == 1 && joins.empty(); }
    bool hasAggregation() const {
        for (auto& t : targets) if (t.isAgg) return true;
        return false;
    }
    bool hasGroupBy() const { return !groupBy.empty(); }
};

// ===================================================================
// PUBLIC API
// ===================================================================

// Parse a SQL string and extract structural information.
// Returns an AnalyzedQuery, or throws std::runtime_error on parse failure.
AnalyzedQuery analyzeSQL(const std::string& sql);

} // namespace codegen
