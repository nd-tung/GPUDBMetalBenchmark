#include "query_analyzer.h"

extern "C" {
#include "pg_query.h"
}
#include "../third_party/nlohmann/json.hpp"

#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <unordered_map>

using json = nlohmann::json;

namespace codegen {

// ===================================================================
// INTERNAL HELPERS
// ===================================================================

namespace {

// File-scope alias map: alias -> real table name (e.g. "l1" -> "lineitem")
std::unordered_map<std::string, std::string> g_aliasMap;

const auto& schema() { return TPCHSchema::instance(); }

// Resolve an unqualified column name to (table, column).
// If multiple tables have the column, we need the table list to disambiguate.
// Returns ("", colName) if not found in any table (could be a SELECT alias).
std::pair<std::string, std::string> resolveColumn(const std::string& colName,
                                                   const std::vector<std::string>& tables) {
    for (auto& t : tables) {
        auto it = schema().tables.find(t);
        if (it == schema().tables.end()) continue;
        if (it->second.nameToIdx.count(colName)) return {t, colName};
    }
    return {"", colName}; // alias or derived column
}

// Parse a date string like "1994-01-01" to YYYYMMDD integer
int parseDateLiteral(const std::string& s) {
    // Expecting YYYY-MM-DD
    if (s.size() >= 10 && s[4] == '-' && s[7] == '-') {
        int y = std::stoi(s.substr(0, 4));
        int m = std::stoi(s.substr(5, 2));
        int d = std::stoi(s.substr(8, 2));
        return y * 10000 + m * 100 + d;
    }
    throw std::runtime_error("Invalid date literal: " + s);
}

// ===================================================================
// AST WALKING
// ===================================================================

// Forward declarations
ExprPtr walkExpr(const json& node, const std::vector<std::string>& tables);
PredPtr walkPredicate(const json& node, const std::vector<std::string>& tables);

ExprPtr walkColumnRef(const json& node, const std::vector<std::string>& tables) {
    auto& fields = node["fields"];
    std::string colName;
    std::string tblQualifier;
    if (fields.size() == 1) {
        colName = fields[0]["String"]["sval"].get<std::string>();
    } else if (fields.size() == 2) {
        tblQualifier = fields[0]["String"]["sval"].get<std::string>();
        colName = fields[1]["String"]["sval"].get<std::string>();
    }

    std::string resolvedTable;
    if (!tblQualifier.empty()) {
        // Resolve alias to real table name if needed
        auto ait = g_aliasMap.find(tblQualifier);
        resolvedTable = (ait != g_aliasMap.end()) ? ait->second : tblQualifier;
    } else {
        auto [t, c] = resolveColumn(colName, tables);
        resolvedTable = t;
    }

    if (resolvedTable.empty()) {
        // This is a SELECT alias (e.g., "revenue" in ORDER BY) — return as unresolved ColRef
        return Expr::col("", colName, -1, DataType::INT);
    }

    auto& tbl = schema().table(resolvedTable);
    auto& col = tbl.col(colName);
    return Expr::col(resolvedTable, colName, col.index, col.type);
}

ExprPtr walkConst(const json& node) {
    if (node.contains("ival")) {
        auto& iv = node["ival"];
        if (iv.is_object() && iv.contains("ival"))
            return Expr::lit(iv["ival"].get<int>());
        return Expr::lit(0); // empty ival = integer 0
    }
    if (node.contains("fval")) {
        auto& fv = node["fval"];
        if (fv.is_object() && fv.contains("fval"))
            return Expr::litf(std::stof(fv["fval"].get<std::string>()));
        return Expr::litf(0.0f);
    }
    if (node.contains("sval")) {
        auto& sv = node["sval"];
        if (sv.is_object() && sv.contains("sval"))
            return Expr::lits(sv["sval"].get<std::string>());
        return Expr::lits("");
    }
    std::cerr << "WARN: unknown A_Const type: " << node.dump().substr(0, 200) << std::endl;
    return Expr::lit(0);
}

ExprPtr walkTypeCast(const json& node, const std::vector<std::string>& tables) {
    auto& typeName = node["typeName"];
    std::string typStr;
    if (typeName.contains("names")) {
        for (auto& n : typeName["names"]) {
            if (n.contains("String"))
                typStr = n["String"]["sval"].get<std::string>();
        }
    }

    auto arg = walkExpr(node["arg"], tables);
    // DATE cast: convert string literal to integer
    if (typStr == "date") {
        if (auto* lit = std::get_if<Literal>(&arg->node)) {
            if (auto* sv = std::get_if<std::string>(&lit->value)) {
                return Expr::lit(parseDateLiteral(*sv));
            }
        }
    }
    // INTERVAL cast: return raw integer value (NOT scaled to YYYYMMDD offset).
    // The unit (YEAR/MONTH/DAY) is resolved at the point of use in walkAExpr
    // by re-inspecting the AST node's typmods.
    if (typStr == "interval") {
        int intervalValue = 0;
        if (auto* lit = std::get_if<Literal>(&arg->node)) {
            if (auto* iv = std::get_if<int>(&lit->value))
                intervalValue = *iv;
            else if (auto* sv = std::get_if<std::string>(&lit->value))
                intervalValue = std::stoi(*sv);
        }
        return Expr::lit(intervalValue);
    }
    return arg; // For other casts, pass through
}

// Helper: proper date arithmetic in YYYYMMDD format for DAY intervals
static int computeDateArithDays(int yyyymmdd, int days, bool isAdd) {
    int dir = isAdd ? 1 : -1;
    int y = yyyymmdd / 10000;
    int m = (yyyymmdd / 100) % 100;
    int d = yyyymmdd % 100;

    auto isLeap = [](int yr) { return (yr % 4 == 0 && yr % 100 != 0) || yr % 400 == 0; };
    auto daysInMonth = [&](int yr, int mo) -> int {
        static const int dim[] = {0,31,28,31,30,31,30,31,31,30,31,30,31};
        if (mo == 2 && isLeap(yr)) return 29;
        return dim[mo];
    };

    d += dir * days;
    while (d > daysInMonth(y, m)) {
        d -= daysInMonth(y, m);
        m++;
        if (m > 12) { m = 1; y++; }
    }
    while (d < 1) {
        m--;
        if (m < 1) { m = 12; y--; }
        d += daysInMonth(y, m);
    }

    return y * 10000 + m * 100 + d;
}

// Interval unit enum
enum class IntervalUnit { UNKNOWN, YEAR, MONTH, DAY };

// Extract interval unit from a TypeCast AST node's typmods
static IntervalUnit extractIntervalUnit(const json& typeCastNode) {
    if (!typeCastNode.contains("typeName")) return IntervalUnit::UNKNOWN;
    auto& tn = typeCastNode["typeName"];
    bool isInterval = false;
    if (tn.contains("names")) {
        for (auto& n : tn["names"]) {
            if (n.contains("String") && n["String"]["sval"] == "interval")
                isInterval = true;
        }
    }
    if (!isInterval) return IntervalUnit::UNKNOWN;

    int typmods = 0;
    if (tn.contains("typmods")) {
        for (auto& tm : tn["typmods"]) {
            if (tm.contains("Integer"))
                typmods = tm["Integer"]["ival"].get<int>();
            else if (tm.contains("A_Const") && tm["A_Const"].contains("ival"))
                typmods = tm["A_Const"]["ival"]["ival"].get<int>();
        }
    }
    // PostgreSQL datetime.h: YEAR=2, MONTH=1, DAY=3
    // INTERVAL_MASK(X) = 1 << X → YEAR=4, MONTH=2, DAY=8
    if (typmods & 4) return IntervalUnit::YEAR;
    if (typmods & 2) return IntervalUnit::MONTH;
    if (typmods & 8) return IntervalUnit::DAY;
    return IntervalUnit::UNKNOWN;
}

// Compute DATE ± INTERVAL with proper unit handling (YEAR/MONTH/DAY)
static int computeDateArith(int yyyymmdd, int intervalVal, bool isAdd, IntervalUnit unit) {
    int dir = isAdd ? 1 : -1;
    int y = yyyymmdd / 10000;
    int m = (yyyymmdd / 100) % 100;
    int d = yyyymmdd % 100;

    switch (unit) {
        case IntervalUnit::YEAR:
            y += dir * intervalVal;
            return y * 10000 + m * 100 + d;
        case IntervalUnit::MONTH: {
            m += dir * intervalVal;
            while (m > 12) { m -= 12; y++; }
            while (m < 1)  { m += 12; y--; }
            return y * 10000 + m * 100 + d;
        }
        case IntervalUnit::DAY:
        default:
            return computeDateArithDays(yyyymmdd, intervalVal, isAdd);
    }
}

ExprPtr walkFuncCall(const json& node, const std::vector<std::string>& tables) {
    std::string funcName;
    if (node.contains("funcname")) {
        for (auto& n : node["funcname"]) {
            if (n.contains("String"))
                funcName = n["String"]["sval"].get<std::string>();
        }
    }
    std::transform(funcName.begin(), funcName.end(), funcName.begin(), ::tolower);

    FuncCall fc;
    fc.name = funcName;
    if (node.contains("args")) {
        for (auto& a : node["args"])
            fc.args.push_back(walkExpr(a, tables));
    }

    auto e = std::make_shared<Expr>();
    e->node = fc;
    return e;
}

ExprPtr walkAExpr(const json& node, const std::vector<std::string>& tables) {
    std::string kind = node.value("kind", "AEXPR_OP");
    std::string opName;
    if (node.contains("name")) {
        for (auto& n : node["name"]) {
            if (n.contains("String"))
                opName = n["String"]["sval"].get<std::string>();
        }
    }

    if (kind == "AEXPR_OP") {
        // Arithmetic operators in expression context
        ExprOp exOp;
        if      (opName == "+") exOp = ExprOp::ADD;
        else if (opName == "-") exOp = ExprOp::SUB;
        else if (opName == "*") exOp = ExprOp::MUL;
        else if (opName == "/") exOp = ExprOp::DIV;
        else {
            // Comparison operators — shouldn't be called from walkExpr, but handle gracefully
            // Return a dummy expression
            if (node.contains("lexpr"))
                return walkExpr(node["lexpr"], tables);
            return Expr::lit(0);
        }
        auto left = node.contains("lexpr") ? walkExpr(node["lexpr"], tables) : Expr::lit(0);
        auto right = node.contains("rexpr") ? walkExpr(node["rexpr"], tables) : Expr::lit(0);

        // Pre-compute date ± interval when both sides are literals
        if (exOp == ExprOp::ADD || exOp == ExprOp::SUB) {
            auto* litL = std::get_if<Literal>(&left->node);
            auto* litR = std::get_if<Literal>(&right->node);
            if (litL && litR) {
                auto* dateVal = std::get_if<int>(&litL->value);
                auto* intVal  = std::get_if<int>(&litR->value);
                if (dateVal && intVal && *dateVal > 19000101 && *dateVal < 21001231) {
                    bool isAdd = (exOp == ExprOp::ADD);
                    // Determine interval unit from the original AST rexpr
                    IntervalUnit unit = IntervalUnit::DAY; // default
                    if (node.contains("rexpr") && node["rexpr"].contains("TypeCast"))
                        unit = extractIntervalUnit(node["rexpr"]["TypeCast"]);
                    int result = computeDateArith(*dateVal, *intVal, isAdd, unit);
                    return Expr::lit(result);
                }
            }
        }

        return Expr::binary(exOp, left, right);
    }
    // Non-arithmetic A_Expr types (LIKE, BETWEEN, IN) in expression context
    // Return lexpr as a pass-through — the planner doesn't inspect these deeply
    if (node.contains("lexpr"))
        return walkExpr(node["lexpr"], tables);
    return Expr::lit(0);
}

ExprPtr walkExpr(const json& node, const std::vector<std::string>& tables) {
    if (node.contains("ColumnRef"))
        return walkColumnRef(node["ColumnRef"], tables);
    if (node.contains("A_Const"))
        return walkConst(node["A_Const"]);
    if (node.contains("TypeCast"))
        return walkTypeCast(node["TypeCast"], tables);
    if (node.contains("FuncCall"))
        return walkFuncCall(node["FuncCall"], tables);
    if (node.contains("A_Expr"))
        return walkAExpr(node["A_Expr"], tables);
    if (node.contains("SubLink")) {
        // Subquery expression — return placeholder
        return Expr::lit(0);
    }
    if (node.contains("BoolExpr")) {
        // Boolean expression in expression context (e.g., in CASE WHEN)
        // Return a placeholder literal
        return Expr::lit(0);
    }
    if (node.contains("CaseExpr")) {
        auto& ce = node["CaseExpr"];
        CaseWhen cw;
        if (ce.contains("args")) {
            for (auto& when : ce["args"]) {
                if (when.contains("CaseWhen")) {
                    auto& caseWhen = when["CaseWhen"];
                    CaseWhen::Branch br;
                    br.condition = walkExpr(caseWhen["expr"], tables);
                    br.result = walkExpr(caseWhen["result"], tables);
                    cw.branches.push_back(std::move(br));
                }
            }
        }
        if (ce.contains("defresult"))
            cw.elseResult = walkExpr(ce["defresult"], tables);
        auto e = std::make_shared<Expr>();
        e->node = std::move(cw);
        return e;
    }
    std::cerr << "WARN: unhandled expr node, returning lit(0): " << node.dump().substr(0, 200) << std::endl;
    return Expr::lit(0);
}

// ===================================================================
// PREDICATE WALKING
// ===================================================================

CmpOp parseCmpOp(const std::string& op) {
    if (op == "=")  return CmpOp::EQ;
    if (op == "<>") return CmpOp::NE;
    if (op == "!=") return CmpOp::NE;
    if (op == "<")  return CmpOp::LT;
    if (op == "<=") return CmpOp::LE;
    if (op == ">")  return CmpOp::GT;
    if (op == ">=") return CmpOp::GE;
    throw std::runtime_error("Unknown CmpOp: " + op);
}

PredPtr walkAExprPred(const json& node, const std::vector<std::string>& tables) {
    std::string kind = node.value("kind", "AEXPR_OP");
    std::string opName;
    if (node.contains("name")) {
        for (auto& n : node["name"])
            if (n.contains("String"))
                opName = n["String"]["sval"].get<std::string>();
    }

    if (kind == "AEXPR_OP") {
        auto left = walkExpr(node["lexpr"], tables);
        auto right = walkExpr(node["rexpr"], tables);
        return Predicate::cmp(parseCmpOp(opName), left, right);
    }
    if (kind == "AEXPR_BETWEEN" || kind == "AEXPR_NOT_BETWEEN") {
        auto expr = walkExpr(node["lexpr"], tables);
        auto& list = node["rexpr"]["List"]["items"];
        auto lo = walkExpr(list[0], tables);
        auto hi = walkExpr(list[1], tables);
        auto p = Predicate::between(expr, lo, hi);
        if (kind == "AEXPR_NOT_BETWEEN")
            return Predicate::logNot(p);
        return p;
    }
    if (kind == "AEXPR_IN") {
        auto expr = walkExpr(node["lexpr"], tables);
        std::vector<ExprPtr> vals;
        if (node["rexpr"].contains("List")) {
            for (auto& item : node["rexpr"]["List"]["items"])
                vals.push_back(walkExpr(item, tables));
        }
        auto p = Predicate::inList(expr, std::move(vals));
        if (opName == "<>")
            return Predicate::logNot(p);
        return p;
    }
    if (kind == "AEXPR_LIKE" || kind == "AEXPR_ILIKE") {
        auto expr = walkExpr(node["lexpr"], tables);
        auto patExpr = walkExpr(node["rexpr"], tables);
        std::string pat;
        if (auto* lit = std::get_if<Literal>(&patExpr->node))
            if (auto* sv = std::get_if<std::string>(&lit->value))
                pat = *sv;
        bool negated = (opName == "!~~" || opName == "!~~*");
        return Predicate::like(expr, pat, negated);
    }
    if (kind == "AEXPR_NOT_DISTINCT") {
        // Treat as equality
        auto left = walkExpr(node["lexpr"], tables);
        auto right = walkExpr(node["rexpr"], tables);
        return Predicate::cmp(CmpOp::EQ, left, right);
    }

    // Fallback: treat as comparison
    auto left = node.contains("lexpr") ? walkExpr(node["lexpr"], tables) : Expr::lit(0);
    auto right = node.contains("rexpr") ? walkExpr(node["rexpr"], tables) : Expr::lit(0);
    return Predicate::cmp(CmpOp::EQ, left, right);
}

PredPtr walkPredicate(const json& node, const std::vector<std::string>& tables) {
    if (node.contains("BoolExpr")) {
        auto& be = node["BoolExpr"];
        std::string boolop = be["boolop"].get<std::string>();
        if (boolop == "AND_EXPR") {
            std::vector<PredPtr> children;
            for (auto& arg : be["args"])
                children.push_back(walkPredicate(arg, tables));
            return Predicate::logAnd(std::move(children));
        }
        if (boolop == "OR_EXPR") {
            std::vector<PredPtr> children;
            for (auto& arg : be["args"])
                children.push_back(walkPredicate(arg, tables));
            return Predicate::logOr(std::move(children));
        }
        if (boolop == "NOT_EXPR") {
            return Predicate::logNot(walkPredicate(be["args"][0], tables));
        }
    }
    if (node.contains("A_Expr")) {
        return walkAExprPred(node["A_Expr"], tables);
    }
    if (node.contains("NullTest")) {
        auto& nt = node["NullTest"];
        auto expr = walkExpr(nt["arg"], tables);
        std::string kind = nt.value("nulltesttype", "IS_NULL");
        if (kind == "IS_NOT_NULL")
            return Predicate::cmp(CmpOp::NE, expr, Expr::lit(0)); // approximate
        return Predicate::cmp(CmpOp::EQ, expr, Expr::lit(0)); // approximate
    }
    if (node.contains("SubLink")) {
        // EXISTS / NOT EXISTS / IN subquery — extract as subquery reference
        auto& sl = node["SubLink"];
        std::string subType = sl.value("subLinkType", "EXISTS_SUBLINK");
        if (subType == "EXISTS_SUBLINK") {
            auto p = std::make_shared<Predicate>();
            p->node = ExistsPred{false, -1};
            return p;
        }
        if (subType == "ALL_SUBLINK" || subType == "ANY_SUBLINK") {
            // IN subquery
            if (sl.contains("testexpr")) {
                auto expr = walkExpr(sl["testexpr"], tables);
                return Predicate::inList(expr, {}); // placeholder
            }
        }
        auto p = std::make_shared<Predicate>();
        p->node = ExistsPred{false, -1};
        return p;
    }
    throw std::runtime_error("Unknown predicate node: " + node.dump().substr(0, 100));
}

// ===================================================================
// EXTRACT TABLES FROM FROM CLAUSE
// ===================================================================

void extractTables(const json& fromItem, std::vector<std::string>& tables,
                   std::vector<std::string>& aliases) {
    if (fromItem.contains("RangeVar")) {
        auto& rv = fromItem["RangeVar"];
        std::string name = rv["relname"].get<std::string>();
        tables.push_back(name);
        if (rv.contains("alias")) {
            auto& a = rv["alias"];
            if (a.contains("Alias"))
                aliases.push_back(a["Alias"]["aliasname"].get<std::string>());
            else if (a.contains("aliasname"))
                aliases.push_back(a["aliasname"].get<std::string>());
            else
                aliases.push_back(name);
        } else {
            aliases.push_back(name);
        }
    }
    if (fromItem.contains("JoinExpr")) {
        auto& je = fromItem["JoinExpr"];
        extractTables(je["larg"], tables, aliases);
        extractTables(je["rarg"], tables, aliases);
    }
    if (fromItem.contains("RangeSubselect")) {
        // Subquery in FROM — push a placeholder
        tables.push_back("__subquery__");
        auto& rs = fromItem["RangeSubselect"];
        if (rs.contains("alias")) {
            auto& a = rs["alias"];
            if (a.contains("Alias"))
                aliases.push_back(a["Alias"]["aliasname"].get<std::string>());
            else if (a.contains("aliasname"))
                aliases.push_back(a["aliasname"].get<std::string>());
            else
                aliases.push_back("__subquery__");
        } else {
            aliases.push_back("__subquery__");
        }
    }
}

// ===================================================================
// EXTRACT JOIN CONDITIONS FROM WHERE
// ===================================================================

// Check if a predicate is a join condition (column = column across different tables)
bool isJoinCondition(const PredPtr& pred, const std::vector<std::string>& tables,
                     JoinClause& jc) {
    auto* cmp = std::get_if<Comparison>(&pred->node);
    if (!cmp || cmp->op != CmpOp::EQ) return false;

    auto* leftCol = std::get_if<ColRef>(&cmp->left->node);
    auto* rightCol = std::get_if<ColRef>(&cmp->right->node);
    if (!leftCol || !rightCol) return false;
    if (leftCol->table == rightCol->table) return false;

    jc.leftTable = leftCol->table;
    jc.leftCol = leftCol->column;
    jc.rightTable = rightCol->table;
    jc.rightCol = rightCol->column;
    return true;
}

// Flatten AND predicates and separate join conditions from filters
void separatePredicates(const PredPtr& pred, const std::vector<std::string>& tables,
                        std::vector<JoinClause>& joins, std::vector<PredPtr>& filters) {
    if (auto* la = std::get_if<LogicalAnd>(&pred->node)) {
        for (auto& child : la->children)
            separatePredicates(child, tables, joins, filters);
        return;
    }
    JoinClause jc;
    if (isJoinCondition(pred, tables, jc))
        joins.push_back(jc);
    else
        filters.push_back(pred);
}

// ===================================================================
// EXTRACT TARGET LIST
// ===================================================================

AggFunc parseAggFunc(const std::string& name) {
    if (name == "sum")   return AggFunc::SUM;
    if (name == "count") return AggFunc::COUNT;
    if (name == "avg")   return AggFunc::AVG;
    if (name == "min")   return AggFunc::MIN;
    if (name == "max")   return AggFunc::MAX;
    throw std::runtime_error("Unknown aggregate: " + name);
}

SelectTarget extractTarget(const json& resTarget, const std::vector<std::string>& tables) {
    SelectTarget st;
    st.alias = resTarget.value("name", "");
    auto& val = resTarget["val"];
    if (val.contains("FuncCall")) {
        auto& fc = val["FuncCall"];
        std::string funcName;
        if (fc.contains("funcname"))
            for (auto& n : fc["funcname"])
                if (n.contains("String"))
                    funcName = n["String"]["sval"].get<std::string>();
        std::transform(funcName.begin(), funcName.end(), funcName.begin(), ::tolower);

        // Check if it's an aggregate function
        if (funcName == "sum" || funcName == "count" || funcName == "avg" ||
            funcName == "min" || funcName == "max") {
            st.isAgg = true;
            AggTarget at;
            at.func = parseAggFunc(funcName);
            at.alias = st.alias;
            if (fc.contains("agg_star") && fc["agg_star"].get<bool>()) {
                at.isStar = true;
            } else if (fc.contains("args") && !fc["args"].empty()) {
                at.innerExpr = walkExpr(fc["args"][0], tables);
            }
            if (fc.contains("agg_distinct") && fc["agg_distinct"].get<bool>()) {
                at.func = AggFunc::COUNT_DISTINCT;
            }
            st.agg = at;
            st.expr = walkExpr(val, tables); // Full FuncCall as expr
        } else {
            st.expr = walkExpr(val, tables);
        }
    } else {
        st.expr = walkExpr(val, tables);
    }
    return st;
}

// ===================================================================
// EXTRACT JOIN CONDITIONS FROM EXPLICIT JOIN ON
// ===================================================================

void extractJoinOns(const json& fromItem, const std::vector<std::string>& tables,
                    std::vector<JoinClause>& joins, std::vector<PredPtr>& filters) {
    if (fromItem.contains("JoinExpr")) {
        auto& je = fromItem["JoinExpr"];
        if (je.contains("quals")) {
            auto pred = walkPredicate(je["quals"], tables);
            separatePredicates(pred, tables, joins, filters);
        }
        extractJoinOns(je["larg"], tables, joins, filters);
        extractJoinOns(je["rarg"], tables, joins, filters);
    }
}

} // anonymous namespace

// ===================================================================
// PUBLIC: analyzeSQL
// ===================================================================

AnalyzedQuery analyzeSQL(const std::string& sql) {
    PgQueryParseResult result = pg_query_parse(sql.c_str());
    if (result.error) {
        std::string msg = result.error->message;
        pg_query_free_parse_result(result);
        throw std::runtime_error("SQL parse error: " + msg);
    }

    json ast;
    try {
        ast = json::parse(result.parse_tree);
    } catch (const std::exception& e) {
        pg_query_free_parse_result(result);
        throw std::runtime_error("JSON parse error: " + std::string(e.what()));
    }
    pg_query_free_parse_result(result);

    AnalyzedQuery aq;

    // Navigate to the SelectStmt
    auto& stmt = ast["stmts"][0]["stmt"];
    if (!stmt.contains("SelectStmt"))
        throw std::runtime_error("Expected SELECT statement");
    auto& sel = stmt["SelectStmt"];

    // 1. Extract tables from FROM clause
    if (sel.contains("fromClause")) {
        for (auto& item : sel["fromClause"])
            extractTables(item, aq.tables, aq.tableAliases);
        // Build alias -> real table name map
        g_aliasMap.clear();
        for (size_t i = 0; i < aq.tables.size(); ++i) {
            if (i < aq.tableAliases.size() && aq.tableAliases[i] != aq.tables[i])
                g_aliasMap[aq.tableAliases[i]] = aq.tables[i];
        }
        // Extract explicit JOIN ON conditions
        for (auto& item : sel["fromClause"])
            extractJoinOns(item, aq.tables, aq.joins, aq.filters);
    }

    // 2. Extract WHERE clause predicates
    if (sel.contains("whereClause")) {
        auto wherePred = walkPredicate(sel["whereClause"], aq.tables);
        separatePredicates(wherePred, aq.tables, aq.joins, aq.filters);
    }

    // 3. Extract SELECT targets
    if (sel.contains("targetList")) {
        for (auto& t : sel["targetList"]) {
            if (t.contains("ResTarget"))
                aq.targets.push_back(extractTarget(t["ResTarget"], aq.tables));
        }
    }

    if (sel.contains("groupClause")) {
        for (auto& g : sel["groupClause"])
            aq.groupBy.push_back(walkExpr(g, aq.tables));
    }

    // 5. Extract HAVING
    if (sel.contains("havingClause"))
        aq.having = walkPredicate(sel["havingClause"], aq.tables);

    // 6. Extract ORDER BY
    if (sel.contains("sortClause")) {
        for (auto& s : sel["sortClause"]) {
            if (s.contains("SortBy")) {
                auto& sb = s["SortBy"];
                OrderByItem obi;
                obi.expr = walkExpr(sb["node"], aq.tables);
                if (sb.contains("sortby_dir")) {
                    std::string dir = sb["sortby_dir"].get<std::string>();
                    obi.descending = (dir == "SORTBY_DESC");
                }
                aq.orderBy.push_back(obi);
            }
        }
    }

    // 7. Extract LIMIT
    if (sel.contains("limitCount")) {
        auto& lc = sel["limitCount"];
        if (lc.contains("A_Const") && lc["A_Const"].contains("ival"))
            aq.limit = lc["A_Const"]["ival"]["ival"].get<int>();
    }

    return aq;
}

} // namespace codegen
