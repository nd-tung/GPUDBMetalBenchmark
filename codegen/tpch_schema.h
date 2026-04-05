#pragma once
#include "query_plan.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

namespace codegen {

// ===================================================================
// TPC-H SCHEMA CATALOG
// ===================================================================
// Hard-coded TPC-H table definitions. Column indices match .tbl file layout.

struct ColumnDef {
    std::string name;
    int         index;
    DataType    type;
    int         fixedWidth = 0; // for CHAR_FIXED
};

struct TableDef {
    std::string name;
    std::vector<ColumnDef> columns;
    std::unordered_map<std::string, int> nameToIdx; // column name → index in columns[]

    const ColumnDef& col(const std::string& n) const {
        auto it = nameToIdx.find(n);
        if (it == nameToIdx.end()) throw std::runtime_error("Unknown column: " + name + "." + n);
        return columns[it->second];
    }
};

inline TableDef makeTable(const std::string& name, std::vector<ColumnDef> cols) {
    TableDef t;
    t.name = name;
    t.columns = std::move(cols);
    for (size_t i = 0; i < t.columns.size(); i++)
        t.nameToIdx[t.columns[i].name] = (int)i;
    return t;
}

// ===================================================================
// SCHEMA SINGLETON
// ===================================================================

struct TPCHSchema {
    std::unordered_map<std::string, TableDef> tables;

    const TableDef& table(const std::string& name) const {
        auto it = tables.find(name);
        if (it == tables.end()) throw std::runtime_error("Unknown table: " + name);
        return it->second;
    }

    ColumnBinding binding(const std::string& tableName, const std::string& colName) const {
        auto& t = table(tableName);
        auto& c = t.col(colName);
        return {tableName, colName, c.index, c.type, c.fixedWidth};
    }

    static const TPCHSchema& instance() {
        static TPCHSchema s = build();
        return s;
    }

private:
    static TPCHSchema build() {
        TPCHSchema s;

        // lineitem: 16 columns (0-15)
        s.tables["lineitem"] = makeTable("lineitem", {
            {"l_orderkey",      0,  DataType::INT},
            {"l_partkey",       1,  DataType::INT},
            {"l_suppkey",       2,  DataType::INT},
            {"l_linenumber",    3,  DataType::INT},
            {"l_quantity",      4,  DataType::FLOAT},
            {"l_extendedprice", 5,  DataType::FLOAT},
            {"l_discount",      6,  DataType::FLOAT},
            {"l_tax",           7,  DataType::FLOAT},
            {"l_returnflag",    8,  DataType::CHAR1},
            {"l_linestatus",    9,  DataType::CHAR1},
            {"l_shipdate",      10, DataType::DATE},
            {"l_commitdate",    11, DataType::DATE},
            {"l_receiptdate",   12, DataType::DATE},
            {"l_shipinstruct",  13, DataType::CHAR_FIXED, 25},
            {"l_shipmode",      14, DataType::CHAR_FIXED, 10},
            {"l_comment",       15, DataType::CHAR_FIXED, 44},
        });

        // orders: 9 columns (0-8)
        s.tables["orders"] = makeTable("orders", {
            {"o_orderkey",      0, DataType::INT},
            {"o_custkey",       1, DataType::INT},
            {"o_orderstatus",   2, DataType::CHAR1},
            {"o_totalprice",    3, DataType::FLOAT},
            {"o_orderdate",     4, DataType::DATE},
            {"o_orderpriority", 5, DataType::CHAR_FIXED, 15},
            {"o_clerk",         6, DataType::CHAR_FIXED, 15},
            {"o_shippriority",  7, DataType::INT},
            {"o_comment",       8, DataType::CHAR_FIXED, 79},
        });

        // customer: 8 columns (0-7)
        s.tables["customer"] = makeTable("customer", {
            {"c_custkey",    0, DataType::INT},
            {"c_name",       1, DataType::CHAR_FIXED, 25},
            {"c_address",    2, DataType::CHAR_FIXED, 40},
            {"c_nationkey",  3, DataType::INT},
            {"c_phone",      4, DataType::CHAR_FIXED, 15},
            {"c_acctbal",    5, DataType::FLOAT},
            {"c_mktsegment", 6, DataType::CHAR_FIXED, 10},
            {"c_comment",    7, DataType::CHAR_FIXED, 117},
        });

        // supplier: 7 columns (0-6)
        s.tables["supplier"] = makeTable("supplier", {
            {"s_suppkey",   0, DataType::INT},
            {"s_name",      1, DataType::CHAR_FIXED, 25},
            {"s_address",   2, DataType::CHAR_FIXED, 40},
            {"s_nationkey", 3, DataType::INT},
            {"s_phone",     4, DataType::CHAR_FIXED, 15},
            {"s_acctbal",   5, DataType::FLOAT},
            {"s_comment",   6, DataType::CHAR_FIXED, 101},
        });

        // part: 9 columns (0-8)
        s.tables["part"] = makeTable("part", {
            {"p_partkey",    0, DataType::INT},
            {"p_name",       1, DataType::CHAR_FIXED, 55},
            {"p_mfgr",       2, DataType::CHAR_FIXED, 25},
            {"p_brand",      3, DataType::CHAR_FIXED, 10},
            {"p_type",       4, DataType::CHAR_FIXED, 25},
            {"p_size",       5, DataType::INT},
            {"p_container",  6, DataType::CHAR_FIXED, 10},
            {"p_retailprice",7, DataType::FLOAT},
            {"p_comment",    8, DataType::CHAR_FIXED, 23},
        });

        // partsupp: 5 columns (0-4)
        s.tables["partsupp"] = makeTable("partsupp", {
            {"ps_partkey",    0, DataType::INT},
            {"ps_suppkey",    1, DataType::INT},
            {"ps_availqty",   2, DataType::INT},
            {"ps_supplycost", 3, DataType::FLOAT},
            {"ps_comment",    4, DataType::CHAR_FIXED, 199},
        });

        // nation: 4 columns (0-3)
        s.tables["nation"] = makeTable("nation", {
            {"n_nationkey",  0, DataType::INT},
            {"n_name",       1, DataType::CHAR_FIXED, 25},
            {"n_regionkey",  2, DataType::INT},
            {"n_comment",    3, DataType::CHAR_FIXED, 152},
        });

        // region: 3 columns (0-2)
        s.tables["region"] = makeTable("region", {
            {"r_regionkey", 0, DataType::INT},
            {"r_name",      1, DataType::CHAR_FIXED, 25},
            {"r_comment",   2, DataType::CHAR_FIXED, 152},
        });

        return s;
    }
};

// ===================================================================
// FILE PATH RESOLUTION
// ===================================================================

inline std::string tblPath(const std::string& dataDir, const std::string& tableName) {
    return dataDir + tableName + ".tbl";
}

} // namespace codegen
