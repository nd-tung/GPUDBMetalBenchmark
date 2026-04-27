// tbl_to_colbin — convert TPC-H .tbl files to the shared .colbin format.
// Output is byte-compatible with GPUDBMetalCodeGen/tools/tbl_to_colbin.
//
// Usage:   ./tbl_to_colbin <data_dir>     e.g.  ./tbl_to_colbin data/SF-1
//   GPUDB_FORCE_REBUILD=1  force rewrite even if .colbin is up to date.
//
// The schema is hardcoded (same as GPUDBMetalCodeGen/codegen/planning/tpch_schema.h)
// so this tool does NOT depend on any codegen library.

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "../src/infra.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>

// ----- Hardcoded TPC-H schema (matches codegen/planning/tpch_schema.h) -----
struct TableSchema {
    const char* name;
    std::vector<ColSpec> specs;
};

static const std::vector<TableSchema>& schemas() {
    static const std::vector<TableSchema> s = {
        { "lineitem", {
            {0,  ColType::INT},        {1,  ColType::INT},       {2,  ColType::INT},
            {3,  ColType::INT},        {4,  ColType::FLOAT},     {5,  ColType::FLOAT},
            {6,  ColType::FLOAT},      {7,  ColType::FLOAT},     {8,  ColType::CHAR1},
            {9,  ColType::CHAR1},      {10, ColType::DATE},      {11, ColType::DATE},
            {12, ColType::DATE},       {13, ColType::CHAR_FIXED, 25},
            {14, ColType::CHAR_FIXED, 10}, {15, ColType::CHAR_FIXED, 44},
        }},
        { "orders", {
            {0, ColType::INT},         {1, ColType::INT},        {2, ColType::CHAR1},
            {3, ColType::FLOAT},       {4, ColType::DATE},       {5, ColType::CHAR_FIXED, 15},
            {6, ColType::CHAR_FIXED, 15}, {7, ColType::INT},     {8, ColType::CHAR_FIXED, 79},
        }},
        { "customer", {
            {0, ColType::INT},         {1, ColType::CHAR_FIXED, 25},
            {2, ColType::CHAR_FIXED, 40}, {3, ColType::INT},
            {4, ColType::CHAR_FIXED, 15},  {5, ColType::FLOAT},
            {6, ColType::CHAR_FIXED, 10}, {7, ColType::CHAR_FIXED, 117},
        }},
        { "supplier", {
            {0, ColType::INT},         {1, ColType::CHAR_FIXED, 25},
            {2, ColType::CHAR_FIXED, 40}, {3, ColType::INT},
            {4, ColType::CHAR_FIXED, 15}, {5, ColType::FLOAT},
            {6, ColType::CHAR_FIXED, 101},
        }},
        { "part", {
            {0, ColType::INT},         {1, ColType::CHAR_FIXED, 55},
            {2, ColType::CHAR_FIXED, 25}, {3, ColType::CHAR_FIXED, 10},
            {4, ColType::CHAR_FIXED, 25}, {5, ColType::INT},
            {6, ColType::CHAR_FIXED, 10}, {7, ColType::FLOAT},
            {8, ColType::CHAR_FIXED, 23},
        }},
        { "partsupp", {
            {0, ColType::INT},         {1, ColType::INT},
            {2, ColType::INT},         {3, ColType::FLOAT},
            {4, ColType::CHAR_FIXED, 199},
        }},
        { "nation", {
            {0, ColType::INT},         {1, ColType::CHAR_FIXED, 25},
            {2, ColType::INT},         {3, ColType::CHAR_FIXED, 152},
        }},
        { "region", {
            {0, ColType::INT},         {1, ColType::CHAR_FIXED, 25},
            {2, ColType::CHAR_FIXED, 152},
        }},
    };
    return s;
}

// Globals required by infra.cpp translation unit (this tool links without it).
std::string g_dataset_path = "";
bool        g_sf100_mode   = false;
std::string g_current_query = "";
std::string g_current_sf    = "";
DetailedTiming g_detailed_timing;

static bool binaryUpToDate(const std::string& tblPath) {
    size_t tblSz = 0, binSz = 0;
    int64_t tblMtime = 0, binMtime = 0;
    if (!colbin::statFile(tblPath, tblSz, tblMtime)) return false;
    if (!colbin::statFile(colbin::binaryPath(tblPath), binSz, binMtime)) return false;
    // Cheap validity check: open the header and compare source metadata.
    MappedFile mf;
    if (!mf.open(colbin::binaryPath(tblPath))) return false;
    if (mf.size < sizeof(colbin::FileHeader)) return false;
    colbin::FileHeader hdr;
    memcpy(&hdr, mf.data, sizeof(hdr));
    if (memcmp(hdr.magic, colbin::MAGIC, 8) != 0) return false;
    if (hdr.version != colbin::VERSION)           return false;
    if (hdr.source_size != tblSz)                 return false;
    if (hdr.source_mtime_ns != tblMtime)          return false;
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <data_dir>\n", argv[0]);
        fprintf(stderr, "   e.g. %s data/SF-1\n", argv[0]);
        return 2;
    }
    std::string dir = argv[1];
    if (!dir.empty() && dir.back() != '/') dir += '/';

    const bool force = []() {
        const char* e = ::getenv("GPUDB_FORCE_REBUILD");
        return e && e[0] == '1';
    }();

    // We call the text-only ingester directly so the tool never recurses on
    // its own output. (loadColumnsMultiAuto would short-circuit on a stale
    // .colbin and never hit the .tbl parser.)
    ::setenv("GPUDB_NO_BINARY", "1", 1);

    int totalOk = 0, totalSkip = 0, totalMiss = 0, totalErr = 0;
    double totalMs = 0.0;

    for (const auto& ts : schemas()) {
        std::string tbl = dir + ts.name + ".tbl";
        // skip missing tables silently (e.g. SF-N dirs generated with a subset)
        struct stat st{};
        if (::stat(tbl.c_str(), &st) != 0) { totalMiss++; continue; }

        if (!force && binaryUpToDate(tbl)) {
            printf("[skip] %-9s  (up to date)\n", ts.name);
            totalSkip++;
            continue;
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        LoadedColumns parsed = loadColumnsMultiText(tbl, ts.specs);
        auto t1 = std::chrono::high_resolution_clock::now();
        double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (!colbin::writeColbin(tbl, ts.specs, parsed)) {
            fprintf(stderr, "[fail] %s  (writeColbin returned false)\n", ts.name);
            totalErr++;
            continue;
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        double writeMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
        totalMs += parseMs + writeMs;

        // find nRows for the log line
        size_t nRows = 0;
        const auto& s0 = ts.specs.front();
        switch (s0.type) {
            case ColType::INT:
            case ColType::DATE:       nRows = parsed.ints(s0.columnIndex).size(); break;
            case ColType::FLOAT:      nRows = parsed.floats(s0.columnIndex).size(); break;
            case ColType::CHAR1:      nRows = parsed.chars(s0.columnIndex).size(); break;
            case ColType::CHAR_FIXED: nRows = s0.fixedWidth > 0
                                              ? parsed.chars(s0.columnIndex).size() / (size_t)s0.fixedWidth
                                              : parsed.chars(s0.columnIndex).size();
                                      break;
        }
        printf("[ok]   %-9s  %10zu rows   parse=%7.1f ms   write=%6.1f ms\n",
               ts.name, nRows, parseMs, writeMs);
        totalOk++;
    }

    printf("---\nconverted=%d  skipped=%d  missing=%d  failed=%d  totalTime=%.1f ms\n",
           totalOk, totalSkip, totalMiss, totalErr, totalMs);
    return totalErr == 0 ? 0 : 1;
}
