// Codegen standalone entry point
// Usage: GPUDBCodegen [sf1|sf10|sf100] q<N>
//
// This binary performs SQL → AnalyzedQuery → MetalQueryPlan → operators → Metal → GPU
// for all 22 TPC-H queries via runtime code generation.

#include "../src/infra.h"
#include "query_analyzer.h"
#include "runtime_compiler.h"
#include "metal_plan_builder.h"
#include "metal_generic_executor.h"
#include "tpch_schema.h"
#include <fstream>
#include <sstream>
#include <set>
#include <map>
#include <unordered_map>

// ===================================================================
// Shared post-processing data structs
// ===================================================================

struct Q20PostData {
    std::vector<int> ps_partkey, ps_suppkey, ps_availqty;
    std::vector<uint32_t> htKeys;
    std::vector<int> htPsIdx;
    uint32_t htMask = 0, htSlots = 0;
    std::vector<int> s_suppkey, s_nationkey;
    std::vector<char> s_name, s_address;
    int canada_nk = -1;
    MTL::Buffer* htValsBuf = nullptr;
};
static Q20PostData g_q20Post;

struct Q2PostData {
    std::vector<int> ps_partkey, ps_suppkey;
    std::vector<float> ps_supplycost;
    std::vector<int> s_suppkey, s_nationkey;
    std::vector<float> s_acctbal;
    std::vector<char> s_name, s_address, s_phone, s_comment;
    std::vector<int> p_partkey;
    std::vector<char> p_mfgr;
    std::vector<std::string> nationNames;
    std::vector<uint32_t> eurSuppBitmap;
    int maxPartkey = 0;
    MTL::Buffer* minCostBuf = nullptr;
};
static Q2PostData g_q2Post;

struct Q16PostData {
    struct GroupKey { std::string brand; std::string type; int size; };
    std::vector<GroupKey> groups;
    uint32_t bvInts = 0;
    uint32_t numGroups = 0;
    MTL::Buffer* groupBitmapsBuf = nullptr;
};
static Q16PostData g_q16Post;

struct Q21PostData {
    std::vector<int> s_suppkey;
    std::vector<char> s_name;
    int maxSuppkey = 0;
};
static Q21PostData g_q21Post;

struct Q18PostData {
    std::vector<int> o_custkey, o_orderdate;
    std::vector<float> o_totalprice;
    // Direct-mapped: okLookup[orderkey] = row index in orders table (or -1)
    std::vector<int> okLookup;
};
static Q18PostData g_q18Post;

// ===================================================================
// Run a query through the codegen pipeline
// ===================================================================

static void runCodegenQuery(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                            const std::string& sql, const std::string& queryName) {
    printf("\n=== Codegen: %s ===\n", queryName.c_str());
    try {
        // 1. Analyze SQL
        codegen::AnalyzedQuery analyzed;
        try {
            analyzed = codegen::analyzeSQL(sql);
        } catch (...) {
            // Name-based builders don't need analyzed query
        }

        // 2. Build operator-based plan
        auto maybePlan = codegen::buildMetalPlan(analyzed, queryName);
        if (!maybePlan) {
            std::cerr << "Codegen: query pattern not yet supported for " << queryName << std::endl;
            return;
        }
        auto& plan = *maybePlan;
        plan.name = queryName;

        // 3. Generate Metal source via producer-consumer operators
        auto cg = codegen::generateFromPlan(plan);
        std::string metalSource = cg.print();

        printf("Generated Metal source (%zu bytes, %d phase(s))\n",
               metalSource.size(), cg.phaseCount());

        // Debug: dump generated source to file
        {
            std::ofstream dbg("codegen_debug_" + queryName + ".metal");
            dbg << metalSource;
            printf("  (written to codegen_debug_%s.metal)\n", queryName.c_str());
        }

        // 4. Compile
        codegen::RuntimeCompiler compiler(device);
        auto* library = compiler.compile(metalSource);
        if (!library) {
            std::cerr << "Codegen: Metal compilation failed" << std::endl;
            return;
        }

        // Build CompiledQuery with PSOs for each phase
        codegen::RuntimeCompiler::CompiledQuery compiled;
        compiled.library = library;
        for (const auto& phase : cg.getPhases()) {
            auto* pso = compiler.getPipeline(library, phase.name);
            if (!pso) {
                std::cerr << "Codegen: PSO creation failed for " << phase.name << std::endl;
                return;
            }
            compiled.pipelines.push_back(pso);
            compiled.kernelNames.push_back(phase.name);
        }

        // 5. Load data — determine which columns to load from bindings
        auto parseStart = std::chrono::high_resolution_clock::now();

        codegen::MetalGenericExecutor executor(device, cmdQueue);
        const auto& schema = codegen::TPCHSchema::instance();

        // Collect columns needed per table from all phases
        std::map<std::string, std::set<std::string>> tableCols;
        for (const auto& phase : cg.getPhases()) {
            for (const auto& b : phase.bindings) {
                if (b.kind == codegen::MetalParamKind::TableData && !b.tableName.empty()) {
                    tableCols[b.tableName].insert(b.name);
                }
            }
        }

        // Load each table's columns and register with executor
        std::vector<std::pair<std::string, LoadedColumns>> loadedTables;
        for (auto& [tableName, colNames] : tableCols) {
            const auto& tdef = schema.table(tableName);
            std::vector<ColSpec> specs;
            for (const auto& colName : colNames) {
                auto& cdef = tdef.col(colName);
                ColType ct = ColType::INT;
                switch (cdef.type) {
                    case codegen::DataType::INT:    ct = ColType::INT; break;
                    case codegen::DataType::FLOAT:  ct = ColType::FLOAT; break;
                    case codegen::DataType::DATE:   ct = ColType::DATE; break;
                    case codegen::DataType::CHAR1:  ct = ColType::CHAR1; break;
                    case codegen::DataType::CHAR_FIXED: ct = ColType::CHAR_FIXED; break;
                }
                specs.emplace_back(cdef.index, ct, cdef.fixedWidth);
            }

            auto cols = loadColumnsMulti(g_dataset_path + tableName + ".tbl", specs);

            size_t rowCount = 0;
            for (const auto& colName : colNames) {
                auto& cdef = tdef.col(colName);
                switch (cdef.type) {
                    case codegen::DataType::INT:
                    case codegen::DataType::DATE: {
                        auto& data = cols.ints(cdef.index);
                        rowCount = data.size();
                        executor.registerColumn(colName, data.data(), data.size(), sizeof(int));
                        break;
                    }
                    case codegen::DataType::FLOAT: {
                        auto& data = cols.floats(cdef.index);
                        rowCount = data.size();
                        executor.registerColumn(colName, data.data(), data.size(), sizeof(float));
                        break;
                    }
                    case codegen::DataType::CHAR1: {
                        auto& data = cols.chars(cdef.index);
                        rowCount = data.size();
                        executor.registerColumn(colName, data.data(), data.size(), sizeof(char));
                        break;
                    }
                    case codegen::DataType::CHAR_FIXED: {
                        auto& data = cols.chars(cdef.index);
                        int fw = cdef.fixedWidth > 0 ? cdef.fixedWidth : 1;
                        rowCount = data.size() / fw;
                        executor.registerColumn(colName, data.data(), rowCount, fw);
                        break;
                    }
                }
            }

            executor.registerTableRowCount(tableName, rowCount);
            loadedTables.emplace_back(tableName, std::move(cols));
        }

        // ---------------------------------------------------------------
        // Compute max key values for dynamic buffer sizing
        // ---------------------------------------------------------------
        {
            int maxCk = 0, maxSk = 0, maxOk = 0, maxPk = 0;
            for (auto& [tblName, cols] : loadedTables) {
                const auto& tdef = schema.table(tblName);
                for (const auto& colName : tableCols[tblName]) {
                    auto& cdef = tdef.col(colName);
                    if (cdef.type != codegen::DataType::INT && cdef.type != codegen::DataType::DATE)
                        continue;
                    auto& data = cols.ints(cdef.index);
                    if (colName == "c_custkey" || colName == "o_custkey")
                        for (int k : data) maxCk = std::max(maxCk, k);
                    else if (colName == "s_suppkey" || colName == "l_suppkey" || colName == "ps_suppkey")
                        for (int k : data) maxSk = std::max(maxSk, k);
                    else if (colName == "o_orderkey" || colName == "l_orderkey")
                        for (int k : data) maxOk = std::max(maxOk, k);
                    else if (colName == "p_partkey" || colName == "l_partkey" || colName == "ps_partkey")
                        for (int k : data) maxPk = std::max(maxPk, k);
                }
            }
            executor.registerSymbol("maxCustkey", maxCk + 1);
            executor.registerSymbol("maxSuppkey", maxSk + 1);
            executor.registerSymbol("maxOrderkey", maxOk + 1);
            executor.registerSymbol("maxPartkey", maxPk + 1);
        }

        // ---------------------------------------------------------------
        // Per-query pre-processing: resolve small lookup tables/scalars
        // ---------------------------------------------------------------

        // Q7: resolve nation keys
        if (plan.name == "Q7") {
            auto nat = loadNation(g_dataset_path);
            int france_nk = findNationKey(nat, "FRANCE");
            int germany_nk = findNationKey(nat, "GERMANY");
            if (france_nk == -1 || germany_nk == -1) {
                std::cerr << "Error: FRANCE/GERMANY not found in nation table" << std::endl;
                return;
            }
            executor.registerScalarInt("france_nk", france_nk);
            executor.registerScalarInt("germany_nk", germany_nk);
        }

        // Q5: resolve ASIA regionkey
        if (plan.name == "Q5") {
            auto reg = loadRegion(g_dataset_path);
            int asia_rk = findRegionKey(reg.regionkey, reg.name.data(),
                                        RegionData::NAME_WIDTH, "ASIA");
            if (asia_rk == -1) {
                std::cerr << "Error: ASIA region not found" << std::endl;
                return;
            }
            executor.registerScalarInt("asia_rk", asia_rk);
        }

        // Q8: resolve AMERICA regionkey and BRAZIL nationkey
        if (plan.name == "Q8") {
            auto reg = loadRegion(g_dataset_path);
            auto nat = loadNation(g_dataset_path);
            int america_rk = findRegionKey(reg.regionkey, reg.name.data(),
                                            RegionData::NAME_WIDTH, "AMERICA");
            int brazil_nk = findNationKey(nat, "BRAZIL");
            if (america_rk == -1 || brazil_nk == -1) {
                std::cerr << "Error: AMERICA/BRAZIL not found" << std::endl;
                return;
            }
            executor.registerScalarInt("america_rk", america_rk);
            executor.registerScalarInt("brazil_nk", brazil_nk);
        }

        // Q22: compute avg balance for valid-prefix customers
        if (plan.name == "Q22") {
            for (auto& [tblName, cols] : loadedTables) {
                if (tblName == "customer") {
                    const auto& tdef = codegen::TPCHSchema::instance().table("customer");
                    auto& phoneCols = cols.chars(tdef.col("c_phone").index);
                    auto& balCols = cols.floats(tdef.col("c_acctbal").index);
                    size_t custCount = balCols.size();
                    double sumBal = 0.0;
                    int countBal = 0;
                    for (size_t j = 0; j < custCount; j++) {
                        int prefix = (phoneCols[j * 15] - '0') * 10 + (phoneCols[j * 15 + 1] - '0');
                        if (balCols[j] > 0.0f &&
                            (prefix == 13 || prefix == 17 || prefix == 18 ||
                             prefix == 23 || prefix == 29 || prefix == 30 || prefix == 31)) {
                            sumBal += balCols[j];
                            countBal++;
                        }
                    }
                    float avgBal = (countBal > 0) ? (float)(sumBal / countBal) : 0.0f;
                    executor.registerScalarFloat("avg_bal", avgBal);
                    break;
                }
            }
        }

        // Q11: resolve GERMANY nationkey
        if (plan.name == "Q11") {
            auto nat = loadNation(g_dataset_path);
            int germany_nk = findNationKey(nat, "GERMANY");
            if (germany_nk == -1) {
                std::cerr << "Error: GERMANY not found in nation table" << std::endl;
                return;
            }
            executor.registerScalarInt("germany_nk", germany_nk);
        }

        // Q17: build per-partkey threshold from loaded data
        if (plan.name == "Q17") {
            auto pCols = loadColumnsMulti(g_dataset_path + "part.tbl",
                {{0, ColType::INT}, {3, ColType::CHAR_FIXED, 10}, {6, ColType::CHAR_FIXED, 10}});
            auto& p_partkey = pCols.ints(0);
            auto& p_brand = pCols.chars(3);
            auto& p_container = pCols.chars(6);
            size_t partRows = p_partkey.size();

            std::vector<int>* pl_partkey = nullptr;
            std::vector<float>* pl_quantity = nullptr;
            size_t liRows = 0;
            const auto& tpchSchema = codegen::TPCHSchema::instance();
            for (auto& [tblName, cols] : loadedTables) {
                if (tblName == "lineitem") {
                    pl_partkey = &cols.ints(tpchSchema.table("lineitem").col("l_partkey").index);
                    pl_quantity = &cols.floats(tpchSchema.table("lineitem").col("l_quantity").index);
                    liRows = pl_partkey->size();
                }
            }
            if (pl_partkey) {
                int maxPk = 0;
                for (int pk : p_partkey) maxPk = std::max(maxPk, pk);
                size_t mapSize = (size_t)(maxPk + 1);

                std::vector<uint32_t> bitmap((mapSize + 31) / 32, 0);
                for (size_t i = 0; i < partRows; i++) {
                    const char* brand = p_brand.data() + i * 10;
                    const char* cont = p_container.data() + i * 10;
                    bool isBrand23 = (brand[0]=='B' && brand[5]=='#' && brand[6]=='2' && brand[7]=='3');
                    bool isMedBox = (cont[0]=='M' && cont[1]=='E' && cont[2]=='D' && cont[3]==' ' &&
                                     cont[4]=='B' && cont[5]=='O' && cont[6]=='X');
                    if (isBrand23 && isMedBox) {
                        int pk = p_partkey[i];
                        bitmap[pk / 32] |= (1u << (pk % 32));
                    }
                }

                std::vector<double> sumQty(mapSize, 0.0);
                std::vector<int> cntQty(mapSize, 0);
                for (size_t i = 0; i < liRows; i++) {
                    int pk = (*pl_partkey)[i];
                    if (pk >= 0 && (size_t)pk < mapSize && (bitmap[pk / 32] >> (pk % 32)) & 1) {
                        sumQty[pk] += (*pl_quantity)[i];
                        cntQty[pk]++;
                    }
                }

                std::vector<float> threshold(mapSize, 0.0f);
                for (size_t pk = 0; pk < mapSize; pk++) {
                    if (cntQty[pk] > 0) {
                        threshold[pk] = (float)(0.2 * sumQty[pk] / cntQty[pk]);
                    }
                }

                auto* threshBuf = device->newBuffer(threshold.data(), mapSize * sizeof(float),
                                                     MTL::ResourceStorageModeShared);
                executor.registerAllocatedBuffer("d_q17_threshold", threshBuf);
            }
        }

        // Q9: build green-parts bitmap, lookup arrays, and partsupp HT
        if (plan.name == "Q9") {
            auto pCols = loadColumnsMulti(g_dataset_path + "part.tbl",
                {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 55}});
            auto& p_partkey = pCols.ints(0);
            auto& p_name = pCols.chars(1);

            auto sCols = loadColumnsMulti(g_dataset_path + "supplier.tbl",
                {{0, ColType::INT}, {3, ColType::INT}});
            auto& s_suppkey = sCols.ints(0);
            auto& s_nationkey = sCols.ints(3);

            auto oCols = loadColumnsMulti(g_dataset_path + "orders.tbl",
                {{0, ColType::INT}, {4, ColType::DATE}});
            auto& o_orderkey = oCols.ints(0);
            auto& o_orderdate = oCols.ints(4);

            auto psCols = loadColumnsMulti(g_dataset_path + "partsupp.tbl",
                {{0, ColType::INT}, {1, ColType::INT}, {3, ColType::FLOAT}});
            auto& ps_partkey = psCols.ints(0);
            auto& ps_suppkey = psCols.ints(1);
            auto& ps_supplycost = psCols.floats(3);

            auto nCols = loadColumnsMulti(g_dataset_path + "nation.tbl",
                {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}});
            auto& n_nationkey = nCols.ints(0);
            auto& n_name = nCols.chars(1);

            std::vector<std::string> nationNames(25);
            for (size_t i = 0; i < n_nationkey.size(); i++) {
                std::string nm;
                for (int c = 0; c < 25; c++) {
                    char ch = n_name[i * 25 + c];
                    if (ch == ' ' || ch == '\0') break;
                    nm += ch;
                }
                nationNames[n_nationkey[i]] = nm;
            }

            int maxPk = 0;
            for (int pk : p_partkey) maxPk = std::max(maxPk, pk);
            size_t bmpInts = ((size_t)maxPk + 32) / 32;
            std::vector<uint32_t> partBitmap(bmpInts, 0);
            for (size_t i = 0; i < p_partkey.size(); i++) {
                bool found = false;
                for (int c = 0; c <= 50; c++) {
                    if (p_name[i * 55 + c] == 'g' && p_name[i * 55 + c + 1] == 'r' &&
                        p_name[i * 55 + c + 2] == 'e' && p_name[i * 55 + c + 3] == 'e' &&
                        p_name[i * 55 + c + 4] == 'n') { found = true; break; }
                }
                if (found) {
                    int pk = p_partkey[i];
                    partBitmap[pk / 32] |= (1u << (pk % 32));
                }
            }

            int maxSk = 0;
            for (int sk : s_suppkey) maxSk = std::max(maxSk, sk);
            std::vector<int> sNatArray((size_t)maxSk + 1, -1);
            for (size_t i = 0; i < s_suppkey.size(); i++)
                sNatArray[s_suppkey[i]] = s_nationkey[i];

            int maxOk = 0;
            for (int ok : o_orderkey) maxOk = std::max(maxOk, ok);
            std::vector<int> oYearArray((size_t)maxOk + 1, 0);
            for (size_t i = 0; i < o_orderkey.size(); i++)
                oYearArray[o_orderkey[i]] = o_orderdate[i] / 10000;

            size_t psEntries = 0;
            for (size_t i = 0; i < ps_partkey.size(); i++) {
                int pk = ps_partkey[i];
                if (pk >= 0 && (size_t)pk / 32 < bmpInts && (partBitmap[pk / 32] >> (pk % 32)) & 1)
                    psEntries++;
            }
            uint32_t htSlots = 1;
            while (htSlots < psEntries * 2) htSlots <<= 1;
            uint32_t htMask = htSlots - 1;
            uint32_t suppMul = (uint32_t)(maxSk + 1);
            std::vector<uint32_t> htKeys(htSlots, 0xFFFFFFFFu);
            std::vector<float> htVals(htSlots, 0.0f);
            for (size_t i = 0; i < ps_partkey.size(); i++) {
                int pk = ps_partkey[i];
                if (pk < 0 || (size_t)pk / 32 >= bmpInts || !((partBitmap[pk / 32] >> (pk % 32)) & 1))
                    continue;
                uint32_t key = (uint32_t)pk * suppMul + (uint32_t)ps_suppkey[i];
                uint32_t h = (key * 2654435769u) & htMask;
                for (uint32_t s = 0; s <= htMask; s++) {
                    uint32_t slot = (h + s) & htMask;
                    if (htKeys[slot] == 0xFFFFFFFFu) {
                        htKeys[slot] = key;
                        htVals[slot] = ps_supplycost[i];
                        break;
                    }
                }
            }

            auto* bmpBuf = device->newBuffer(partBitmap.data(), bmpInts * sizeof(uint32_t),
                                              MTL::ResourceStorageModeShared);
            auto* natBuf = device->newBuffer(sNatArray.data(), sNatArray.size() * sizeof(int),
                                              MTL::ResourceStorageModeShared);
            auto* yrBuf = device->newBuffer(oYearArray.data(), oYearArray.size() * sizeof(int),
                                             MTL::ResourceStorageModeShared);
            auto* htKeysBuf = device->newBuffer(htKeys.data(), htSlots * sizeof(uint32_t),
                                                 MTL::ResourceStorageModeShared);
            auto* htValsBuf = device->newBuffer(htVals.data(), htSlots * sizeof(float),
                                                 MTL::ResourceStorageModeShared);

            executor.registerAllocatedBuffer("d_q9_part_bitmap", bmpBuf);
            executor.registerAllocatedBuffer("d_q9_s_nationkey", natBuf);
            executor.registerAllocatedBuffer("d_q9_o_year", yrBuf);
            executor.registerAllocatedBuffer("d_ps_ht_keys", htKeysBuf);
            executor.registerAllocatedBuffer("d_ps_ht_vals", htValsBuf);
            executor.registerScalarInt("d_ps_ht_mask", (int)htMask);
            executor.registerScalarInt("supp_mul", (int)suppMul);

            static std::vector<std::string> q9NationNames;
            q9NationNames = std::move(nationNames);
        }

        // Q20: build forest% bitmap, partsupp HT, CANADA filter
        if (plan.name == "Q20") {
            auto pCols = loadColumnsMulti(g_dataset_path + "part.tbl",
                {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 55}});
            auto& p_partkey = pCols.ints(0);
            auto& p_name = pCols.chars(1);

            auto psCols = loadColumnsMulti(g_dataset_path + "partsupp.tbl",
                {{0, ColType::INT}, {1, ColType::INT}, {2, ColType::INT}});
            auto& ps_partkey = psCols.ints(0);
            auto& ps_suppkey = psCols.ints(1);
            auto& ps_availqty = psCols.ints(2);

            auto sCols = loadColumnsMulti(g_dataset_path + "supplier.tbl",
                {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}, {2, ColType::CHAR_FIXED, 40}, {3, ColType::INT}});
            auto& s_suppkey = sCols.ints(0);
            auto& s_name = sCols.chars(1);
            auto& s_address = sCols.chars(2);
            auto& s_nationkey = sCols.ints(3);

            auto nCols = loadColumnsMulti(g_dataset_path + "nation.tbl",
                {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}});
            auto& n_nationkey = nCols.ints(0);
            auto& n_name = nCols.chars(1);

            int canada_nk = -1;
            for (size_t i = 0; i < n_nationkey.size(); i++) {
                if (n_name[i * 25] == 'C' && n_name[i * 25 + 1] == 'A' &&
                    n_name[i * 25 + 2] == 'N') {
                    canada_nk = n_nationkey[i];
                    break;
                }
            }

            int maxPk = 0;
            for (int pk : p_partkey) maxPk = std::max(maxPk, pk);
            size_t bmpInts = ((size_t)maxPk + 32) / 32;
            std::vector<uint32_t> partBitmap(bmpInts, 0);
            for (size_t i = 0; i < p_partkey.size(); i++) {
                const char* nm = p_name.data() + i * 55;
                if (nm[0]=='f' && nm[1]=='o' && nm[2]=='r' && nm[3]=='e' &&
                    nm[4]=='s' && nm[5]=='t') {
                    partBitmap[p_partkey[i] / 32] |= (1u << (p_partkey[i] % 32));
                }
            }

            size_t psEntries = 0;
            for (size_t i = 0; i < ps_partkey.size(); i++) {
                int pk = ps_partkey[i];
                if (pk >= 0 && (size_t)pk / 32 < bmpInts && (partBitmap[pk / 32] >> (pk % 32)) & 1)
                    psEntries++;
            }
            uint32_t htSlots = 1;
            while (htSlots < psEntries * 2) htSlots <<= 1;
            uint32_t htMask = htSlots - 1;
            int maxSk = 0;
            for (int sk : ps_suppkey) maxSk = std::max(maxSk, sk);
            uint32_t suppMul = (uint32_t)(maxSk + 1);
            std::vector<uint32_t> htKeys(htSlots, 0xFFFFFFFFu);
            std::vector<float> htVals(htSlots, 0.0f);
            std::vector<int> htPsIdx(htSlots, -1);
            for (size_t i = 0; i < ps_partkey.size(); i++) {
                int pk = ps_partkey[i];
                if (pk < 0 || (size_t)pk / 32 >= bmpInts || !((partBitmap[pk / 32] >> (pk % 32)) & 1))
                    continue;
                uint32_t key = (uint32_t)pk * suppMul + (uint32_t)ps_suppkey[i];
                uint32_t h = (key * 2654435769u) & htMask;
                for (uint32_t s = 0; s <= htMask; s++) {
                    uint32_t slot = (h + s) & htMask;
                    if (htKeys[slot] == 0xFFFFFFFFu) {
                        htKeys[slot] = key;
                        htPsIdx[slot] = (int)i;
                        break;
                    }
                }
            }

            auto* bmpBuf = device->newBuffer(partBitmap.data(), bmpInts * sizeof(uint32_t),
                                              MTL::ResourceStorageModeShared);
            auto* htKeysBuf = device->newBuffer(htKeys.data(), htSlots * sizeof(uint32_t),
                                                 MTL::ResourceStorageModeShared);
            auto* htValsBuf = device->newBuffer(htSlots * sizeof(float),
                                                 MTL::ResourceStorageModeShared);
            memset(htValsBuf->contents(), 0, htSlots * sizeof(float));

            executor.registerAllocatedBuffer("d_q20_part_bitmap", bmpBuf);
            executor.registerAllocatedBuffer("d_q20_ht_keys", htKeysBuf);
            executor.registerAllocatedBuffer("d_q20_ht_vals", htValsBuf);
            executor.registerScalarInt("d_q20_ht_mask", (int)htMask);
            executor.registerScalarInt("supp_mul", (int)suppMul);

            g_q20Post.ps_partkey = std::move(ps_partkey);
            g_q20Post.ps_suppkey = std::move(ps_suppkey);
            g_q20Post.ps_availqty = std::move(ps_availqty);
            g_q20Post.htKeys = std::move(htKeys);
            g_q20Post.htPsIdx = std::move(htPsIdx);
            g_q20Post.htMask = htMask;
            g_q20Post.htSlots = htSlots;
            g_q20Post.s_suppkey = std::move(s_suppkey);
            g_q20Post.s_nationkey = std::move(s_nationkey);
            g_q20Post.s_name = std::move(s_name);
            g_q20Post.s_address = std::move(s_address);
            g_q20Post.canada_nk = canada_nk;
            g_q20Post.htValsBuf = htValsBuf;
        }

        // Q2: build EUROPE supplier bitmap, allocate part bitmap (filled by GPU Phase 1)
        if (plan.name == "Q2") {
            auto pCols = loadColumnsMulti(g_dataset_path + "part.tbl",
                {{0, ColType::INT}, {2, ColType::CHAR_FIXED, 25}});
            auto& p_partkey = pCols.ints(0);
            auto& p_mfgr = pCols.chars(2);

            auto sCols = loadColumnsMulti(g_dataset_path + "supplier.tbl", {
                {0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}, {2, ColType::CHAR_FIXED, 40},
                {3, ColType::INT}, {4, ColType::CHAR_FIXED, 15}, {5, ColType::FLOAT},
                {6, ColType::CHAR_FIXED, 101}
            });
            auto& s_suppkey = sCols.ints(0);
            auto& s_name = sCols.chars(1);
            auto& s_address = sCols.chars(2);
            auto& s_nationkey = sCols.ints(3);
            auto& s_phone = sCols.chars(4);
            auto& s_acctbal = sCols.floats(5);
            auto& s_comment = sCols.chars(6);

            auto psCols = loadColumnsMulti(g_dataset_path + "partsupp.tbl",
                {{0, ColType::INT}, {1, ColType::INT}, {3, ColType::FLOAT}});
            auto& ps_partkey = psCols.ints(0);
            auto& ps_suppkey = psCols.ints(1);
            auto& ps_supplycost = psCols.floats(3);

            auto nCols = loadColumnsMulti(g_dataset_path + "nation.tbl",
                {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}, {2, ColType::INT}});
            auto& n_nationkey = nCols.ints(0);
            auto& n_name = nCols.chars(1);
            auto& n_regionkey = nCols.ints(2);

            auto rCols = loadColumnsMulti(g_dataset_path + "region.tbl",
                {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}});
            auto& r_regionkey = rCols.ints(0);
            auto& r_name = rCols.chars(1);

            int europe_rk = -1;
            for (size_t i = 0; i < r_regionkey.size(); i++) {
                if (r_name[i * 25] == 'E' && r_name[i * 25 + 1] == 'U' &&
                    r_name[i * 25 + 2] == 'R' && r_name[i * 25 + 3] == 'O') {
                    europe_rk = r_regionkey[i];
                    break;
                }
            }

            std::set<int> europeNks;
            for (size_t i = 0; i < n_nationkey.size(); i++) {
                if (n_regionkey[i] == europe_rk)
                    europeNks.insert(n_nationkey[i]);
            }

            std::vector<std::string> nationNames(25);
            for (size_t i = 0; i < n_nationkey.size(); i++) {
                int len = 25;
                while (len > 0 && (n_name[i * 25 + len - 1] == ' ' || n_name[i * 25 + len - 1] == '\0')) len--;
                nationNames[n_nationkey[i]] = std::string(n_name.data() + i * 25, len);
            }

            int maxSk = 0;
            for (int sk : s_suppkey) maxSk = std::max(maxSk, sk);
            size_t suppBmpInts = ((size_t)maxSk + 32) / 32;
            std::vector<uint32_t> eurSuppBitmap(suppBmpInts, 0);
            for (size_t i = 0; i < s_suppkey.size(); i++) {
                if (europeNks.count(s_nationkey[i])) {
                    int sk = s_suppkey[i];
                    eurSuppBitmap[sk / 32] |= (1u << (sk % 32));
                }
            }

            int maxPk = 0;
            for (int pk : p_partkey) maxPk = std::max(maxPk, pk);
            size_t partBmpInts = ((size_t)maxPk + 32) / 32;

            // Part bitmap is filled by GPU Phase 1
            auto* partBmpBuf = device->newBuffer(partBmpInts * sizeof(uint32_t),
                                                  MTL::ResourceStorageModeShared);
            memset(partBmpBuf->contents(), 0, partBmpInts * sizeof(uint32_t));

            size_t minCostSize = (size_t)maxPk + 1;
            auto* minCostBuf = device->newBuffer(minCostSize * sizeof(uint32_t),
                                                  MTL::ResourceStorageModeShared);
            memset(minCostBuf->contents(), 0xFF, minCostSize * sizeof(uint32_t));

            auto* suppBmpBuf = device->newBuffer(eurSuppBitmap.data(), suppBmpInts * sizeof(uint32_t),
                                                  MTL::ResourceStorageModeShared);

            executor.registerAllocatedBuffer("d_q2_part_bitmap", partBmpBuf);
            executor.registerAllocatedBuffer("d_q2_supp_bitmap", suppBmpBuf);
            executor.registerAllocatedBuffer("d_q2_min_cost", minCostBuf);

            g_q2Post.ps_partkey = std::move(ps_partkey);
            g_q2Post.ps_suppkey = std::move(ps_suppkey);
            g_q2Post.ps_supplycost = std::move(ps_supplycost);
            g_q2Post.s_suppkey = std::move(s_suppkey);
            g_q2Post.s_nationkey = std::move(s_nationkey);
            g_q2Post.s_acctbal = std::move(s_acctbal);
            g_q2Post.s_name = std::move(s_name);
            g_q2Post.s_address = std::move(s_address);
            g_q2Post.s_phone = std::move(s_phone);
            g_q2Post.s_comment = std::move(s_comment);
            g_q2Post.p_partkey = std::move(p_partkey);
            g_q2Post.p_mfgr = std::move(p_mfgr);
            g_q2Post.nationNames = std::move(nationNames);
            g_q2Post.eurSuppBitmap = std::move(eurSuppBitmap);
            g_q2Post.maxPartkey = maxPk;
            g_q2Post.minCostBuf = minCostBuf;
        }

        // Q16: build part_group_map, allocate complaint bitmap (filled by GPU Phase 1)
        if (plan.name == "Q16") {
            auto pCols = loadColumnsMulti(g_dataset_path + "part.tbl",
                {{0, ColType::INT}, {3, ColType::CHAR_FIXED, 10}, {4, ColType::CHAR_FIXED, 25}, {5, ColType::INT}});
            auto& p_partkey = pCols.ints(0);
            auto& p_brand = pCols.chars(3);
            auto& p_type = pCols.chars(4);
            auto& p_size = pCols.ints(5);

            // Get maxSk from loaded supplier data (standard mechanism loads it for Phase 1)
            int maxSk = 0;
            const auto& tpchSchema3 = codegen::TPCHSchema::instance();
            for (auto& [tblName, cols] : loadedTables) {
                if (tblName == "supplier") {
                    auto& skCol = cols.ints(tpchSchema3.table("supplier").col("s_suppkey").index);
                    for (int sk : skCol) maxSk = std::max(maxSk, sk);
                }
            }
            size_t complaintBmpInts = ((size_t)maxSk + 32) / 32;

            // Complaint bitmap is filled by GPU Phase 1
            auto* cbmBuf = device->newBuffer(complaintBmpInts * sizeof(uint32_t),
                                              MTL::ResourceStorageModeShared);
            memset(cbmBuf->contents(), 0, complaintBmpInts * sizeof(uint32_t));

            std::set<int> validSizes = {49, 14, 23, 45, 19, 3, 36, 9};
            struct GroupKey { std::string brand; std::string type; int size;
                bool operator<(const GroupKey& o) const {
                    if (brand != o.brand) return brand < o.brand;
                    if (type != o.type) return type < o.type;
                    return size < o.size;
                }
            };
            std::map<GroupKey, int> groupMap;
            std::vector<Q16PostData::GroupKey> groups;

            int maxPk = 0;
            for (int pk : p_partkey) maxPk = std::max(maxPk, pk);
            std::vector<int> partGroupMap((size_t)maxPk + 1, -1);

            for (size_t i = 0; i < p_partkey.size(); i++) {
                const char* br = p_brand.data() + i * 10;
                int brLen = 10;
                while (brLen > 0 && (br[brLen-1] == ' ' || br[brLen-1] == '\0')) brLen--;
                std::string brand(br, brLen);

                const char* tp = p_type.data() + i * 25;
                int tpLen = 25;
                while (tpLen > 0 && (tp[tpLen-1] == ' ' || tp[tpLen-1] == '\0')) tpLen--;
                std::string type(tp, tpLen);

                int size = p_size[i];

                if (brand == "Brand#45") continue;
                if (tpLen >= 15 && tp[0]=='M' && tp[1]=='E' && tp[2]=='D' && tp[3]=='I' &&
                    tp[4]=='U' && tp[5]=='M' && tp[6]==' ' && tp[7]=='P' && tp[8]=='O' &&
                    tp[9]=='L' && tp[10]=='I' && tp[11]=='S' && tp[12]=='H' && tp[13]=='E' &&
                    tp[14]=='D') continue;
                if (!validSizes.count(size)) continue;

                GroupKey gk{brand, type, size};
                auto it = groupMap.find(gk);
                int gid;
                if (it == groupMap.end()) {
                    gid = (int)groups.size();
                    groupMap[gk] = gid;
                    groups.push_back({brand, type, size});
                } else {
                    gid = it->second;
                }
                partGroupMap[p_partkey[i]] = gid;
            }

            uint32_t numGroups = (uint32_t)groups.size();
            uint32_t bvInts = ((uint32_t)maxSk + 32) / 32;

            auto* pgmBuf = device->newBuffer(partGroupMap.data(), partGroupMap.size() * sizeof(int),
                                              MTL::ResourceStorageModeShared);
            size_t gbmBytes = (size_t)numGroups * bvInts * sizeof(uint32_t);
            auto* gbmBuf = device->newBuffer(std::max(gbmBytes, (size_t)4),
                                              MTL::ResourceStorageModeShared);
            memset(gbmBuf->contents(), 0, gbmBytes);

            executor.registerAllocatedBuffer("d_q16_part_group_map", pgmBuf);
            executor.registerAllocatedBuffer("d_q16_complaint_bitmap", cbmBuf);
            executor.registerAllocatedBuffer("d_q16_group_bitmaps", gbmBuf);
            executor.registerScalarInt("d_q16_bv_ints", (int)bvInts);

            g_q16Post.groups = std::move(groups);
            g_q16Post.bvInts = bvInts;
            g_q16Post.numGroups = numGroups;
            g_q16Post.groupBitmapsBuf = gbmBuf;
        }

        // Q18: preload orders.tbl for post-processing (avoids reloading during post)
        if (plan.name == "Q18") {
            auto oCols = loadColumnsMulti(g_dataset_path + "orders.tbl",
                {{0, ColType::INT}, {1, ColType::INT}, {3, ColType::FLOAT}, {4, ColType::DATE}});
            auto& okeys = oCols.ints(0);
            g_q18Post.o_custkey = std::move(oCols.ints(1));
            g_q18Post.o_totalprice = std::move(oCols.floats(3));
            g_q18Post.o_orderdate = std::move(oCols.ints(4));
            // Build direct-mapped orderkey -> row index
            int maxOk = 0;
            for (int k : okeys) if (k > maxOk) maxOk = k;
            g_q18Post.okLookup.assign(maxOk + 1, -1);
            for (size_t j = 0; j < okeys.size(); j++)
                g_q18Post.okLookup[okeys[j]] = (int)j;
        }

        // Q21: allocate GPU buffers for phases, build SA-supp bitmap (tiny)
        if (plan.name == "Q21") {
            // Load supplier/nation for SA bitmap + post-processing (small tables)
            auto sCols = loadColumnsMulti(g_dataset_path + "supplier.tbl",
                {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}, {3, ColType::INT}});
            auto& s_suppkey = sCols.ints(0);
            auto& s_name = sCols.chars(1);
            auto& s_nationkey = sCols.ints(3);

            auto nCols = loadColumnsMulti(g_dataset_path + "nation.tbl",
                {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}});
            auto& n_nationkey = nCols.ints(0);
            auto& n_name = nCols.chars(1);

            int sa_nk = -1;
            for (size_t i = 0; i < n_nationkey.size(); i++) {
                if (n_name[i * 25] == 'S' && n_name[i * 25 + 1] == 'A' &&
                    n_name[i * 25 + 2] == 'U' && n_name[i * 25 + 3] == 'D') {
                    sa_nk = n_nationkey[i];
                    break;
                }
            }

            int maxSk = 0;
            for (int sk : s_suppkey) maxSk = std::max(maxSk, sk);
            size_t saBmpInts = ((size_t)maxSk + 32) / 32;
            std::vector<uint32_t> saBitmap(saBmpInts, 0);
            for (size_t i = 0; i < s_suppkey.size(); i++) {
                if (s_nationkey[i] == sa_nk) {
                    int sk = s_suppkey[i];
                    saBitmap[sk / 32] |= (1u << (sk % 32));
                }
            }

            auto* saBuf = device->newBuffer(saBitmap.data(), saBmpInts * sizeof(uint32_t),
                                             MTL::ResourceStorageModeShared);
            executor.registerAllocatedBuffer("d_q21_sa_supp", saBuf);

            // Find max orderkey from loaded orders data for buffer sizing
            int maxOk = 0;
            const auto& tpchSchema2 = codegen::TPCHSchema::instance();
            for (auto& [tblName, cols] : loadedTables) {
                if (tblName == "orders") {
                    auto& okCol = cols.ints(tpchSchema2.table("orders").col("o_orderkey").index);
                    for (int ok : okCol) maxOk = std::max(maxOk, ok);
                }
            }

            size_t fBmpInts = ((size_t)maxOk + 32) / 32;
            size_t okMapSize = (size_t)maxOk + 1;

            // GPU buffers filled by Phase 1 (f_orders) and Phase 2 (multi_supp/multi_late)
            auto* fBuf = device->newBuffer(fBmpInts * sizeof(uint32_t),
                                            MTL::ResourceStorageModeShared);
            memset(fBuf->contents(), 0, fBmpInts * sizeof(uint32_t));

            auto* firstSuppBuf = device->newBuffer(okMapSize * sizeof(int),
                                                    MTL::ResourceStorageModeShared);
            memset(firstSuppBuf->contents(), 0xFF, okMapSize * sizeof(int));

            auto* firstLateBuf = device->newBuffer(okMapSize * sizeof(int),
                                                    MTL::ResourceStorageModeShared);
            memset(firstLateBuf->contents(), 0xFF, okMapSize * sizeof(int));

            auto* msBuf = device->newBuffer(fBmpInts * sizeof(uint32_t),
                                             MTL::ResourceStorageModeShared);
            memset(msBuf->contents(), 0, fBmpInts * sizeof(uint32_t));

            auto* mlBuf = device->newBuffer(fBmpInts * sizeof(uint32_t),
                                             MTL::ResourceStorageModeShared);
            memset(mlBuf->contents(), 0, fBmpInts * sizeof(uint32_t));

            executor.registerAllocatedBuffer("d_q21_f_orders", fBuf);
            executor.registerAllocatedBuffer("d_q21_first_supp", firstSuppBuf);
            executor.registerAllocatedBuffer("d_q21_first_late", firstLateBuf);
            executor.registerAllocatedBuffer("d_q21_multi_supp", msBuf);
            executor.registerAllocatedBuffer("d_q21_multi_late", mlBuf);

            size_t suppCountSize = (size_t)maxSk + 1;
            auto* suppCountBuf = device->newBuffer(suppCountSize * sizeof(uint32_t),
                                                    MTL::ResourceStorageModeShared);
            memset(suppCountBuf->contents(), 0, suppCountSize * sizeof(uint32_t));
            executor.registerAllocatedBuffer("d_q21_supp_count", suppCountBuf);

            g_q21Post.s_suppkey = std::move(s_suppkey);
            g_q21Post.s_name = std::move(s_name);
            g_q21Post.maxSuppkey = maxSk;
        }

        // 6. Execute
        auto parseEnd = std::chrono::high_resolution_clock::now();
        double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

        auto result = executor.execute(compiled, cg, 2, 1);
        result.parseTimeMs = static_cast<float>(parseMs);

        auto postStart = std::chrono::high_resolution_clock::now();

        // 7. Print results
        printf("\n%s Results:\n", queryName.c_str());
        result.result.print();

        // ---------------------------------------------------------------
        // Per-query post-processing
        // ---------------------------------------------------------------

        // Q14: compute 100 * promo / total
        if (plan.name == "Q14" && result.result.numRows() == 1 &&
            result.result.columns.size() == 2) {
            double promo = std::get<double>(result.result.rows[0][0]);
            double total = std::get<double>(result.result.rows[0][1]);
            if (total > 0)
                printf("  → promo_revenue = %.2f%%\n", 100.0 * promo / total);
        }

        // Q10: top-20 from cust_revenue array
        if (plan.name == "Q10") {
            auto* revBuf = executor.getAllocatedBuffer("d_cust_revenue");
            if (revBuf) {
                float* rev = (float*)revBuf->contents();
                size_t numCust = revBuf->length() / sizeof(float);
                std::vector<std::pair<float, int>> entries;
                for (size_t ck = 0; ck < numCust; ck++) {
                    if (rev[ck] > 0.0f) entries.push_back({rev[ck], (int)ck});
                }
                std::sort(entries.begin(), entries.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
                int show = std::min((int)entries.size(), 20);
                printf("  Top-%d customers by returned-item revenue:\n", show);
                printf("  +----------+--------------+\n");
                printf("  | c_custkey|      revenue |\n");
                printf("  +----------+--------------+\n");
                for (int j = 0; j < show; j++) {
                    printf("  | %8d | %12.2f |\n", entries[j].second, entries[j].first);
                }
                printf("  +----------+--------------+\n");
            }
        }

        // Q7: print 4 revenue bins
        if (plan.name == "Q7") {
            auto* binsBuf = executor.getAllocatedBuffer("d_revenue_bins");
            if (binsBuf) {
                float* bins = (float*)binsBuf->contents();
                const char* pair_supp[] = {"FRANCE", "GERMANY"};
                const char* pair_cust[] = {"GERMANY", "FRANCE"};
                printf("  +----------+----------+--------+-----------------+\n");
                printf("  | supp_nat | cust_nat | l_year |         revenue |\n");
                printf("  +----------+----------+--------+-----------------+\n");
                for (int p = 0; p < 2; p++) {
                    for (int y = 0; y < 2; y++) {
                        printf("  | %-8s | %-8s | %6d | $%14.2f |\n",
                               pair_supp[p], pair_cust[p], 1995 + y, bins[p * 2 + y]);
                    }
                }
                printf("  +----------+----------+--------+-----------------+\n");
            }
        }

        // Q5: nation revenue sorted desc
        if (plan.name == "Q5") {
            auto* revBuf = executor.getAllocatedBuffer("d_nation_revenue");
            if (revBuf) {
                float* rev = (float*)revBuf->contents();
                auto nat = loadNation(g_dataset_path);
                auto nationNames = buildNationNames(nat.nationkey, nat.name.data(),
                                                     NationData::NAME_WIDTH);
                std::vector<std::pair<float, int>> entries;
                for (int nk = 0; nk < 25; nk++) {
                    if (rev[nk] > 0.0f) entries.push_back({rev[nk], nk});
                }
                std::sort(entries.begin(), entries.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
                printf("  +------------------+-----------------+\n");
                printf("  | n_name           |         revenue |\n");
                printf("  +------------------+-----------------+\n");
                for (auto& [r, nk] : entries) {
                    printf("  | %-16s | $%14.2f |\n", nationNames[nk].c_str(), r);
                }
                printf("  +------------------+-----------------+\n");
            }
        }

        // Q8: market share = brazil / total per year
        if (plan.name == "Q8") {
            auto* binsBuf = executor.getAllocatedBuffer("d_result_bins");
            if (binsBuf) {
                float* bins = (float*)binsBuf->contents();
                printf("  +--------+------------+\n");
                printf("  | o_year |  mkt_share |\n");
                printf("  +--------+------------+\n");
                for (int y = 0; y < 2; y++) {
                    float brazil = bins[y];
                    float total = bins[2 + y];
                    float share = (total > 0.0f) ? (brazil / total) : 0.0f;
                    printf("  | %6d | %10.4f |\n", 1995 + y, share);
                }
                printf("  +--------+------------+\n");
            }
        }

        // Q3: top-10 by revenue desc with date/prio
        if (plan.name == "Q3") {
            auto* revBuf = executor.getAllocatedBuffer("d_order_revenue");
            auto* dateBuf = executor.getAllocatedBuffer("d_orders_date_map");
            auto* prioBuf = executor.getAllocatedBuffer("d_orders_prio_map");
            if (revBuf && dateBuf && prioBuf) {
                float* rev = (float*)revBuf->contents();
                int* dates = (int*)dateBuf->contents();
                int* prios = (int*)prioBuf->contents();
                size_t n = revBuf->length() / sizeof(float);
                std::vector<std::tuple<float, int, int, int>> entries;
                for (size_t ok = 0; ok < n; ok++) {
                    if (rev[ok] > 0.0f) {
                        entries.push_back({rev[ok], dates[ok], (int)ok, prios[ok]});
                    }
                }
                std::sort(entries.begin(), entries.end(),
                    [](auto& a, auto& b) {
                        if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) > std::get<0>(b);
                        return std::get<1>(a) < std::get<1>(b);
                    });
                int show = std::min((int)entries.size(), 10);
                printf("  +----------+--------------+------------+---------------+\n");
                printf("  |l_orderkey|      revenue | o_orderdate|o_shippriority |\n");
                printf("  +----------+--------------+------------+---------------+\n");
                for (int j = 0; j < show; j++) {
                    auto& [r, d, ok, p] = entries[j];
                    printf("  | %8d | %12.2f | %10d | %13d |\n", ok, r, d, p);
                }
                printf("  +----------+--------------+------------+---------------+\n");
            }
        }

        // Q13: histogram of order counts
        if (plan.name == "Q13") {
            auto* histBuf = executor.getAllocatedBuffer("d_histogram");
            if (histBuf) {
                uint32_t* hist = (uint32_t*)histBuf->contents();
                size_t maxBin = histBuf->length() / sizeof(uint32_t);
                std::vector<std::pair<uint32_t, int>> entries;
                for (size_t c = 0; c < maxBin; c++) {
                    if (hist[c] > 0) entries.push_back({hist[c], (int)c});
                }
                std::sort(entries.begin(), entries.end(),
                    [](auto& a, auto& b) {
                        if (a.first != b.first) return a.first > b.first;
                        return a.second > b.second;
                    });
                printf("  +--------+----------+\n");
                printf("  | c_count|  custdist|\n");
                printf("  +--------+----------+\n");
                for (auto& [dist, cnt] : entries) {
                    printf("  | %6d | %8u |\n", cnt, dist);
                }
                printf("  +--------+----------+\n");
            }
        }

        // Q22: 7 country-code bins
        if (plan.name == "Q22") {
            auto* cntBuf = executor.getAllocatedBuffer("d_q22_count");
            auto* sumBuf = executor.getAllocatedBuffer("d_q22_sum");
            if (cntBuf && sumBuf) {
                uint32_t* counts = (uint32_t*)cntBuf->contents();
                float* sums = (float*)sumBuf->contents();
                const int valid_prefixes[] = {13, 17, 18, 23, 29, 30, 31};
                printf("  +----------+----------+---------------+\n");
                printf("  | cntrycode|  numcust |    totacctbal |\n");
                printf("  +----------+----------+---------------+\n");
                for (int b = 0; b < 7; b++) {
                    if (counts[b] > 0) {
                        printf("  | %8d | %8u | %13.2f |\n",
                               valid_prefixes[b], counts[b], sums[b]);
                    }
                }
                printf("  +----------+----------+---------------+\n");
            }
        }

        // Q11: total sum → threshold → filter → sort
        if (plan.name == "Q11") {
            auto* valBuf = executor.getAllocatedBuffer("d_part_value");
            if (valBuf) {
                float* values = (float*)valBuf->contents();
                size_t n = valBuf->length() / sizeof(float);
                double globalSum = 0.0;
                for (size_t k = 0; k < n; k++) globalSum += values[k];
                double threshold = globalSum * 0.0001;
                struct Q11Entry { int partkey; double value; };
                std::vector<Q11Entry> results;
                for (size_t k = 0; k < n; k++) {
                    if (values[k] > threshold) results.push_back({(int)k, (double)values[k]});
                }
                std::sort(results.begin(), results.end(),
                    [](auto& a, auto& b) { return a.value > b.value; });
                int show = std::min((int)results.size(), 20);
                printf("  Top-%d of %zu qualifying parts (threshold %.2f):\n",
                       show, results.size(), threshold);
                printf("  +-----------+------------------+\n");
                printf("  | ps_partkey|            value |\n");
                printf("  +-----------+------------------+\n");
                for (int j = 0; j < show; j++) {
                    printf("  | %9d | %16.2f |\n", results[j].partkey, results[j].value);
                }
                printf("  +-----------+------------------+\n");
            }
        }

        // Q15: find max revenue supplier
        if (plan.name == "Q15") {
            auto* revBuf = executor.getAllocatedBuffer("d_supp_revenue");
            if (revBuf) {
                float* rev = (float*)revBuf->contents();
                size_t n = revBuf->length() / sizeof(float);
                float maxRev = 0.0f;
                for (size_t k = 0; k < n; k++) maxRev = std::max(maxRev, rev[k]);
                printf("  Top supplier(s) with max revenue %.2f:\n", maxRev);
                printf("  +----------+------------------+\n");
                printf("  |  s_suppkey|    total_revenue |\n");
                printf("  +----------+------------------+\n");
                for (size_t k = 0; k < n; k++) {
                    if (rev[k] >= maxRev - 0.01f) {
                        printf("  | %9zu | %16.2f |\n", k, rev[k]);
                    }
                }
                printf("  +----------+------------------+\n");
            }
        }

        // Q18: filter qty > 300, join with preloaded orders, sort, top 100
        if (plan.name == "Q18") {
            auto* qtyBuf = executor.getAllocatedBuffer("d_order_qty");
            if (qtyBuf) {
                float* qtys = (float*)qtyBuf->contents();
                size_t n = qtyBuf->length() / sizeof(float);
                auto& o_custkey = g_q18Post.o_custkey;
                auto& o_totalprice = g_q18Post.o_totalprice;
                auto& o_orderdate = g_q18Post.o_orderdate;
                auto& okLookup = g_q18Post.okLookup;

                struct Q18Entry { int orderkey; int custkey; float totalprice; int orderdate; float qty; };
                std::vector<Q18Entry> results;
                for (size_t ok = 0; ok < n; ok++) {
                    if (qtys[ok] > 300.0f) {
                        results.push_back({(int)ok, 0, 0.0f, 0, qtys[ok]});
                    }
                }
                for (auto& r : results) {
                    if (r.orderkey >= 0 && (size_t)r.orderkey < okLookup.size()) {
                        int idx = okLookup[r.orderkey];
                        if (idx >= 0) {
                            r.custkey = o_custkey[idx];
                            r.totalprice = o_totalprice[idx];
                            r.orderdate = o_orderdate[idx];
                        }
                    }
                }
                std::sort(results.begin(), results.end(),
                    [](auto& a, auto& b) {
                        if (a.totalprice != b.totalprice) return a.totalprice > b.totalprice;
                        return a.orderdate < b.orderdate;
                    });
                int show = std::min((int)results.size(), 100);
                printf("  Top-%d large volume orders (qty > 300):\n", show);
                printf("  +----------+----------+---------------+------------+----------+\n");
                printf("  | c_custkey| o_orderkey| o_totalprice |  o_orderdate| o_qty   |\n");
                printf("  +----------+----------+---------------+------------+----------+\n");
                for (int j = 0; j < show; j++) {
                    auto& r = results[j];
                    printf("  | %8d | %8d | %13.2f | %10d | %8.2f |\n",
                           r.custkey, r.orderkey, r.totalprice, r.orderdate, r.qty);
                }
                printf("  +----------+----------+---------------+------------+----------+\n");
            }
        }

        // Q17: read revenue and divide by 7.0
        if (plan.name == "Q17") {
            auto* revLoBuf = executor.getAllocatedBuffer("d_q17_revenue_lo");
            auto* revHiBuf = executor.getAllocatedBuffer("d_q17_revenue_hi");
            if (revLoBuf && revHiBuf) {
                uint32_t lo = *(const uint32_t*)revLoBuf->contents();
                uint32_t hi = *(const uint32_t*)revHiBuf->contents();
                uint64_t combined = ((uint64_t)hi << 32) | (uint64_t)lo;
                long rawCents = (long)combined;
                double revenue = rawCents / 100.0;
                double avgYearly = revenue / 7.0;
                printf("  +------------------+\n");
                printf("  |      avg_yearly  |\n");
                printf("  +------------------+\n");
                printf("  | %16.2f |\n", avgYearly);
                printf("  +------------------+\n");
            }
        }

        // Q9: read profit bins, sort by nation ASC, year DESC
        if (plan.name == "Q9") {
            auto* profitBuf = executor.getAllocatedBuffer("d_q9_profit");
            if (profitBuf) {
                const float* profits = (const float*)profitBuf->contents();
                auto nCols = loadColumnsMulti(g_dataset_path + "nation.tbl",
                    {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}});
                auto& n_nk = nCols.ints(0);
                auto& n_nm = nCols.chars(1);
                std::vector<std::string> nationNames(25);
                for (size_t i = 0; i < n_nk.size(); i++) {
                    std::string nm;
                    for (int c = 0; c < 25; c++) {
                        char ch = n_nm[i * 25 + c];
                        if (ch == ' ' || ch == '\0') break;
                        nm += ch;
                    }
                    nationNames[n_nk[i]] = nm;
                }

                struct Q9Row { std::string nation; int year; float profit; };
                std::vector<Q9Row> rows;
                for (int nk = 0; nk < 25; nk++) {
                    for (int yr = 1992; yr <= 1998; yr++) {
                        int bin = nk * 8 + (yr - 1992);
                        float p = profits[bin];
                        if (p != 0.0f) {
                            rows.push_back({nationNames[nk], yr, p});
                        }
                    }
                }
                std::sort(rows.begin(), rows.end(), [](const Q9Row& a, const Q9Row& b) {
                    if (a.nation != b.nation) return a.nation < b.nation;
                    return a.year > b.year;
                });
                printf("  +------------+------+---------------+\n");
                printf("  | Nation     | Year |        Profit |\n");
                printf("  +------------+------+---------------+\n");
                int show = std::min((int)rows.size(), 15);
                for (int j = 0; j < show; j++) {
                    printf("  | %-10s | %4d | $%13.2f |\n",
                           rows[j].nation.c_str(), rows[j].year, rows[j].profit);
                }
                printf("  +------------+------+---------------+\n");
                printf("  Total results: %d\n", (int)rows.size());
            }
        }

        // Q20: check HT values against availqty, filter CANADA suppliers
        if (plan.name == "Q20") {
            auto& pd = g_q20Post;
            if (pd.htValsBuf) {
                const float* htVals = (const float*)pd.htValsBuf->contents();
                std::set<int> qualSuppkeys;
                for (uint32_t slot = 0; slot < pd.htSlots; slot++) {
                    if (pd.htKeys[slot] == 0xFFFFFFFFu) continue;
                    int psIdx = pd.htPsIdx[slot];
                    if (psIdx < 0) continue;
                    float sumQty = htVals[slot];
                    if (sumQty > 0.0f && (float)pd.ps_availqty[psIdx] > 0.5f * sumQty) {
                        qualSuppkeys.insert(pd.ps_suppkey[psIdx]);
                    }
                }

                struct Q20Row { std::string name; std::string address; };
                std::vector<Q20Row> rows;
                for (size_t i = 0; i < pd.s_suppkey.size(); i++) {
                    if (pd.s_nationkey[i] != pd.canada_nk) continue;
                    if (!qualSuppkeys.count(pd.s_suppkey[i])) continue;
                    std::string nm, addr;
                    for (int c = 0; c < 25; c++) {
                        char ch = pd.s_name[i * 25 + c];
                        if (ch == ' ' || ch == '\0') break;
                        nm += ch;
                    }
                    for (int c = 0; c < 40; c++) {
                        char ch = pd.s_address[i * 40 + c];
                        if (ch == '\0') break;
                        addr += ch;
                    }
                    rows.push_back({nm, addr});
                }
                std::sort(rows.begin(), rows.end(), [](const Q20Row& a, const Q20Row& b) {
                    return a.name < b.name;
                });

                printf("  +---------------------------+------------------------------------------+\n");
                printf("  | s_name                    | s_address                                |\n");
                printf("  +---------------------------+------------------------------------------+\n");
                int show = std::min((int)rows.size(), 10);
                for (int j = 0; j < show; j++) {
                    printf("  | %-25s | %-40s |\n", rows[j].name.c_str(), rows[j].address.c_str());
                }
                printf("  +---------------------------+------------------------------------------+\n");
                printf("  Total qualifying suppliers: %d\n", (int)rows.size());
            }
        }

        // Q2: read min_cost, match suppliers, join strings, sort, top 100
        if (plan.name == "Q2") {
            auto& pd = g_q2Post;
            auto* partBmpBuf = executor.getAllocatedBuffer("d_q2_part_bitmap");
            if (pd.minCostBuf && partBmpBuf) {
                const uint32_t* minCostU = (const uint32_t*)pd.minCostBuf->contents();
                const uint32_t* partBitmap = (const uint32_t*)partBmpBuf->contents();

                std::unordered_map<int, int> suppIdx;
                for (size_t i = 0; i < pd.s_suppkey.size(); i++)
                    suppIdx[pd.s_suppkey[i]] = (int)i;

                std::unordered_map<int, int> partIdx;
                for (size_t i = 0; i < pd.p_partkey.size(); i++)
                    partIdx[pd.p_partkey[i]] = (int)i;

                struct Q2Row {
                    float s_acctbal;
                    std::string s_name, n_name, s_address, s_phone, s_comment, p_mfgr;
                    int p_partkey;
                };
                std::vector<Q2Row> rows;
                for (size_t i = 0; i < pd.ps_partkey.size(); i++) {
                    int pk = pd.ps_partkey[i];
                    int sk = pd.ps_suppkey[i];

                    if (pk < 0 || pk > pd.maxPartkey) continue;
                    if (!((partBitmap[pk / 32] >> (pk % 32)) & 1)) continue;

                    if (sk < 0 || (size_t)(sk / 32) >= pd.eurSuppBitmap.size()) continue;
                    if (!((pd.eurSuppBitmap[sk / 32] >> (sk % 32)) & 1)) continue;

                    uint32_t minU = minCostU[pk];
                    if (minU == 0xFFFFFFFFu) continue;
                    float minCost;
                    memcpy(&minCost, &minU, sizeof(float));
                    if (pd.ps_supplycost[i] != minCost) continue;

                    auto sit = suppIdx.find(sk);
                    auto pit = partIdx.find(pk);
                    if (sit == suppIdx.end() || pit == partIdx.end()) continue;
                    int si = sit->second;
                    int pi = pit->second;

                    Q2Row row;
                    row.s_acctbal = pd.s_acctbal[si];
                    row.p_partkey = pk;

                    auto extractStr = [](const std::vector<char>& data, int idx, int width) {
                        std::string s;
                        for (int c = 0; c < width; c++) {
                            char ch = data[idx * width + c];
                            if (ch == '\0') break;
                            if (ch == ' ' && c > 0) {
                                bool allSpace = true;
                                for (int d = c; d < width; d++) {
                                    if (data[idx * width + d] != ' ' && data[idx * width + d] != '\0') {
                                        allSpace = false; break;
                                    }
                                }
                                if (allSpace) break;
                            }
                            s += ch;
                        }
                        return s;
                    };

                    row.s_name = extractStr(pd.s_name, si, 25);
                    row.s_address = extractStr(pd.s_address, si, 40);
                    row.s_phone = extractStr(pd.s_phone, si, 15);
                    row.s_comment = extractStr(pd.s_comment, si, 101);
                    row.p_mfgr = extractStr(pd.p_mfgr, pi, 25);
                    row.n_name = (pd.s_nationkey[si] >= 0 && pd.s_nationkey[si] < (int)pd.nationNames.size())
                        ? pd.nationNames[pd.s_nationkey[si]] : "?";

                    rows.push_back(std::move(row));
                }

                std::sort(rows.begin(), rows.end(), [](const Q2Row& a, const Q2Row& b) {
                    if (a.s_acctbal != b.s_acctbal) return a.s_acctbal > b.s_acctbal;
                    if (a.n_name != b.n_name) return a.n_name < b.n_name;
                    if (a.s_name != b.s_name) return a.s_name < b.s_name;
                    return a.p_partkey < b.p_partkey;
                });

                int limit = std::min((int)rows.size(), 100);
                printf("\nQ2 Results:\n");
                printf("  %-10s | %-25s | %-15s | %-8s | %-25s\n",
                       "s_acctbal", "s_name", "n_name", "p_partkey", "p_mfgr");
                printf("  ----------+---------------------------+-----------------+----------+---------------------------\n");
                int show = std::min(limit, 10);
                for (int j = 0; j < show; j++) {
                    printf("  %9.2f | %-25s | %-15s | %8d | %-25s\n",
                           rows[j].s_acctbal, rows[j].s_name.c_str(), rows[j].n_name.c_str(),
                           rows[j].p_partkey, rows[j].p_mfgr.c_str());
                }
                if (limit > 10) printf("  ... (%d more rows)\n", limit - 10);
                printf("  Total rows: %d\n", limit);
            }
        }

        // Q16: popcount per-group bitmaps, sort, print
        if (plan.name == "Q16") {
            auto& pd = g_q16Post;
            if (pd.groupBitmapsBuf) {
                const uint32_t* gbm = (const uint32_t*)pd.groupBitmapsBuf->contents();

                struct Q16Result { std::string brand; std::string type; int size; int supplier_cnt; };
                std::vector<Q16Result> results;
                for (uint32_t g = 0; g < pd.numGroups; g++) {
                    int cnt = 0;
                    for (uint32_t w = 0; w < pd.bvInts; w++) {
                        cnt += __builtin_popcount(gbm[g * pd.bvInts + w]);
                    }
                    if (cnt > 0) {
                        results.push_back({pd.groups[g].brand, pd.groups[g].type, pd.groups[g].size, cnt});
                    }
                }

                std::sort(results.begin(), results.end(), [](const Q16Result& a, const Q16Result& b) {
                    if (a.supplier_cnt != b.supplier_cnt) return a.supplier_cnt > b.supplier_cnt;
                    if (a.brand != b.brand) return a.brand < b.brand;
                    if (a.type != b.type) return a.type < b.type;
                    return a.size < b.size;
                });

                printf("\nQ16 Results:\n");
                printf("  +-----------+---------------------------+------+--------------+\n");
                printf("  | p_brand   | p_type                    |p_size| supplier_cnt |\n");
                printf("  +-----------+---------------------------+------+--------------+\n");
                int show = std::min((int)results.size(), 10);
                for (int j = 0; j < show; j++) {
                    printf("  | %-9s | %-25s | %4d | %12d |\n",
                           results[j].brand.c_str(), results[j].type.c_str(),
                           results[j].size, results[j].supplier_cnt);
                }
                printf("  +-----------+---------------------------+------+--------------+\n");
                printf("  Total groups: %d\n", (int)results.size());
            }
        }

        // Q21: read per-supp counts, join names, sort, top 100
        if (plan.name == "Q21") {
            auto& pd = g_q21Post;
            auto* buf = executor.getAllocatedBuffer("d_q21_supp_count");
            if (buf) {
                const uint32_t* suppCounts = (const uint32_t*)buf->contents();

                std::unordered_map<int, int> suppIdx;
                for (size_t i = 0; i < pd.s_suppkey.size(); i++)
                    suppIdx[pd.s_suppkey[i]] = (int)i;

                struct Q21Row { std::string s_name; int numwait; };
                std::vector<Q21Row> rows;
                for (int sk = 0; sk <= pd.maxSuppkey; sk++) {
                    if (suppCounts[sk] > 0) {
                        auto it = suppIdx.find(sk);
                        if (it != suppIdx.end()) {
                            int si = it->second;
                            int len = 25;
                            while (len > 0 && (pd.s_name[si * 25 + len - 1] == ' ' ||
                                               pd.s_name[si * 25 + len - 1] == '\0')) len--;
                            rows.push_back({std::string(pd.s_name.data() + si * 25, len),
                                           (int)suppCounts[sk]});
                        }
                    }
                }

                std::sort(rows.begin(), rows.end(), [](const Q21Row& a, const Q21Row& b) {
                    if (a.numwait != b.numwait) return a.numwait > b.numwait;
                    return a.s_name < b.s_name;
                });

                int limit = std::min((int)rows.size(), 100);
                printf("\nQ21 Results:\n");
                printf("  +---------------------------+----------+\n");
                printf("  | s_name                    | numwait  |\n");
                printf("  +---------------------------+----------+\n");
                int show = std::min(limit, 10);
                for (int j = 0; j < show; j++) {
                    printf("  | %-25s | %8d |\n", rows[j].s_name.c_str(), rows[j].numwait);
                }
                printf("  +---------------------------+----------+\n");
                printf("  Total qualifying suppliers: %d\n", (int)rows.size());
            }
        }

        auto postEnd = std::chrono::high_resolution_clock::now();
        double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

        printf("\nTiming:\n");
        printTimingSummary(result.parseTimeMs, result.totalKernelTimeMs, postMs);

        executor.releaseAllocatedBuffers();
    } catch (const std::exception& e) {
        std::cerr << "Codegen error (" << queryName << "): " << e.what() << std::endl;
    }
}

// ===================================================================
// main
// ===================================================================

int main(int argc, const char* argv[]) {
    std::string query;
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "help" || arg == "--help" || arg == "-h") {
            printf("GPU Database Codegen\n");
            printf("Usage: GPUDBCodegen [sf1|sf10|sf100] q<N>\n");
            printf("  q1..q22  - Run TPC-H query via codegen pipeline\n");
            printf("  all      - Run all 22 queries\n");
            return 0;
        }
        if (arg == "sf1")  { g_dataset_path = "data/SF-1/"; continue; }
        if (arg == "sf10") { g_dataset_path = "data/SF-10/"; continue; }
        if (arg == "sf100") { g_dataset_path = "data/SF-100/"; continue; }
        query = arg;
    }

    if (query.empty()) {
        std::cerr << "Usage: GPUDBCodegen [sf1|sf10|sf100] q<N>" << std::endl;
        return 1;
    }

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "No Metal device found" << std::endl;
        return 1;
    }
    device->setShouldMaximizeConcurrentCompilation(true);
    MTL::CommandQueue* cmdQueue = device->newCommandQueue();

    auto runQuery = [&](int qNum) {
        std::string path = "sql/q" + std::to_string(qNum) + ".sql";
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "Cannot open SQL file: " << path << std::endl;
            return;
        }
        std::stringstream ss;
        ss << f.rdbuf();
        std::string sql = ss.str();
        std::string name = "Q" + std::to_string(qNum);
        runCodegenQuery(device, cmdQueue, sql, name);
    };

    if (query == "all") {
        for (int q = 1; q <= 22; q++) runQuery(q);
    } else if (query.size() >= 2 && query[0] == 'q') {
        int qNum = std::stoi(query.substr(1));
        if (qNum >= 1 && qNum <= 22) {
            runQuery(qNum);
        } else {
            std::cerr << "Unknown query: " << query << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Unknown query: " << query << std::endl;
        return 1;
    }

    pool->release();
    return 0;
}
