#include "infra.h"

// ===================================================================
// TPC-H Q10 — Returned Item Reporting
// ===================================================================

void runQ10Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n--- Running TPC-H Query 10 Benchmark ---" << std::endl;

    const std::string sf_path = g_dataset_path;

    auto parseStart = std::chrono::high_resolution_clock::now();
    auto cCols = loadQueryColumns(device, sf_path + "customer.tbl", {
        {0, ColType::INT}, {1, ColType::CHAR_FIXED, 18}, {2, ColType::CHAR_FIXED, 40},
        {3, ColType::INT}, {4, ColType::CHAR_FIXED, 15}, {5, ColType::FLOAT}, {7, ColType::CHAR_FIXED, 117}
    });
    const int* c_custkey_p = cCols.ints(0);
    const char* c_name_p = cCols.chars(1);
    const int* c_nationkey_p = cCols.ints(3);
    const float* c_acctbal_p = cCols.floats(5);
    size_t custCount = cCols.rows();

    auto oCols = loadQueryColumns(device, sf_path + "orders.tbl", {{0, ColType::INT}, {1, ColType::INT}, {4, ColType::DATE}});

    auto lCols = loadQueryColumns(device, sf_path + "lineitem.tbl", {{0, ColType::INT}, {5, ColType::FLOAT}, {6, ColType::FLOAT}, {8, ColType::CHAR1}});

    auto nCols = loadQueryColumns(device, sf_path + "nation.tbl", {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}});
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double cpuParseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    uint ordSize = (uint)oCols.rows();
    uint liSize = (uint)lCols.rows();

    // Build custkey → row index and nation names
    int max_custkey = 0;
    for (size_t i = 0; i < custCount; i++) max_custkey = std::max(max_custkey, c_custkey_p[i]);
    std::vector<size_t> cust_index(max_custkey + 1, SIZE_MAX);
    for (size_t i = 0; i < custCount; i++) cust_index[c_custkey_p[i]] = i;

    auto nat = loadNation(sf_path);
    auto nation_names = buildNationNames(nat.nationkey, nat.name.data(), NationData::NAME_WIDTH);

    // Setup GPU
    auto pBuildOrdersPipe = createPipeline(device, library, "q10_build_orders_map_kernel");
    auto pProbeAggPipe = createPipeline(device, library, "q10_probe_and_aggregate_kernel");
    if (!pBuildOrdersPipe || !pProbeAggPipe) return;

    int max_orderkey = 0;
    for (int k : oCols.intSpan(0)) max_orderkey = std::max(max_orderkey, k);
    uint map_size = max_orderkey + 1;

    MTL::Buffer* pOrdKeyBuf = oCols.buffer(0);
    MTL::Buffer* pOrdCustBuf = oCols.buffer(1);
    MTL::Buffer* pOrdDateBuf = oCols.buffer(4);
    MTL::Buffer* pOrdersMapBuf = device->newBuffer((size_t)map_size * sizeof(int), MTL::ResourceStorageModeShared);

    MTL::Buffer* pLineOrdKeyBuf = lCols.buffer(0);
    MTL::Buffer* pLineRetFlagBuf = lCols.buffer(8);
    MTL::Buffer* pLinePriceBuf = lCols.buffer(5);
    MTL::Buffer* pLineDiscBuf = lCols.buffer(6);

    uint cust_rev_size = max_custkey + 1;
    MTL::Buffer* pCustRevenueBuf = device->newBuffer(cust_rev_size * sizeof(float), MTL::ResourceStorageModeShared);

    int date_start = 19931001, date_end = 19940101; // 1993-10-01 to 1994-01-01

    double gpuSec = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(pOrdersMapBuf->contents(), -1, (size_t)map_size * sizeof(int));
        memset(pCustRevenueBuf->contents(), 0, cust_rev_size * sizeof(float));

        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

        // Build orders map
        enc->setComputePipelineState(pBuildOrdersPipe);
        enc->setBuffer(pOrdKeyBuf, 0, 0);
        enc->setBuffer(pOrdCustBuf, 0, 1);
        enc->setBuffer(pOrdDateBuf, 0, 2);
        enc->setBuffer(pOrdersMapBuf, 0, 3);
        enc->setBytes(&ordSize, sizeof(ordSize), 4);
        enc->setBytes(&date_start, sizeof(date_start), 5);
        enc->setBytes(&date_end, sizeof(date_end), 6);
        enc->dispatchThreads(MTL::Size(ordSize, 1, 1), MTL::Size(256, 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // Probe lineitem
        enc->setComputePipelineState(pProbeAggPipe);
        enc->setBuffer(pLineOrdKeyBuf, 0, 0);
        enc->setBuffer(pLineRetFlagBuf, 0, 1);
        enc->setBuffer(pLinePriceBuf, 0, 2);
        enc->setBuffer(pLineDiscBuf, 0, 3);
        enc->setBuffer(pOrdersMapBuf, 0, 4);
        enc->setBuffer(pCustRevenueBuf, 0, 5);
        enc->setBytes(&liSize, sizeof(liSize), 6);
        enc->setBytes(&map_size, sizeof(map_size), 7);
        enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));

        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        if (iter == 2) gpuSec = cb->GPUEndTime() - cb->GPUStartTime();
    }

    // CPU post-processing: Top 20 by revenue DESC using bounded min-heap
    auto postStart = std::chrono::high_resolution_clock::now();
    float* cust_revenue = (float*)pCustRevenueBuf->contents();

    struct Q10Result { int custkey; float revenue; };
    auto cmp = [](const Q10Result& a, const Q10Result& b) { return a.revenue > b.revenue; };
    std::vector<Q10Result> topHeap;
    topHeap.reserve(21);
    for (uint i = 0; i < cust_rev_size; i++) {
        if (cust_revenue[i] > 0.0f) {
            if (topHeap.size() < 20) {
                topHeap.push_back({(int)i, cust_revenue[i]});
                std::push_heap(topHeap.begin(), topHeap.end(), cmp);
            } else if (cust_revenue[i] > topHeap.front().revenue) {
                std::pop_heap(topHeap.begin(), topHeap.end(), cmp);
                topHeap.back() = {(int)i, cust_revenue[i]};
                std::push_heap(topHeap.begin(), topHeap.end(), cmp);
            }
        }
    }
    std::sort_heap(topHeap.begin(), topHeap.end(), cmp);

    printf("\nTPC-H Q10 Results (Top 20):\n");
    printf("+---------+------------------+------------+----------+------------------+\n");
    printf("| custkey |           c_name |    revenue | c_acctbal|           n_name |\n");
    printf("+---------+------------------+------------+----------+------------------+\n");
    for (size_t i = 0; i < topHeap.size(); i++) {
        int ck = topHeap[i].custkey;
        size_t ci = cust_index[ck];
        if (ci == SIZE_MAX) continue;
        printf("| %7d | %-16s | $%10.2f| %8.2f | %-16s |\n",
               ck, trimFixed(c_name_p, ci, 18).c_str(),
               topHeap[i].revenue, c_acctbal_p[ci],
               nation_names[c_nationkey_p[ci]].c_str());
    }
    printf("+---------+------------------+------------+----------+------------------+\n");

    auto postEnd = std::chrono::high_resolution_clock::now();
    double cpuPostMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nQ10 | %u orders, %u lineitem\n", ordSize, liSize);
    printTimingSummary(cpuParseMs, gpuSec * 1000.0, cpuPostMs);

    releaseAll(pBuildOrdersPipe, pProbeAggPipe,
              pOrdersMapBuf, pCustRevenueBuf);
    // Input buffers owned by cCols/oCols/lCols/nCols (QueryColumns).
}

// --- SF100 Chunked ---
void runQ10BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q10 Benchmark (SF100 Chunked) ===" << std::endl;

    MappedFile custFile, ordFile, liFile, natFile;
    if (!custFile.open(g_dataset_path + "customer.tbl") ||
        !ordFile.open(g_dataset_path + "orders.tbl") ||
        !liFile.open(g_dataset_path + "lineitem.tbl") ||
        !natFile.open(g_dataset_path + "nation.tbl")) {
        std::cerr << "Q10 SF100: Cannot open required TBL files" << std::endl;
        return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto custIdx = buildLineIndex(custFile);
    auto ordIdx = buildLineIndex(ordFile);
    auto liIdx = buildLineIndex(liFile);
    auto natIdx = buildLineIndex(natFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    size_t custRows = custIdx.size(), ordRows = ordIdx.size(), liRows = liIdx.size();

    // Load dimension tables
    auto bpT0 = std::chrono::high_resolution_clock::now();
    std::vector<int> n_nationkey, n_regionkey;
    std::vector<char> n_name_chars;
    parseNationRegionSF100(natFile, natIdx, n_nationkey, n_regionkey, n_name_chars);
    auto nation_names = buildNationNames(n_nationkey, n_name_chars.data(), 25);

    std::vector<int> c_custkey(custRows), c_nationkey(custRows);
    std::vector<float> c_acctbal(custRows);
    std::vector<char> c_name(custRows * 18);
    parseIntColumnChunk(custFile, custIdx, 0, custRows, 0, c_custkey.data());
    parseCharColumnChunkFixed(custFile, custIdx, 0, custRows, 1, 18, c_name.data());
    parseIntColumnChunk(custFile, custIdx, 0, custRows, 3, c_nationkey.data());
    parseFloatColumnChunk(custFile, custIdx, 0, custRows, 5, c_acctbal.data());

    int max_custkey = 0;
    for (auto k : c_custkey) max_custkey = std::max(max_custkey, k);
    std::vector<size_t> cust_index(max_custkey + 1, SIZE_MAX);
    for (size_t i = 0; i < custRows; i++) cust_index[c_custkey[i]] = i;

    // Load orders
    std::vector<int> o_orderkey(ordRows), o_custkey(ordRows), o_orderdate(ordRows);
    parseIntColumnChunk(ordFile, ordIdx, 0, ordRows, 0, o_orderkey.data());
    parseIntColumnChunk(ordFile, ordIdx, 0, ordRows, 1, o_custkey.data());
    parseDateColumnChunk(ordFile, ordIdx, 0, ordRows, 4, o_orderdate.data());
    auto bpT1 = std::chrono::high_resolution_clock::now();
    double buildParseMs = indexBuildMs + std::chrono::duration<double, std::milli>(bpT1 - bpT0).count();

    // Setup GPU
    auto pBuildOrdersPipe = createPipeline(device, library, "q10_build_orders_map_kernel");
    auto pProbeAggPipe = createPipeline(device, library, "q10_probe_and_aggregate_kernel");
    if (!pBuildOrdersPipe || !pProbeAggPipe) return;

    int max_orderkey = 0;
    for (auto k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    uint map_size = max_orderkey + 1;
    uint ordSize = (uint)ordRows;
    int date_start = 19931001, date_end = 19940101;

    MTL::Buffer* pOrdKeyBuf = device->newBuffer(o_orderkey.data(), ordRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdCustBuf = device->newBuffer(o_custkey.data(), ordRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdDateBuf = device->newBuffer(o_orderdate.data(), ordRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdersMapBuf = device->newBuffer((size_t)map_size * sizeof(int), MTL::ResourceStorageModeShared);
    memset(pOrdersMapBuf->contents(), -1, (size_t)map_size * sizeof(int));

    uint cust_rev_size = max_custkey + 1;
    MTL::Buffer* pCustRevenueBuf = device->newBuffer(cust_rev_size * sizeof(float), MTL::ResourceStorageModeShared);
    memset(pCustRevenueBuf->contents(), 0, cust_rev_size * sizeof(float));

    // Build orders map
    double buildGpuMs = 0;
    {
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pBuildOrdersPipe);
        enc->setBuffer(pOrdKeyBuf, 0, 0);
        enc->setBuffer(pOrdCustBuf, 0, 1);
        enc->setBuffer(pOrdDateBuf, 0, 2);
        enc->setBuffer(pOrdersMapBuf, 0, 3);
        enc->setBytes(&ordSize, sizeof(ordSize), 4);
        enc->setBytes(&date_start, sizeof(date_start), 5);
        enc->setBytes(&date_end, sizeof(date_end), 6);
        enc->dispatchThreads(MTL::Size(ordSize, 1, 1), MTL::Size(256, 1, 1));
        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
        buildGpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
    }

    // Stream lineitem
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 14, liRows);
    struct Q10Slot { MTL::Buffer* orderkey; MTL::Buffer* retflag; MTL::Buffer* extprice; MTL::Buffer* discount; };
    Q10Slot slots[2];
    for (int s = 0; s < 2; s++) {
        slots[s].orderkey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].retflag = device->newBuffer(chunkRows * sizeof(char), MTL::ResourceStorageModeShared);
        slots[s].extprice = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].discount = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
    }

    auto timing = chunkedStreamLoop(
        commandQueue, slots, 2, liRows, chunkRows,
        [&](Q10Slot& slot, size_t startRow, size_t rowCount) {
            parseIntColumnChunk(liFile, liIdx, startRow, rowCount, 0, (int*)slot.orderkey->contents());
            parseCharColumnChunk(liFile, liIdx, startRow, rowCount, 8, (char*)slot.retflag->contents());
            parseFloatColumnChunk(liFile, liIdx, startRow, rowCount, 5, (float*)slot.extprice->contents());
            parseFloatColumnChunk(liFile, liIdx, startRow, rowCount, 6, (float*)slot.discount->contents());
        },
        [&](Q10Slot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(pProbeAggPipe);
            enc->setBuffer(slot.orderkey, 0, 0);
            enc->setBuffer(slot.retflag, 0, 1);
            enc->setBuffer(slot.extprice, 0, 2);
            enc->setBuffer(slot.discount, 0, 3);
            enc->setBuffer(pOrdersMapBuf, 0, 4);
            enc->setBuffer(pCustRevenueBuf, 0, 5);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 6);
            enc->setBytes(&map_size, sizeof(map_size), 7);
            enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {}
    );

    // CPU post-processing: Top 20 using bounded min-heap
    auto postT0 = std::chrono::high_resolution_clock::now();
    float* cust_revenue = (float*)pCustRevenueBuf->contents();
    struct Q10Result { int custkey; float revenue; };
    auto cmp = [](const Q10Result& a, const Q10Result& b) { return a.revenue > b.revenue; };
    std::vector<Q10Result> topHeap;
    topHeap.reserve(21);
    for (uint i = 0; i < cust_rev_size; i++) {
        if (cust_revenue[i] > 0.0f) {
            if (topHeap.size() < 20) {
                topHeap.push_back({(int)i, cust_revenue[i]});
                std::push_heap(topHeap.begin(), topHeap.end(), cmp);
            } else if (cust_revenue[i] > topHeap.front().revenue) {
                std::pop_heap(topHeap.begin(), topHeap.end(), cmp);
                topHeap.back() = {(int)i, cust_revenue[i]};
                std::push_heap(topHeap.begin(), topHeap.end(), cmp);
            }
        }
    }
    std::sort_heap(topHeap.begin(), topHeap.end(), cmp);

    printf("\nTPC-H Q10 Results (Top 20):\n");
    printf("+---------+------------------+------------+\n");
    printf("| custkey |           c_name |    revenue |\n");
    printf("+---------+------------------+------------+\n");
    for (size_t i = 0; i < topHeap.size(); i++) {
        int ck = topHeap[i].custkey;
        size_t ci = (ck >= 0 && (size_t)ck < cust_index.size()) ? cust_index[ck] : SIZE_MAX;
        if (ci == SIZE_MAX) continue;
        printf("| %7d | %-16s | $%10.2f|\n",
               ck, trimFixed(c_name.data(), ci, 18).c_str(), topHeap[i].revenue);
    }
    printf("+---------+------------------+------------+\n");
    auto postT1 = std::chrono::high_resolution_clock::now();
    double cpuPostMs = std::chrono::duration<double, std::milli>(postT1 - postT0).count();

    printf("\nSF100 Q10 | %zu chunks | %zu lineitem\n", timing.chunkCount, liRows);
    printTimingSummary(buildParseMs + timing.parseMs, buildGpuMs + timing.gpuMs, cpuPostMs);

    releaseAll(pBuildOrdersPipe, pProbeAggPipe,
              pOrdKeyBuf, pOrdCustBuf, pOrdDateBuf, pOrdersMapBuf, pCustRevenueBuf);
    for (int s = 0; s < 2; s++)
        releaseAll(slots[s].orderkey, slots[s].retflag, slots[s].extprice, slots[s].discount);
}
