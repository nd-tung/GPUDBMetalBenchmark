#include "infra.h"

// ===================================================================
// TPC-H Q8 — National Market Share
// ===================================================================

void runQ8Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n--- Running TPC-H Query 8 Benchmark ---" << std::endl;

    const std::string sf_path = g_dataset_path;

    auto parseStart = std::chrono::high_resolution_clock::now();
    // Part: filter p_type = 'ECONOMY ANODIZED STEEL'
    auto pCols = loadQueryColumns(device, sf_path + "part.tbl", {{0, ColType::INT}, {4, ColType::CHAR_FIXED, 25}});

    auto s = loadSupplierBasic(sf_path);
    auto& s_suppkey = s.suppkey;
    auto& s_nationkey = s.nationkey;

    auto cCols = loadQueryColumns(device, sf_path + "customer.tbl", {{0, ColType::INT}, {3, ColType::INT}});

    auto oCols = loadQueryColumns(device, sf_path + "orders.tbl", {{0, ColType::INT}, {1, ColType::INT}, {4, ColType::DATE}});

    auto lCols = loadQueryColumns(device, sf_path + "lineitem.tbl", {{0, ColType::INT}, {1, ColType::INT}, {2, ColType::INT}, {5, ColType::FLOAT}, {6, ColType::FLOAT}});

    auto nat = loadNation(sf_path, true);
    auto reg = loadRegion(sf_path);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double cpuParseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    // Find AMERICA region, BRAZIL nation
    int america_rk = findRegionKey(reg.regionkey, reg.name.data(), RegionData::NAME_WIDTH, "AMERICA");
    int brazil_nk = findNationKey(nat, "BRAZIL");

    // Build part bitmap: p_type = 'ECONOMY ANODIZED STEEL'
    const char* p_type_p = pCols.chars(4);
    auto part_bm = buildCPUBitmap(pCols.ints(0), pCols.rows(), [&](size_t i) {
        return trimFixed(p_type_p, i, 25) == "ECONOMY ANODIZED STEEL";
    });

    // Build customer→nationkey map (only AMERICA customers)
    const int* c_custkey_p = cCols.ints(0);
    const int* c_nationkey_p = cCols.ints(3);
    size_t custCount = cCols.rows();
    int max_custkey = 0;
    for (size_t i = 0; i < custCount; i++) max_custkey = std::max(max_custkey, c_custkey_p[i]);
    uint cust_map_size = max_custkey + 1;
    std::vector<int> cust_nation_map(cust_map_size, -1);
    uint america_bitmap = buildNationBitmap(nat.nationkey, nat.regionkey, america_rk);
    for (size_t i = 0; i < custCount; i++) {
        int nk = c_nationkey_p[i];
        if ((america_bitmap >> nk) & 1)
            cust_nation_map[c_custkey_p[i]] = nk;
    }

    // Build supplier→nationkey map
    int max_suppkey = 0;
    for (int k : s_suppkey) max_suppkey = std::max(max_suppkey, k);
    uint supp_map_size = max_suppkey + 1;
    std::vector<int> supp_nation_map(supp_map_size, -1);
    for (size_t i = 0; i < s_suppkey.size(); i++) supp_nation_map[s_suppkey[i]] = s_nationkey[i];

    uint ordSize = (uint)oCols.rows();
    uint liSize = (uint)lCols.rows();

    int max_orderkey = 0;
    for (int k : oCols.intSpan(0)) max_orderkey = std::max(max_orderkey, k);
    uint ord_map_size = max_orderkey + 1;

    auto pBuildOrdersPipe = createPipeline(device, library, "q8_build_orders_map_kernel");
    auto pProbePipe = createPipeline(device, library, "q8_probe_and_aggregate_kernel");
    if (!pBuildOrdersPipe || !pProbePipe) return;

    MTL::Buffer* pPartBitmapBuf = uploadBitmap(device, part_bm);
    MTL::Buffer* pCustNationMapBuf = device->newBuffer(cust_nation_map.data(), cust_map_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSuppNationMapBuf = device->newBuffer(supp_nation_map.data(), supp_map_size * sizeof(int), MTL::ResourceStorageModeShared);

    MTL::Buffer* pOrdKeyBuf = oCols.buffer(0);
    MTL::Buffer* pOrdCustBuf = oCols.buffer(1);
    MTL::Buffer* pOrdDateBuf = oCols.buffer(4);
    MTL::Buffer* pOrdCustMapBuf = device->newBuffer((size_t)ord_map_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdYearMapBuf = device->newBuffer((size_t)ord_map_size * sizeof(int), MTL::ResourceStorageModeShared);

    MTL::Buffer* pLineOrdKeyBuf = lCols.buffer(0);
    MTL::Buffer* pLinePartKeyBuf = lCols.buffer(1);
    MTL::Buffer* pLineSuppKeyBuf = lCols.buffer(2);
    MTL::Buffer* pLinePriceBuf = lCols.buffer(5);
    MTL::Buffer* pLineDiscBuf = lCols.buffer(6);

    MTL::Buffer* pResultBinsBuf = device->newBuffer(4 * sizeof(float), MTL::ResourceStorageModeShared);

    int date_start = 19950101, date_end = 19961231;

    double gpuSec = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(pOrdCustMapBuf->contents(), -1, (size_t)ord_map_size * sizeof(int));
        memset(pOrdYearMapBuf->contents(), 0, (size_t)ord_map_size * sizeof(int));
        memset(pResultBinsBuf->contents(), 0, 4 * sizeof(float));

        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

        // Build orders map
        enc->setComputePipelineState(pBuildOrdersPipe);
        enc->setBuffer(pOrdKeyBuf, 0, 0); enc->setBuffer(pOrdCustBuf, 0, 1);
        enc->setBuffer(pOrdDateBuf, 0, 2);
        enc->setBuffer(pOrdCustMapBuf, 0, 3); enc->setBuffer(pOrdYearMapBuf, 0, 4);
        enc->setBuffer(pCustNationMapBuf, 0, 5);
        enc->setBytes(&ordSize, sizeof(ordSize), 6);
        enc->setBytes(&date_start, sizeof(date_start), 7);
        enc->setBytes(&date_end, sizeof(date_end), 8);
        enc->dispatchThreads(MTL::Size(ordSize, 1, 1), MTL::Size(256, 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // Probe lineitem
        enc->setComputePipelineState(pProbePipe);
        enc->setBuffer(pLineOrdKeyBuf, 0, 0); enc->setBuffer(pLinePartKeyBuf, 0, 1);
        enc->setBuffer(pLineSuppKeyBuf, 0, 2);
        enc->setBuffer(pLinePriceBuf, 0, 3); enc->setBuffer(pLineDiscBuf, 0, 4);
        enc->setBuffer(pPartBitmapBuf, 0, 5);
        enc->setBuffer(pOrdCustMapBuf, 0, 6); enc->setBuffer(pOrdYearMapBuf, 0, 7);
        enc->setBuffer(pSuppNationMapBuf, 0, 8);
        enc->setBuffer(pResultBinsBuf, 0, 9);
        enc->setBytes(&liSize, sizeof(liSize), 10);
        enc->setBytes(&ord_map_size, sizeof(ord_map_size), 11);
        enc->setBytes(&brazil_nk, sizeof(brazil_nk), 12);
        enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));

        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
        if (iter == 2) gpuSec = cb->GPUEndTime() - cb->GPUStartTime();
    }

    // Results
    auto postStart = std::chrono::high_resolution_clock::now();
    float* bins = (float*)pResultBinsBuf->contents();
    printf("\nTPC-H Q8 Results:\n");
    printf("+--------+------------+\n");
    printf("| o_year |  mkt_share |\n");
    printf("+--------+------------+\n");
    for (int y = 0; y < 2; y++) {
        float total = bins[2 + y];
        float mkt_share = (total > 0.0f) ? bins[y] / total : 0.0f;
        printf("| %6d | %10.6f |\n", 1995 + y, mkt_share);
    }
    printf("+--------+------------+\n");
    auto postEnd = std::chrono::high_resolution_clock::now();
    double cpuPostMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nQ8 | %u lineitem\n", liSize);
    printTimingSummary(cpuParseMs, gpuSec * 1000.0, cpuPostMs);

    releaseAll(pBuildOrdersPipe, pProbePipe, pPartBitmapBuf, pCustNationMapBuf, pSuppNationMapBuf,
              pOrdCustMapBuf, pOrdYearMapBuf,
              pResultBinsBuf);
    // Input buffers owned by pCols/cCols/oCols/lCols (QueryColumns).
}

// --- SF100 Chunked ---
void runQ8BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q8 Benchmark (SF100 Chunked) ===" << std::endl;

    MappedFile partFile, suppFile, custFile, ordFile, liFile, natFile, regFile;
    if (!partFile.open(g_dataset_path + "part.tbl") ||
        !suppFile.open(g_dataset_path + "supplier.tbl") ||
        !custFile.open(g_dataset_path + "customer.tbl") ||
        !ordFile.open(g_dataset_path + "orders.tbl") ||
        !liFile.open(g_dataset_path + "lineitem.tbl") ||
        !natFile.open(g_dataset_path + "nation.tbl") ||
        !regFile.open(g_dataset_path + "region.tbl")) {
        std::cerr << "Q8 SF100: Cannot open required TBL files" << std::endl;
        return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto partIdx = buildLineIndex(partFile);
    auto suppIdx = buildLineIndex(suppFile);
    auto custIdx = buildLineIndex(custFile);
    auto ordIdx = buildLineIndex(ordFile);
    auto liIdx = buildLineIndex(liFile);
    auto natIdx = buildLineIndex(natFile);
    auto regIdx = buildLineIndex(regFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    size_t partRows = partIdx.size(), suppRows = suppIdx.size();
    size_t custRows = custIdx.size(), ordRows = ordIdx.size(), liRows = liIdx.size();

    auto bpT0 = std::chrono::high_resolution_clock::now();
    std::vector<int> n_nationkey, n_regionkey;
    std::vector<char> n_name_chars;
    std::vector<int> r_regionkey;
    std::vector<char> r_name_chars;
    parseNationRegionSF100(natFile, natIdx, n_nationkey, n_regionkey, n_name_chars,
                           &regFile, &regIdx, &r_regionkey, &r_name_chars);

    int america_rk = findRegionKey(r_regionkey, r_name_chars.data(), 25, "AMERICA");
    int brazil_nk = -1;
    for (size_t i = 0; i < n_nationkey.size(); i++) {
        if (trimFixed(n_name_chars.data(), i, 25) == "BRAZIL") { brazil_nk = n_nationkey[i]; break; }
    }
    uint america_bitmap = buildNationBitmap(n_nationkey, n_regionkey, america_rk);

    // Build part bitmap
    std::vector<int> p_partkey(partRows);
    std::vector<char> p_type(partRows * 25);
    parseIntColumnChunk(partFile, partIdx, 0, partRows, 0, p_partkey.data());
    parseCharColumnChunkFixed(partFile, partIdx, 0, partRows, 4, 25, p_type.data());

    int max_partkey = 0;
    for (auto k : p_partkey) max_partkey = std::max(max_partkey, k);
    uint part_bitmap_ints = (max_partkey + 31) / 32 + 1;
    std::vector<uint> part_bitmap(part_bitmap_ints, 0);
    for (size_t i = 0; i < partRows; i++) {
        if (trimFixed(p_type.data(), i, 25) == "ECONOMY ANODIZED STEEL")
            part_bitmap[p_partkey[i] / 32] |= (1u << (p_partkey[i] % 32));
    }

    // Build supplier→nationkey map
    std::vector<int> s_suppkey(suppRows), s_nationkey(suppRows);
    parseIntColumnChunk(suppFile, suppIdx, 0, suppRows, 0, s_suppkey.data());
    parseIntColumnChunk(suppFile, suppIdx, 0, suppRows, 3, s_nationkey.data());
    int max_suppkey = 0;
    for (auto k : s_suppkey) max_suppkey = std::max(max_suppkey, k);
    std::vector<int> supp_nation_map(max_suppkey + 1, -1);
    for (size_t i = 0; i < suppRows; i++) supp_nation_map[s_suppkey[i]] = s_nationkey[i];

    // Build customer→nationkey map (AMERICA only)
    std::vector<int> c_custkey(custRows), c_nationkey(custRows);
    parseIntColumnChunk(custFile, custIdx, 0, custRows, 0, c_custkey.data());
    parseIntColumnChunk(custFile, custIdx, 0, custRows, 3, c_nationkey.data());
    int max_custkey = 0;
    for (auto k : c_custkey) max_custkey = std::max(max_custkey, k);
    std::vector<int> cust_nation_map(max_custkey + 1, -1);
    for (size_t i = 0; i < custRows; i++) {
        if ((america_bitmap >> c_nationkey[i]) & 1)
            cust_nation_map[c_custkey[i]] = c_nationkey[i];
    }

    // Load orders
    std::vector<int> o_orderkey(ordRows), o_custkey(ordRows), o_orderdate(ordRows);
    parseIntColumnChunk(ordFile, ordIdx, 0, ordRows, 0, o_orderkey.data());
    parseIntColumnChunk(ordFile, ordIdx, 0, ordRows, 1, o_custkey.data());
    parseDateColumnChunk(ordFile, ordIdx, 0, ordRows, 4, o_orderdate.data());
    auto bpT1 = std::chrono::high_resolution_clock::now();
    double buildParseMs = indexBuildMs + std::chrono::duration<double, std::milli>(bpT1 - bpT0).count();

    int max_orderkey = 0;
    for (auto k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    uint ord_map_size = max_orderkey + 1;
    uint ordSize = (uint)ordRows;
    int date_start = 19950101, date_end = 19961231;

    auto pBuildOrdersPipe = createPipeline(device, library, "q8_build_orders_map_kernel");
    auto pProbePipe = createPipeline(device, library, "q8_probe_and_aggregate_kernel");
    if (!pBuildOrdersPipe || !pProbePipe) return;

    MTL::Buffer* pPartBitmapBuf = device->newBuffer(part_bitmap.data(), part_bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCustNationMapBuf = device->newBuffer(cust_nation_map.data(), (max_custkey + 1) * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSuppNationMapBuf = device->newBuffer(supp_nation_map.data(), (max_suppkey + 1) * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdKeyBuf = device->newBuffer(o_orderkey.data(), ordRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdCustBuf = device->newBuffer(o_custkey.data(), ordRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdDateBuf = device->newBuffer(o_orderdate.data(), ordRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdCustMapBuf = device->newBuffer((size_t)ord_map_size * sizeof(int), MTL::ResourceStorageModeShared);
    memset(pOrdCustMapBuf->contents(), -1, (size_t)ord_map_size * sizeof(int));
    MTL::Buffer* pOrdYearMapBuf = device->newBuffer((size_t)ord_map_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pResultBinsBuf = device->newBuffer(4 * sizeof(float), MTL::ResourceStorageModeShared);
    memset(pResultBinsBuf->contents(), 0, 4 * sizeof(float));

    // Build orders map
    double buildGpuMs = 0;
    {
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pBuildOrdersPipe);
        enc->setBuffer(pOrdKeyBuf, 0, 0); enc->setBuffer(pOrdCustBuf, 0, 1);
        enc->setBuffer(pOrdDateBuf, 0, 2);
        enc->setBuffer(pOrdCustMapBuf, 0, 3); enc->setBuffer(pOrdYearMapBuf, 0, 4);
        enc->setBuffer(pCustNationMapBuf, 0, 5);
        enc->setBytes(&ordSize, sizeof(ordSize), 6);
        enc->setBytes(&date_start, sizeof(date_start), 7);
        enc->setBytes(&date_end, sizeof(date_end), 8);
        enc->dispatchThreads(MTL::Size(ordSize, 1, 1), MTL::Size(256, 1, 1));
        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
        buildGpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
    }

    // Stream lineitem
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 20, liRows);
    struct Q8Slot { MTL::Buffer* orderkey; MTL::Buffer* partkey; MTL::Buffer* suppkey;
                    MTL::Buffer* extprice; MTL::Buffer* discount; };
    Q8Slot slots[2];
    for (int s = 0; s < 2; s++) {
        slots[s].orderkey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].partkey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].suppkey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].extprice = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].discount = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
    }

    auto timing = chunkedStreamLoop(
        commandQueue, slots, 2, liRows, chunkRows,
        [&](Q8Slot& slot, size_t startRow, size_t rowCount) {
            parseIntColumnChunk(liFile, liIdx, startRow, rowCount, 0, (int*)slot.orderkey->contents());
            parseIntColumnChunk(liFile, liIdx, startRow, rowCount, 1, (int*)slot.partkey->contents());
            parseIntColumnChunk(liFile, liIdx, startRow, rowCount, 2, (int*)slot.suppkey->contents());
            parseFloatColumnChunk(liFile, liIdx, startRow, rowCount, 5, (float*)slot.extprice->contents());
            parseFloatColumnChunk(liFile, liIdx, startRow, rowCount, 6, (float*)slot.discount->contents());
        },
        [&](Q8Slot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(pProbePipe);
            enc->setBuffer(slot.orderkey, 0, 0); enc->setBuffer(slot.partkey, 0, 1);
            enc->setBuffer(slot.suppkey, 0, 2);
            enc->setBuffer(slot.extprice, 0, 3); enc->setBuffer(slot.discount, 0, 4);
            enc->setBuffer(pPartBitmapBuf, 0, 5);
            enc->setBuffer(pOrdCustMapBuf, 0, 6); enc->setBuffer(pOrdYearMapBuf, 0, 7);
            enc->setBuffer(pSuppNationMapBuf, 0, 8);
            enc->setBuffer(pResultBinsBuf, 0, 9);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 10);
            enc->setBytes(&ord_map_size, sizeof(ord_map_size), 11);
            enc->setBytes(&brazil_nk, sizeof(brazil_nk), 12);
            enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {}
    );

    float* bins = (float*)pResultBinsBuf->contents();
    printf("\nTPC-H Q8 Results:\n");
    printf("+--------+------------+\n");
    printf("| o_year |  mkt_share |\n");
    printf("+--------+------------+\n");
    for (int y = 0; y < 2; y++) {
        float total = bins[2 + y];
        float mkt_share = (total > 0.0f) ? bins[y] / total : 0.0f;
        printf("| %6d | %10.6f |\n", 1995 + y, mkt_share);
    }
    printf("+--------+------------+\n");

    printf("\nSF100 Q8 | %zu chunks | %zu lineitem\n", timing.chunkCount, liRows);
    printTimingSummary(buildParseMs + timing.parseMs, buildGpuMs + timing.gpuMs, 0.0);

    releaseAll(pBuildOrdersPipe, pProbePipe, pPartBitmapBuf, pCustNationMapBuf, pSuppNationMapBuf,
              pOrdKeyBuf, pOrdCustBuf, pOrdDateBuf, pOrdCustMapBuf, pOrdYearMapBuf, pResultBinsBuf);
    for (int s = 0; s < 2; s++)
        releaseAll(slots[s].orderkey, slots[s].partkey, slots[s].suppkey,
                  slots[s].extprice, slots[s].discount);
}
