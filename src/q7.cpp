#include "infra.h"

// ===================================================================
// TPC-H Q7 — Volume Shipping
// ===================================================================

void runQ7Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n--- Running TPC-H Query 7 Benchmark ---" << std::endl;

    const std::string sf_path = g_dataset_path;

    auto parseStart = std::chrono::high_resolution_clock::now();
    auto s = loadSupplierBasic(sf_path);
    auto& s_suppkey = s.suppkey;
    auto& s_nationkey = s.nationkey;
    auto cCols = loadQueryColumns(device, sf_path + "customer.tbl", {{0, ColType::INT}, {3, ColType::INT}});
    auto oCols = loadQueryColumns(device, sf_path + "orders.tbl", {{0, ColType::INT}, {1, ColType::INT}});

    auto lCols = loadQueryColumns(device, sf_path + "lineitem.tbl", {{0, ColType::INT}, {2, ColType::INT}, {5, ColType::FLOAT}, {6, ColType::FLOAT}, {10, ColType::DATE}});

    auto nat = loadNation(sf_path);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double cpuParseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    // Find FRANCE and GERMANY nationkeys
    int france_nk = findNationKey(nat, "FRANCE");
    int germany_nk = findNationKey(nat, "GERMANY");
    if (france_nk == -1 || germany_nk == -1) {
        std::cerr << "Error: FRANCE/GERMANY not found" << std::endl;
        return;
    }

    uint suppSize = (uint)s_suppkey.size();
    uint custSize = (uint)cCols.rows();
    uint ordSize = (uint)oCols.rows();
    uint liSize = (uint)lCols.rows();

    int max_suppkey = 0, max_custkey = 0, max_orderkey = 0;
    for (int k : s_suppkey) max_suppkey = std::max(max_suppkey, k);
    for (int k : cCols.intSpan(0)) max_custkey = std::max(max_custkey, k);
    for (int k : oCols.intSpan(0)) max_orderkey = std::max(max_orderkey, k);
    uint supp_map_size = max_suppkey + 1;
    uint cust_map_size = max_custkey + 1;
    uint ord_map_size = max_orderkey + 1;

    auto pSuppMapPipe = createPipeline(device, library, "q7_build_supplier_map_kernel");
    auto pCustMapPipe = createPipeline(device, library, "q7_build_customer_map_kernel");
    auto pOrdMapPipe = createPipeline(device, library, "q7_build_orders_map_kernel");
    auto pProbePipe = createPipeline(device, library, "q7_probe_and_aggregate_kernel");
    if (!pSuppMapPipe || !pCustMapPipe || !pOrdMapPipe || !pProbePipe) return;

    MTL::Buffer* pSuppKeyBuf = device->newBuffer(s_suppkey.data(), suppSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSuppNationBuf = device->newBuffer(s_nationkey.data(), suppSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSuppNationMapBuf = device->newBuffer(supp_map_size * sizeof(int), MTL::ResourceStorageModeShared);

    MTL::Buffer* pCustKeyBuf = cCols.buffer(0);
    MTL::Buffer* pCustNationBuf = cCols.buffer(3);
    MTL::Buffer* pCustNationMapBuf = device->newBuffer(cust_map_size * sizeof(int), MTL::ResourceStorageModeShared);

    MTL::Buffer* pOrdKeyBuf = oCols.buffer(0);
    MTL::Buffer* pOrdCustBuf = oCols.buffer(1);
    MTL::Buffer* pOrdMapBuf = device->newBuffer((size_t)ord_map_size * sizeof(int), MTL::ResourceStorageModeShared);

    MTL::Buffer* pLineOrdKeyBuf = lCols.buffer(0);
    MTL::Buffer* pLineSuppKeyBuf = lCols.buffer(2);
    MTL::Buffer* pLineShipDateBuf = lCols.buffer(10);
    MTL::Buffer* pLinePriceBuf = lCols.buffer(5);
    MTL::Buffer* pLineDiscBuf = lCols.buffer(6);

    MTL::Buffer* pRevenueBinsBuf = device->newBuffer(4 * sizeof(float), MTL::ResourceStorageModeShared);

    int date_start = 19950101, date_end = 19961231;

    double gpuSec = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(pSuppNationMapBuf->contents(), -1, supp_map_size * sizeof(int));
        memset(pCustNationMapBuf->contents(), -1, cust_map_size * sizeof(int));
        memset(pOrdMapBuf->contents(), -1, (size_t)ord_map_size * sizeof(int));
        memset(pRevenueBinsBuf->contents(), 0, 4 * sizeof(float));

        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

        // Build maps
        enc->setComputePipelineState(pSuppMapPipe);
        enc->setBuffer(pSuppKeyBuf, 0, 0); enc->setBuffer(pSuppNationBuf, 0, 1);
        enc->setBuffer(pSuppNationMapBuf, 0, 2);
        enc->setBytes(&france_nk, sizeof(france_nk), 3);
        enc->setBytes(&germany_nk, sizeof(germany_nk), 4);
        enc->setBytes(&suppSize, sizeof(suppSize), 5);
        enc->dispatchThreads(MTL::Size(suppSize, 1, 1), MTL::Size(256, 1, 1));

        enc->setComputePipelineState(pCustMapPipe);
        enc->setBuffer(pCustKeyBuf, 0, 0); enc->setBuffer(pCustNationBuf, 0, 1);
        enc->setBuffer(pCustNationMapBuf, 0, 2);
        enc->setBytes(&france_nk, sizeof(france_nk), 3);
        enc->setBytes(&germany_nk, sizeof(germany_nk), 4);
        enc->setBytes(&custSize, sizeof(custSize), 5);
        enc->dispatchThreads(MTL::Size(custSize, 1, 1), MTL::Size(256, 1, 1));

        enc->setComputePipelineState(pOrdMapPipe);
        enc->setBuffer(pOrdKeyBuf, 0, 0); enc->setBuffer(pOrdCustBuf, 0, 1);
        enc->setBuffer(pOrdMapBuf, 0, 2);
        enc->setBytes(&ordSize, sizeof(ordSize), 3);
        enc->dispatchThreads(MTL::Size(ordSize, 1, 1), MTL::Size(256, 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // Probe lineitem
        enc->setComputePipelineState(pProbePipe);
        enc->setBuffer(pLineOrdKeyBuf, 0, 0); enc->setBuffer(pLineSuppKeyBuf, 0, 1);
        enc->setBuffer(pLineShipDateBuf, 0, 2); enc->setBuffer(pLinePriceBuf, 0, 3);
        enc->setBuffer(pLineDiscBuf, 0, 4);
        enc->setBuffer(pOrdMapBuf, 0, 5); enc->setBuffer(pCustNationMapBuf, 0, 6);
        enc->setBuffer(pSuppNationMapBuf, 0, 7); enc->setBuffer(pRevenueBinsBuf, 0, 8);
        enc->setBytes(&liSize, sizeof(liSize), 9);
        enc->setBytes(&ord_map_size, sizeof(ord_map_size), 10);
        enc->setBytes(&france_nk, sizeof(france_nk), 11);
        enc->setBytes(&germany_nk, sizeof(germany_nk), 12);
        enc->setBytes(&date_start, sizeof(date_start), 13);
        enc->setBytes(&date_end, sizeof(date_end), 14);
        enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));

        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
        if (iter == 2) gpuSec = cb->GPUEndTime() - cb->GPUStartTime();
    }

    // CPU post-processing
    auto postStart = std::chrono::high_resolution_clock::now();
    float* bins = (float*)pRevenueBinsBuf->contents();
    printf("\nTPC-H Q7 Results:\n");
    printf("+----------+----------+--------+-----------------+\n");
    printf("| supp_nat | cust_nat | l_year |         revenue |\n");
    printf("+----------+----------+--------+-----------------+\n");
    const char* pair_supp[] = {"FRANCE", "GERMANY"};
    const char* pair_cust[] = {"GERMANY", "FRANCE"};
    for (int p = 0; p < 2; p++) {
        for (int y = 0; y < 2; y++) {
            printf("| %-8s | %-8s | %6d | $%14.2f |\n",
                   pair_supp[p], pair_cust[p], 1995 + y, bins[p * 2 + y]);
        }
    }
    printf("+----------+----------+--------+-----------------+\n");
    auto postEnd = std::chrono::high_resolution_clock::now();
    double cpuPostMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nQ7 | %u lineitem\n", liSize);
    printTimingSummary(cpuParseMs, gpuSec * 1000.0, cpuPostMs);

    releaseAll(pSuppMapPipe, pCustMapPipe, pOrdMapPipe, pProbePipe,
              pSuppKeyBuf, pSuppNationBuf, pSuppNationMapBuf,
              pCustNationMapBuf,
              pOrdMapBuf,
              pRevenueBinsBuf);
    // Input buffers owned by cCols/oCols/lCols (QueryColumns).
}

// --- SF100 Chunked ---
void runQ7BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q7 Benchmark (SF100 Chunked) ===" << std::endl;

    MappedFile suppFile, custFile, ordFile, liFile, natFile;
    if (!suppFile.open(g_dataset_path + "supplier.tbl") ||
        !custFile.open(g_dataset_path + "customer.tbl") ||
        !ordFile.open(g_dataset_path + "orders.tbl") ||
        !liFile.open(g_dataset_path + "lineitem.tbl") ||
        !natFile.open(g_dataset_path + "nation.tbl")) {
        std::cerr << "Q7 SF100: Cannot open required TBL files" << std::endl;
        return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto suppIdx = buildLineIndex(suppFile);
    auto custIdx = buildLineIndex(custFile);
    auto ordIdx = buildLineIndex(ordFile);
    auto liIdx = buildLineIndex(liFile);
    auto natIdx = buildLineIndex(natFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    size_t suppRows = suppIdx.size(), custRows = custIdx.size(), ordRows = ordIdx.size(), liRows = liIdx.size();

    auto bpT0 = std::chrono::high_resolution_clock::now();
    std::vector<int> n_nationkey, n_regionkey;
    std::vector<char> n_name_chars;
    parseNationRegionSF100(natFile, natIdx, n_nationkey, n_regionkey, n_name_chars);

    int france_nk = -1, germany_nk = -1;
    for (size_t i = 0; i < n_nationkey.size(); i++) {
        std::string name = trimFixed(n_name_chars.data(), i, 25);
        if (name == "FRANCE") france_nk = n_nationkey[i];
        if (name == "GERMANY") germany_nk = n_nationkey[i];
    }

    // Load dimension tables
    std::vector<int> s_suppkey(suppRows), s_nationkey(suppRows);
    parseIntColumnChunk(suppFile, suppIdx, 0, suppRows, 0, s_suppkey.data());
    parseIntColumnChunk(suppFile, suppIdx, 0, suppRows, 3, s_nationkey.data());

    std::vector<int> c_custkey(custRows), c_nationkey(custRows);
    parseIntColumnChunk(custFile, custIdx, 0, custRows, 0, c_custkey.data());
    parseIntColumnChunk(custFile, custIdx, 0, custRows, 3, c_nationkey.data());

    std::vector<int> o_orderkey(ordRows), o_custkey(ordRows);
    parseIntColumnChunk(ordFile, ordIdx, 0, ordRows, 0, o_orderkey.data());
    parseIntColumnChunk(ordFile, ordIdx, 0, ordRows, 1, o_custkey.data());
    auto bpT1 = std::chrono::high_resolution_clock::now();
    double buildParseMs = indexBuildMs + std::chrono::duration<double, std::milli>(bpT1 - bpT0).count();

    int max_suppkey = 0, max_custkey = 0, max_orderkey = 0;
    for (auto k : s_suppkey) max_suppkey = std::max(max_suppkey, k);
    for (auto k : c_custkey) max_custkey = std::max(max_custkey, k);
    for (auto k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    uint supp_map_size = max_suppkey + 1, cust_map_size = max_custkey + 1, ord_map_size = max_orderkey + 1;
    uint suppSize = (uint)suppRows, custSize = (uint)custRows, ordSize = (uint)ordRows;

    auto pSuppMapPipe = createPipeline(device, library, "q7_build_supplier_map_kernel");
    auto pCustMapPipe = createPipeline(device, library, "q7_build_customer_map_kernel");
    auto pOrdMapPipe = createPipeline(device, library, "q7_build_orders_map_kernel");
    auto pProbePipe = createPipeline(device, library, "q7_probe_and_aggregate_kernel");
    if (!pSuppMapPipe || !pCustMapPipe || !pOrdMapPipe || !pProbePipe) return;

    MTL::Buffer* pSuppKeyBuf = device->newBuffer(s_suppkey.data(), suppRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSuppNationBuf = device->newBuffer(s_nationkey.data(), suppRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSuppNationMapBuf = device->newBuffer(supp_map_size * sizeof(int), MTL::ResourceStorageModeShared);
    memset(pSuppNationMapBuf->contents(), -1, supp_map_size * sizeof(int));

    MTL::Buffer* pCustKeyBuf = device->newBuffer(c_custkey.data(), custRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCustNationBuf = device->newBuffer(c_nationkey.data(), custRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCustNationMapBuf = device->newBuffer(cust_map_size * sizeof(int), MTL::ResourceStorageModeShared);
    memset(pCustNationMapBuf->contents(), -1, cust_map_size * sizeof(int));

    MTL::Buffer* pOrdKeyBuf = device->newBuffer(o_orderkey.data(), ordRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdCustBuf = device->newBuffer(o_custkey.data(), ordRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdMapBuf = device->newBuffer((size_t)ord_map_size * sizeof(int), MTL::ResourceStorageModeShared);
    memset(pOrdMapBuf->contents(), -1, (size_t)ord_map_size * sizeof(int));

    MTL::Buffer* pRevenueBinsBuf = device->newBuffer(4 * sizeof(float), MTL::ResourceStorageModeShared);
    memset(pRevenueBinsBuf->contents(), 0, 4 * sizeof(float));

    int date_start = 19950101, date_end = 19961231;

    // Build maps
    double buildGpuMs = 0;
    {
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

        enc->setComputePipelineState(pSuppMapPipe);
        enc->setBuffer(pSuppKeyBuf, 0, 0); enc->setBuffer(pSuppNationBuf, 0, 1);
        enc->setBuffer(pSuppNationMapBuf, 0, 2);
        enc->setBytes(&france_nk, sizeof(france_nk), 3);
        enc->setBytes(&germany_nk, sizeof(germany_nk), 4);
        enc->setBytes(&suppSize, sizeof(suppSize), 5);
        enc->dispatchThreads(MTL::Size(suppSize, 1, 1), MTL::Size(256, 1, 1));

        enc->setComputePipelineState(pCustMapPipe);
        enc->setBuffer(pCustKeyBuf, 0, 0); enc->setBuffer(pCustNationBuf, 0, 1);
        enc->setBuffer(pCustNationMapBuf, 0, 2);
        enc->setBytes(&france_nk, sizeof(france_nk), 3);
        enc->setBytes(&germany_nk, sizeof(germany_nk), 4);
        enc->setBytes(&custSize, sizeof(custSize), 5);
        enc->dispatchThreads(MTL::Size(custSize, 1, 1), MTL::Size(256, 1, 1));

        enc->setComputePipelineState(pOrdMapPipe);
        enc->setBuffer(pOrdKeyBuf, 0, 0); enc->setBuffer(pOrdCustBuf, 0, 1);
        enc->setBuffer(pOrdMapBuf, 0, 2);
        enc->setBytes(&ordSize, sizeof(ordSize), 3);
        enc->dispatchThreads(MTL::Size(ordSize, 1, 1), MTL::Size(256, 1, 1));

        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
        buildGpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
    }

    // Stream lineitem
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 20, liRows);
    struct Q7Slot { MTL::Buffer* orderkey; MTL::Buffer* suppkey; MTL::Buffer* shipdate;
                    MTL::Buffer* extprice; MTL::Buffer* discount; };
    Q7Slot slots[2];
    for (int s = 0; s < 2; s++) {
        slots[s].orderkey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].suppkey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].shipdate = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].extprice = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].discount = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
    }

    auto timing = chunkedStreamLoop(
        commandQueue, slots, 2, liRows, chunkRows,
        [&](Q7Slot& slot, size_t startRow, size_t rowCount) {
            parseIntColumnChunk(liFile, liIdx, startRow, rowCount, 0, (int*)slot.orderkey->contents());
            parseIntColumnChunk(liFile, liIdx, startRow, rowCount, 2, (int*)slot.suppkey->contents());
            parseDateColumnChunk(liFile, liIdx, startRow, rowCount, 10, (int*)slot.shipdate->contents());
            parseFloatColumnChunk(liFile, liIdx, startRow, rowCount, 5, (float*)slot.extprice->contents());
            parseFloatColumnChunk(liFile, liIdx, startRow, rowCount, 6, (float*)slot.discount->contents());
        },
        [&](Q7Slot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(pProbePipe);
            enc->setBuffer(slot.orderkey, 0, 0); enc->setBuffer(slot.suppkey, 0, 1);
            enc->setBuffer(slot.shipdate, 0, 2); enc->setBuffer(slot.extprice, 0, 3);
            enc->setBuffer(slot.discount, 0, 4);
            enc->setBuffer(pOrdMapBuf, 0, 5); enc->setBuffer(pCustNationMapBuf, 0, 6);
            enc->setBuffer(pSuppNationMapBuf, 0, 7); enc->setBuffer(pRevenueBinsBuf, 0, 8);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 9);
            enc->setBytes(&ord_map_size, sizeof(ord_map_size), 10);
            enc->setBytes(&france_nk, sizeof(france_nk), 11);
            enc->setBytes(&germany_nk, sizeof(germany_nk), 12);
            enc->setBytes(&date_start, sizeof(date_start), 13);
            enc->setBytes(&date_end, sizeof(date_end), 14);
            enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {}
    );

    float* bins = (float*)pRevenueBinsBuf->contents();
    printf("\nTPC-H Q7 Results:\n");
    printf("+----------+----------+--------+-----------------+\n");
    printf("| supp_nat | cust_nat | l_year |         revenue |\n");
    printf("+----------+----------+--------+-----------------+\n");
    const char* pair_supp[] = {"FRANCE", "GERMANY"};
    const char* pair_cust[] = {"GERMANY", "FRANCE"};
    for (int p = 0; p < 2; p++)
        for (int y = 0; y < 2; y++)
            printf("| %-8s | %-8s | %6d | $%14.2f |\n", pair_supp[p], pair_cust[p], 1995 + y, bins[p * 2 + y]);
    printf("+----------+----------+--------+-----------------+\n");

    printf("\nSF100 Q7 | %zu chunks | %zu lineitem\n", timing.chunkCount, liRows);
    printTimingSummary(buildParseMs + timing.parseMs, buildGpuMs + timing.gpuMs, 0.0);

    releaseAll(pSuppMapPipe, pCustMapPipe, pOrdMapPipe, pProbePipe,
              pSuppKeyBuf, pSuppNationBuf, pSuppNationMapBuf,
              pCustKeyBuf, pCustNationBuf, pCustNationMapBuf,
              pOrdKeyBuf, pOrdCustBuf, pOrdMapBuf, pRevenueBinsBuf);
    for (int s = 0; s < 2; s++)
        releaseAll(slots[s].orderkey, slots[s].suppkey, slots[s].shipdate,
                  slots[s].extprice, slots[s].discount);
}
