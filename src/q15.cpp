#include "infra.h"

// ===================================================================
// TPC-H Q15 — Top Supplier
// ===================================================================

void runQ15Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n--- Running TPC-H Query 15 Benchmark ---" << std::endl;

    const std::string sf_path = g_dataset_path;

    auto parseStart = std::chrono::high_resolution_clock::now();
    auto lCols = loadQueryColumns(device, sf_path + "lineitem.tbl", {{2, ColType::INT}, {5, ColType::FLOAT}, {6, ColType::FLOAT}, {10, ColType::DATE}});

    auto sCols = loadQueryColumns(device, sf_path + "supplier.tbl", {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}, {2, ColType::CHAR_FIXED, 40}, {4, ColType::CHAR_FIXED, 15}});
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double cpuParseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    uint liSize = (uint)lCols.rows();
    int max_suppkey = 0;
    for (int k : sCols.intSpan(0)) max_suppkey = std::max(max_suppkey, k);
    uint map_size = max_suppkey + 1;

    auto pAggregatePipe = createPipeline(device, library, "q15_aggregate_revenue_kernel");
    if (!pAggregatePipe) return;

    MTL::Buffer* pSuppKeyBuf = lCols.buffer(2);
    MTL::Buffer* pShipDateBuf = lCols.buffer(10);
    MTL::Buffer* pExtPriceBuf = lCols.buffer(5);
    MTL::Buffer* pDiscountBuf = lCols.buffer(6);
    MTL::Buffer* pRevenueMapBuf = device->newBuffer(map_size * sizeof(float), MTL::ResourceStorageModeShared);

    int date_start = 19960101, date_end = 19960401;

    double gpuSec = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(pRevenueMapBuf->contents(), 0, map_size * sizeof(float));

        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pAggregatePipe);
        enc->setBuffer(pSuppKeyBuf, 0, 0);
        enc->setBuffer(pShipDateBuf, 0, 1);
        enc->setBuffer(pExtPriceBuf, 0, 2);
        enc->setBuffer(pDiscountBuf, 0, 3);
        enc->setBuffer(pRevenueMapBuf, 0, 4);
        enc->setBytes(&liSize, sizeof(liSize), 5);
        enc->setBytes(&date_start, sizeof(date_start), 6);
        enc->setBytes(&date_end, sizeof(date_end), 7);
        enc->dispatchThreads(MTL::Size(liSize, 1, 1), MTL::Size(256, 1, 1));
        enc->endEncoding();

        cb->commit();
        cb->waitUntilCompleted();
        if (iter == 2) gpuSec = cb->GPUEndTime() - cb->GPUStartTime();
    }

    // CPU post-processing: find max revenue, print matching supplier(s)
    auto postStart = std::chrono::high_resolution_clock::now();
    float* revenue_map = (float*)pRevenueMapBuf->contents();
    float max_revenue = 0.0f;
    for (uint i = 0; i < map_size; i++) {
        if (revenue_map[i] > max_revenue) max_revenue = revenue_map[i];
    }

    // Build suppkey → row index
    std::vector<size_t> supp_index(map_size, SIZE_MAX);
    const int* s_suppkey_p = sCols.ints(0);
    for (size_t i = 0; i < sCols.rows(); i++) supp_index[s_suppkey_p[i]] = i;

    printf("\nTPC-H Q15 Results:\n");
    printf("+---------+------------------+------------------+------------------+------------------+\n");
    printf("| suppkey |           s_name |        s_address |          s_phone |    total_revenue |\n");
    printf("+---------+------------------+------------------+------------------+------------------+\n");
    for (uint i = 0; i < map_size; i++) {
        if (revenue_map[i] == max_revenue && max_revenue > 0.0f) {
            size_t si = supp_index[i];
            if (si == SIZE_MAX) continue;
            printf("| %7d | %-16s | %-16s | %-16s | %16.2f |\n",
                   (int)i,
                   trimFixed(sCols.chars(1), si, 25).c_str(),
                   trimFixed(sCols.chars(2), si, 40).c_str(),
                   trimFixed(sCols.chars(4), si, 15).c_str(),
                   revenue_map[i]);
        }
    }
    printf("+---------+------------------+------------------+------------------+------------------+\n");

    auto postEnd = std::chrono::high_resolution_clock::now();
    double cpuPostMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    double gpuMs = gpuSec * 1000.0;
    printf("\nQ15 | %u rows (lineitem)\n", liSize);
    printTimingSummary(cpuParseMs, gpuMs, cpuPostMs);

    releaseAll(pAggregatePipe, pRevenueMapBuf);
    // Input buffers owned by lCols (QueryColumns).
}

// --- SF100 Chunked ---
void runQ15BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q15 Benchmark (SF100 Chunked) ===" << std::endl;

    MappedFile liFile, suppFile;
    if (!liFile.open(g_dataset_path + "lineitem.tbl") ||
        !suppFile.open(g_dataset_path + "supplier.tbl")) {
        std::cerr << "Q15 SF100: Cannot open required TBL files" << std::endl;
        return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto liIdx = buildLineIndex(liFile);
    auto suppIdx = buildLineIndex(suppFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    size_t liRows = liIdx.size(), suppRows = suppIdx.size();

    // Load supplier (small)
    auto bpT0 = std::chrono::high_resolution_clock::now();
    std::vector<int> s_suppkey(suppRows);
    std::vector<char> s_name(suppRows * 25), s_address(suppRows * 40), s_phone(suppRows * 15);
    parseIntColumnChunk(suppFile, suppIdx, 0, suppRows, 0, s_suppkey.data());
    parseCharColumnChunkFixed(suppFile, suppIdx, 0, suppRows, 1, 25, s_name.data());
    parseCharColumnChunkFixed(suppFile, suppIdx, 0, suppRows, 2, 40, s_address.data());
    parseCharColumnChunkFixed(suppFile, suppIdx, 0, suppRows, 4, 15, s_phone.data());
    auto bpT1 = std::chrono::high_resolution_clock::now();
    double buildParseMs = indexBuildMs + std::chrono::duration<double, std::milli>(bpT1 - bpT0).count();

    int max_suppkey = 0;
    for (size_t i = 0; i < suppRows; i++) max_suppkey = std::max(max_suppkey, s_suppkey[i]);
    uint map_size = max_suppkey + 1;

    auto pAggregatePipe = createPipeline(device, library, "q15_aggregate_revenue_kernel");
    if (!pAggregatePipe) return;

    MTL::Buffer* pRevenueMapBuf = device->newBuffer(map_size * sizeof(float), MTL::ResourceStorageModeShared);
    memset(pRevenueMapBuf->contents(), 0, map_size * sizeof(float));

    int date_start = 19960101, date_end = 19960401;

    // Stream lineitem
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 16, liRows);
    struct Q15Slot { MTL::Buffer* suppkey; MTL::Buffer* shipdate; MTL::Buffer* extprice; MTL::Buffer* discount; };
    Q15Slot slots[2];
    for (int s = 0; s < 2; s++) {
        slots[s].suppkey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].shipdate = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].extprice = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].discount = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
    }

    auto timing = chunkedStreamLoop(
        commandQueue, slots, 2, liRows, chunkRows,
        [&](Q15Slot& slot, size_t startRow, size_t rowCount) {
            parseIntColumnChunk(liFile, liIdx, startRow, rowCount, 2, (int*)slot.suppkey->contents());
            parseDateColumnChunk(liFile, liIdx, startRow, rowCount, 10, (int*)slot.shipdate->contents());
            parseFloatColumnChunk(liFile, liIdx, startRow, rowCount, 5, (float*)slot.extprice->contents());
            parseFloatColumnChunk(liFile, liIdx, startRow, rowCount, 6, (float*)slot.discount->contents());
        },
        [&](Q15Slot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(pAggregatePipe);
            enc->setBuffer(slot.suppkey, 0, 0);
            enc->setBuffer(slot.shipdate, 0, 1);
            enc->setBuffer(slot.extprice, 0, 2);
            enc->setBuffer(slot.discount, 0, 3);
            enc->setBuffer(pRevenueMapBuf, 0, 4);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 5);
            enc->setBytes(&date_start, sizeof(date_start), 6);
            enc->setBytes(&date_end, sizeof(date_end), 7);
            enc->dispatchThreads(MTL::Size(chunkSize, 1, 1), MTL::Size(256, 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {}
    );

    // CPU post-processing
    auto postT0 = std::chrono::high_resolution_clock::now();
    float* revenue_map = (float*)pRevenueMapBuf->contents();
    float max_revenue = 0.0f;
    for (uint i = 0; i < map_size; i++) {
        if (revenue_map[i] > max_revenue) max_revenue = revenue_map[i];
    }

    std::vector<size_t> supp_index(map_size, SIZE_MAX);
    for (size_t i = 0; i < suppRows; i++) supp_index[s_suppkey[i]] = i;

    printf("\nTPC-H Q15 Results:\n");
    printf("+---------+------------------+------------------+\n");
    printf("| suppkey |           s_name |    total_revenue |\n");
    printf("+---------+------------------+------------------+\n");
    for (uint i = 0; i < map_size; i++) {
        if (revenue_map[i] == max_revenue && max_revenue > 0.0f) {
            size_t si = supp_index[i];
            if (si == SIZE_MAX) continue;
            printf("| %7d | %-16s | %16.2f |\n",
                   (int)i, trimFixed(s_name.data(), si, 25).c_str(), revenue_map[i]);
        }
    }
    printf("+---------+------------------+------------------+\n");
    auto postT1 = std::chrono::high_resolution_clock::now();
    double cpuPostMs = std::chrono::duration<double, std::milli>(postT1 - postT0).count();

    double allParseMs = buildParseMs + timing.parseMs;
    printf("\nSF100 Q15 | %zu chunks | %zu rows\n", timing.chunkCount, liRows);
    printTimingSummary(allParseMs, timing.gpuMs, cpuPostMs);

    releaseAll(pAggregatePipe, pRevenueMapBuf);
    for (int s = 0; s < 2; s++)
        releaseAll(slots[s].suppkey, slots[s].shipdate, slots[s].extprice, slots[s].discount);
}
