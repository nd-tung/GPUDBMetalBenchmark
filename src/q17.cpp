#include "infra.h"

// ===================================================================
// TPC-H Q17 — Small-Quantity-Order Revenue
// ===================================================================

void runQ17Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n--- Running TPC-H Query 17 Benchmark ---" << std::endl;

    const std::string sf_path = g_dataset_path;

    auto parseStart = std::chrono::high_resolution_clock::now();
    auto pCols = loadColumnsMulti(sf_path + "part.tbl", {{0, ColType::INT}, {3, ColType::CHAR_FIXED, 10}, {6, ColType::CHAR_FIXED, 10}});
    auto& p_partkey = pCols.ints(0); auto& p_brand = pCols.chars(3); auto& p_container = pCols.chars(6);

    auto lCols = loadColumnsMulti(sf_path + "lineitem.tbl", {{1, ColType::INT}, {4, ColType::FLOAT}, {5, ColType::FLOAT}});
    auto& l_partkey = lCols.ints(1); auto& l_quantity = lCols.floats(4); auto& l_extendedprice = lCols.floats(5);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double cpuParseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    // Build part bitmap: Brand#23, MED BOX
    auto part_bm = buildCPUBitmap(p_partkey, [&](size_t i) {
        return trimFixed(p_brand.data(), i, 10) == "Brand#23" &&
               trimFixed(p_container.data(), i, 10) == "MED BOX";
    });
    int max_partkey = part_bm.max_key;

    uint liSize = (uint)l_partkey.size();
    uint mapSize = max_partkey + 1;

    auto pStatsPipe = createPipeline(device, library, "q17_aggregate_qty_stats_kernel");
    auto pRevPipe = createPipeline(device, library, "q17_sum_revenue_kernel");
    if (!pStatsPipe || !pRevPipe) return;

    MTL::Buffer* pPartBitmapBuf = uploadBitmap(device, part_bm);
    MTL::Buffer* pLinePartKeyBuf = device->newBuffer(l_partkey.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineQtyBuf = device->newBuffer(l_quantity.data(), liSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLinePriceBuf = device->newBuffer(l_extendedprice.data(), liSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSumQtyMapBuf = device->newBuffer((size_t)mapSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCountMapBuf = device->newBuffer((size_t)mapSize * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* pThresholdMapBuf = device->newBuffer((size_t)mapSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pTotalRevenueBuf = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    double gpuSec = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(pSumQtyMapBuf->contents(), 0, (size_t)mapSize * sizeof(float));
        memset(pCountMapBuf->contents(), 0, (size_t)mapSize * sizeof(uint));
        *(float*)pTotalRevenueBuf->contents() = 0.0f;

        // Pass 1: aggregate stats
        MTL::CommandBuffer* cb1 = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc1 = cb1->computeCommandEncoder();
        enc1->setComputePipelineState(pStatsPipe);
        enc1->setBuffer(pLinePartKeyBuf, 0, 0);
        enc1->setBuffer(pLineQtyBuf, 0, 1);
        enc1->setBuffer(pPartBitmapBuf, 0, 2);
        enc1->setBuffer(pSumQtyMapBuf, 0, 3);
        enc1->setBuffer(pCountMapBuf, 0, 4);
        enc1->setBytes(&liSize, sizeof(liSize), 5);
        enc1->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));
        enc1->endEncoding();
        cb1->commit(); cb1->waitUntilCompleted();

        // CPU: compute threshold = 0.2 * avg per partkey
        float* sumQty = (float*)pSumQtyMapBuf->contents();
        uint* countQty = (uint*)pCountMapBuf->contents();
        float* threshold = (float*)pThresholdMapBuf->contents();
        for (uint pk = 0; pk < mapSize; pk++) {
            if (countQty[pk] > 0) {
                threshold[pk] = 0.2f * (sumQty[pk] / (float)countQty[pk]);
            } else {
                threshold[pk] = 0.0f;
            }
        }

        // Pass 2: sum revenue
        MTL::CommandBuffer* cb2 = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc2 = cb2->computeCommandEncoder();
        enc2->setComputePipelineState(pRevPipe);
        enc2->setBuffer(pLinePartKeyBuf, 0, 0);
        enc2->setBuffer(pLineQtyBuf, 0, 1);
        enc2->setBuffer(pLinePriceBuf, 0, 2);
        enc2->setBuffer(pPartBitmapBuf, 0, 3);
        enc2->setBuffer(pThresholdMapBuf, 0, 4);
        enc2->setBuffer(pTotalRevenueBuf, 0, 5);
        enc2->setBytes(&liSize, sizeof(liSize), 6);
        enc2->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));
        enc2->endEncoding();
        cb2->commit(); cb2->waitUntilCompleted();

        if (iter == 2) {
            gpuSec = (cb1->GPUEndTime() - cb1->GPUStartTime()) +
                     (cb2->GPUEndTime() - cb2->GPUStartTime());
        }
    }

    float totalRevenue = *(float*)pTotalRevenueBuf->contents();
    float avgYearly = totalRevenue / 7.0f;
    printf("\nTPC-H Q17 Results:\n");
    printf("+------------------+\n");
    printf("|      avg_yearly  |\n");
    printf("+------------------+\n");
    printf("| %16.2f |\n", avgYearly);
    printf("+------------------+\n");

    printf("\nQ17 | %u lineitem\n", liSize);
    printTimingSummary(cpuParseMs, gpuSec * 1000.0, 0.0);

    releaseAll(pStatsPipe, pRevPipe, pPartBitmapBuf, pLinePartKeyBuf, pLineQtyBuf, pLinePriceBuf,
              pSumQtyMapBuf, pCountMapBuf, pThresholdMapBuf, pTotalRevenueBuf);
}

// --- SF100 Chunked ---
void runQ17BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q17 Benchmark (SF100 Chunked) ===" << std::endl;

    MappedFile partFile, liFile;
    if (!partFile.open(g_dataset_path + "part.tbl") ||
        !liFile.open(g_dataset_path + "lineitem.tbl")) {
        std::cerr << "Q17 SF100: Cannot open required TBL files" << std::endl;
        return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto partIdx = buildLineIndex(partFile);
    auto liIdx = buildLineIndex(liFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    auto bpT0 = std::chrono::high_resolution_clock::now();
    size_t partRows = partIdx.size(), liRows = liIdx.size();

    // Build part bitmap
    std::vector<int> p_partkey(partRows);
    std::vector<char> p_brand(partRows * 10), p_container(partRows * 10);
    parseIntColumnChunk(partFile, partIdx, 0, partRows, 0, p_partkey.data());
    parseCharColumnChunkFixed(partFile, partIdx, 0, partRows, 3, 10, p_brand.data());
    parseCharColumnChunkFixed(partFile, partIdx, 0, partRows, 6, 10, p_container.data());

    int max_partkey = 0;
    for (auto k : p_partkey) max_partkey = std::max(max_partkey, k);
    uint part_bitmap_ints = (max_partkey + 31) / 32 + 1;
    std::vector<uint> part_bitmap(part_bitmap_ints, 0);
    for (size_t i = 0; i < partRows; i++) {
        if (trimFixed(p_brand.data(), i, 10) == "Brand#23" &&
            trimFixed(p_container.data(), i, 10) == "MED BOX") {
            int pk = p_partkey[i];
            part_bitmap[pk / 32] |= (1u << (pk % 32));
        }
    }
    auto bpT1 = std::chrono::high_resolution_clock::now();
    double buildParseMs = indexBuildMs + std::chrono::duration<double, std::milli>(bpT1 - bpT0).count();

    uint mapSize = max_partkey + 1;

    auto pStatsPipe = createPipeline(device, library, "q17_aggregate_qty_stats_kernel");
    auto pRevPipe = createPipeline(device, library, "q17_sum_revenue_kernel");
    if (!pStatsPipe || !pRevPipe) return;

    MTL::Buffer* pPartBitmapBuf = device->newBuffer(part_bitmap.data(), part_bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSumQtyMapBuf = device->newBuffer((size_t)mapSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCountMapBuf = device->newBuffer((size_t)mapSize * sizeof(uint), MTL::ResourceStorageModeShared);
    memset(pSumQtyMapBuf->contents(), 0, (size_t)mapSize * sizeof(float));
    memset(pCountMapBuf->contents(), 0, (size_t)mapSize * sizeof(uint));

    // Pass 1: Stream lineitem for stats
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 12, liRows);
    struct Q17Slot { MTL::Buffer* partkey; MTL::Buffer* quantity; MTL::Buffer* extprice; };
    Q17Slot slots[2];
    for (int s = 0; s < 2; s++) {
        slots[s].partkey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].quantity = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].extprice = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
    }

    auto parseChunk = [&](Q17Slot& slot, size_t startRow, size_t rowCount) {
        parseIntColumnChunk(liFile, liIdx, startRow, rowCount, 1, (int*)slot.partkey->contents());
        parseFloatColumnChunk(liFile, liIdx, startRow, rowCount, 4, (float*)slot.quantity->contents());
        parseFloatColumnChunk(liFile, liIdx, startRow, rowCount, 5, (float*)slot.extprice->contents());
    };

    auto timing1 = chunkedStreamLoop(
        commandQueue, slots, 2, liRows, chunkRows,
        parseChunk,
        [&](Q17Slot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(pStatsPipe);
            enc->setBuffer(slot.partkey, 0, 0);
            enc->setBuffer(slot.quantity, 0, 1);
            enc->setBuffer(pPartBitmapBuf, 0, 2);
            enc->setBuffer(pSumQtyMapBuf, 0, 3);
            enc->setBuffer(pCountMapBuf, 0, 4);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 5);
            enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {}
    );

    // CPU: compute thresholds
    float* sumQty = (float*)pSumQtyMapBuf->contents();
    uint* countQty = (uint*)pCountMapBuf->contents();
    MTL::Buffer* pThresholdMapBuf = device->newBuffer((size_t)mapSize * sizeof(float), MTL::ResourceStorageModeShared);
    float* threshold = (float*)pThresholdMapBuf->contents();
    for (uint pk = 0; pk < mapSize; pk++) {
        threshold[pk] = (countQty[pk] > 0) ? 0.2f * (sumQty[pk] / (float)countQty[pk]) : 0.0f;
    }

    // Pass 2: Stream lineitem for revenue
    MTL::Buffer* pTotalRevenueBuf = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
    *(float*)pTotalRevenueBuf->contents() = 0.0f;

    auto timing2 = chunkedStreamLoop(
        commandQueue, slots, 2, liRows, chunkRows,
        parseChunk,
        [&](Q17Slot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(pRevPipe);
            enc->setBuffer(slot.partkey, 0, 0);
            enc->setBuffer(slot.quantity, 0, 1);
            enc->setBuffer(slot.extprice, 0, 2);
            enc->setBuffer(pPartBitmapBuf, 0, 3);
            enc->setBuffer(pThresholdMapBuf, 0, 4);
            enc->setBuffer(pTotalRevenueBuf, 0, 5);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 6);
            enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {}
    );

    float totalRevenue = *(float*)pTotalRevenueBuf->contents();
    float avgYearly = totalRevenue / 7.0f;
    printf("\nTPC-H Q17 Results:\n");
    printf("+------------------+\n");
    printf("|      avg_yearly  |\n");
    printf("+------------------+\n");
    printf("| %16.2f |\n", avgYearly);
    printf("+------------------+\n");

    printf("\nSF100 Q17 | %zu chunks (2 passes) | %zu lineitem\n",
           timing1.chunkCount + timing2.chunkCount, liRows);
    printTimingSummary(buildParseMs + timing1.parseMs + timing2.parseMs,
                       timing1.gpuMs + timing2.gpuMs, 0.0);

    releaseAll(pStatsPipe, pRevPipe, pPartBitmapBuf, pSumQtyMapBuf, pCountMapBuf,
              pThresholdMapBuf, pTotalRevenueBuf);
    for (int s = 0; s < 2; s++) releaseAll(slots[s].partkey, slots[s].quantity, slots[s].extprice);
}
