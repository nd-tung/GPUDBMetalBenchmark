#include "infra.h"

// ===================================================================
// TPC-H Q14 — Promotion Effect
// ===================================================================

// --- Standard (SF1/SF10) ---
void runQ14Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "--- Running TPC-H Query 14 Benchmark ---" << std::endl;

    auto q14ParseStart = std::chrono::high_resolution_clock::now();
    const std::string sf_path = g_dataset_path;

    auto pCols = loadQueryColumns(device, sf_path + "part.tbl", {{0, ColType::INT}, {4, ColType::CHAR_FIXED, 25}});
    auto lCols = loadQueryColumns(device, sf_path + "lineitem.tbl", {{1, ColType::INT}, {5, ColType::FLOAT}, {6, ColType::FLOAT}, {10, ColType::DATE}});
    auto q14ParseEnd = std::chrono::high_resolution_clock::now();
    double cpuParseMs = std::chrono::duration<double, std::milli>(q14ParseEnd - q14ParseStart).count();

    uint dataSize = (uint)lCols.rows();
    uint partSize = (uint)pCols.rows();
    if (dataSize == 0) { std::cerr << "Q14: no data loaded" << std::endl; return; }
    std::cout << "Loaded " << dataSize << " lineitem rows for Q14." << std::endl;

    int max_partkey = 0;
    for (int k : pCols.intSpan(0)) max_partkey = std::max(max_partkey, k);
    uint bitmapInts = (max_partkey + 31) / 32 + 1;
    const uint type_stride = 25;

    auto bitmapPSO = createPipeline(device, library, "q14_build_promo_bitmap");
    auto s1PSO = createPipeline(device, library, "q14_filter_and_sum_stage1");
    auto s2PSO = createPipeline(device, library, "q14_final_sum_stage2");
    if (!bitmapPSO || !s1PSO || !s2PSO) return;

    const int numTG = 2048;
    int start_date = 19950901, end_date = 19951001;

    MTL::Buffer* pPartKeyBuf = pCols.buffer(0);
    MTL::Buffer* pTypeBuf    = pCols.buffer(4);
    MTL::Buffer* bitmapBuf   = device->newBuffer(bitmapInts * sizeof(uint), MTL::ResourceStorageModeShared);
    memset(bitmapBuf->contents(), 0, bitmapInts * sizeof(uint));
    MTL::Buffer* partkeyBuf  = lCols.buffer(1);
    MTL::Buffer* shipdateBuf = lCols.buffer(10);
    MTL::Buffer* priceBuf    = lCols.buffer(5);
    MTL::Buffer* discBuf     = lCols.buffer(6);
    MTL::Buffer* partialPromo = device->newBuffer(numTG * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* partialTotal = device->newBuffer(numTG * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* finalBuf     = device->newBuffer(2 * sizeof(float), MTL::ResourceStorageModeShared);

    double gpuSec = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(bitmapBuf->contents(), 0, bitmapInts * sizeof(uint));

        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

        // Phase 1: Build promo bitmap on GPU
        enc->setComputePipelineState(bitmapPSO);
        enc->setBuffer(pPartKeyBuf, 0, 0);
        enc->setBuffer(pTypeBuf, 0, 1);
        enc->setBuffer(bitmapBuf, 0, 2);
        enc->setBytes(&partSize, sizeof(partSize), 3);
        enc->setBytes(&type_stride, sizeof(type_stride), 4);
        {
            NS::UInteger tgSize = bitmapPSO->maxTotalThreadsPerThreadgroup();
            if (tgSize > 1024) tgSize = 1024;
            uint numGroups = (partSize + (uint)tgSize - 1) / (uint)tgSize;
            enc->dispatchThreadgroups(MTL::Size::Make(numGroups, 1, 1), MTL::Size::Make(tgSize, 1, 1));
        }

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // Phase 2: Filter lineitem and accumulate
        enc->setComputePipelineState(s1PSO);
        enc->setBuffer(partkeyBuf, 0, 0);
        enc->setBuffer(shipdateBuf, 0, 1);
        enc->setBuffer(priceBuf, 0, 2);
        enc->setBuffer(discBuf, 0, 3);
        enc->setBuffer(bitmapBuf, 0, 4);
        enc->setBuffer(partialPromo, 0, 5);
        enc->setBuffer(partialTotal, 0, 6);
        enc->setBytes(&dataSize, sizeof(dataSize), 7);
        enc->setBytes(&start_date, sizeof(start_date), 8);
        enc->setBytes(&end_date, sizeof(end_date), 9);
        NS::UInteger tgSize = s1PSO->maxTotalThreadsPerThreadgroup();
        if (tgSize > 1024) tgSize = 1024;
        enc->dispatchThreadgroups(MTL::Size::Make(numTG, 1, 1), MTL::Size::Make(tgSize, 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);
        const uint numTGs = numTG;
        enc->setComputePipelineState(s2PSO);
        enc->setBuffer(partialPromo, 0, 0);
        enc->setBuffer(partialTotal, 0, 1);
        enc->setBuffer(finalBuf, 0, 2);
        enc->setBytes(&numTGs, sizeof(numTGs), 3);
        enc->dispatchThreads(MTL::Size::Make(1,1,1), MTL::Size::Make(1,1,1));
        enc->endEncoding();

        cb->commit();
        cb->waitUntilCompleted();
        if (iter == 2) gpuSec = cb->GPUEndTime() - cb->GPUStartTime();
    }

    auto cpuPostStart = std::chrono::high_resolution_clock::now();
    float* res = (float*)finalBuf->contents();
    double promo_pct = (res[1] > 0.0f) ? 100.0 * (double)res[0] / (double)res[1] : 0.0;
    auto cpuPostEnd = std::chrono::high_resolution_clock::now();
    double cpuPostMs = std::chrono::duration<double, std::milli>(cpuPostEnd - cpuPostStart).count();

    printf("\nTPC-H Q14 Result: Promo Revenue = %.2f%%\n", promo_pct);
    printf("\nQ14 | %u rows\n", dataSize);
    printTimingSummary(cpuParseMs, gpuSec * 1000.0, cpuPostMs);

    releaseAll(bitmapPSO, s1PSO, s2PSO,
               bitmapBuf, partialPromo, partialTotal, finalBuf);
    // Input buffers owned by pCols/lCols (QueryColumns).
}

// --- SF100 Chunked ---
void runQ14BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q14 Benchmark (SF100 Chunked) ===" << std::endl;

    MappedFile partFile, liFile;
    if (!partFile.open(g_dataset_path + "part.tbl") || !liFile.open(g_dataset_path + "lineitem.tbl")) {
        std::cerr << "Q14 SF100: Cannot open required TBL files" << std::endl;
        return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto partIndex = buildLineIndex(partFile);
    auto liIndex   = buildLineIndex(liFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    size_t partRows = partIndex.size(), liRows = liIndex.size();
    printf("Q14 SF100: part=%zu, lineitem=%zu rows (index %.1f ms)\n", partRows, liRows, indexBuildMs);

    // Load part data and build promo bitmap on GPU
    auto buildT0 = std::chrono::high_resolution_clock::now();
    std::vector<int> p_partkey(partRows);
    parseIntColumnChunk(partFile, partIndex, 0, partRows, 0, p_partkey.data());
    const uint type_stride = 25;
    std::vector<char> p_type(partRows * type_stride);
    parseCharColumnChunkFixed(partFile, partIndex, 0, partRows, 4, type_stride, p_type.data());

    int max_partkey = 0;
    for (auto k : p_partkey) max_partkey = std::max(max_partkey, k);
    uint bitmap_ints = (max_partkey + 31) / 32 + 1;
    auto buildT1 = std::chrono::high_resolution_clock::now();
    double buildMs = std::chrono::duration<double, std::milli>(buildT1 - buildT0).count();

    auto bitmapPSO = createPipeline(device, library, "q14_build_promo_bitmap");
    auto s1PSO = createPipeline(device, library, "q14_chunked_stage1");
    auto s2PSO = createPipeline(device, library, "q14_chunked_stage2");
    if (!bitmapPSO || !s1PSO || !s2PSO) return;

    // Upload part data and build bitmap on GPU
    uint partSizeU = (uint)partRows;
    MTL::Buffer* pPartKeyBuf = device->newBuffer(p_partkey.data(), partRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pTypeBuf    = device->newBuffer(p_type.data(), partRows * type_stride * sizeof(char), MTL::ResourceStorageModeShared);
    MTL::Buffer* bitmapBuf   = device->newBuffer(bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);
    memset(bitmapBuf->contents(), 0, bitmap_ints * sizeof(uint));

    // Dispatch bitmap kernel on GPU
    {
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(bitmapPSO);
        enc->setBuffer(pPartKeyBuf, 0, 0);
        enc->setBuffer(pTypeBuf, 0, 1);
        enc->setBuffer(bitmapBuf, 0, 2);
        enc->setBytes(&partSizeU, sizeof(partSizeU), 3);
        enc->setBytes(&type_stride, sizeof(type_stride), 4);
        NS::UInteger tgSize = bitmapPSO->maxTotalThreadsPerThreadgroup();
        if (tgSize > 1024) tgSize = 1024;
        uint numGroups = (partSizeU + (uint)tgSize - 1) / (uint)tgSize;
        enc->dispatchThreadgroups(MTL::Size::Make(numGroups, 1, 1), MTL::Size::Make(tgSize, 1, 1));
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        double bitmapGpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
        printf("Promo bitmap built on GPU in %.2f ms\n", bitmapGpuMs);
    }
    releaseAll(pPartKeyBuf, pTypeBuf);

    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 16, liRows); // 4 cols × 4 bytes
    const uint num_tg = 2048;
    printf("Chunk size: %zu rows\n", chunkRows);

    const int NUM_SLOTS = 2;
    struct Q14Slot { MTL::Buffer* partkey; MTL::Buffer* shipdate; MTL::Buffer* extprice; MTL::Buffer* discount; };
    Q14Slot slots[NUM_SLOTS];
    for (int s = 0; s < NUM_SLOTS; s++) {
        slots[s].partkey  = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].shipdate = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].extprice = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].discount = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
    }

    MTL::Buffer* partials_promo = device->newBuffer(num_tg * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* partials_total = device->newBuffer(num_tg * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* finalBuf = device->newBuffer(2 * sizeof(float), MTL::ResourceStorageModeShared);

    int start_date = 19950901, end_date = 19951001;
    double globalPromo = 0.0, globalTotal = 0.0;

    auto timing = chunkedStreamLoop(
        commandQueue, slots, NUM_SLOTS, liRows, chunkRows,
        // Parse
        [&](Q14Slot& slot, size_t startRow, size_t rowCount) {
            parseIntColumnChunk(liFile, liIndex, startRow, rowCount, 1, (int*)slot.partkey->contents());
            parseDateColumnChunk(liFile, liIndex, startRow, rowCount, 10, (int*)slot.shipdate->contents());
            parseFloatColumnChunk(liFile, liIndex, startRow, rowCount, 5, (float*)slot.extprice->contents());
            parseFloatColumnChunk(liFile, liIndex, startRow, rowCount, 6, (float*)slot.discount->contents());
        },
        // Dispatch
        [&](Q14Slot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(s1PSO);
            enc->setBuffer(slot.partkey, 0, 0);
            enc->setBuffer(slot.shipdate, 0, 1);
            enc->setBuffer(slot.extprice, 0, 2);
            enc->setBuffer(slot.discount, 0, 3);
            enc->setBuffer(bitmapBuf, 0, 4);
            enc->setBuffer(partials_promo, 0, 5);
            enc->setBuffer(partials_total, 0, 6);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 7);
            enc->setBytes(&start_date, sizeof(start_date), 8);
            enc->setBytes(&end_date, sizeof(end_date), 9);
            NS::UInteger tgSize = s1PSO->maxTotalThreadsPerThreadgroup();
            if (tgSize > 1024) tgSize = 1024;
            enc->dispatchThreadgroups(MTL::Size::Make(num_tg, 1, 1), MTL::Size::Make(tgSize, 1, 1));

            enc->memoryBarrier(MTL::BarrierScopeBuffers);
            enc->setComputePipelineState(s2PSO);
            enc->setBuffer(partials_promo, 0, 0);
            enc->setBuffer(partials_total, 0, 1);
            enc->setBuffer(finalBuf, 0, 2);
            enc->setBytes(&num_tg, sizeof(num_tg), 3);
            enc->dispatchThreads(MTL::Size::Make(1,1,1), MTL::Size::Make(1,1,1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        // Accumulate
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {
            float* res = (float*)finalBuf->contents();
            globalPromo += res[0];
            globalTotal += res[1];
        }
    );

    double promo_pct = (globalTotal > 0.0) ? 100.0 * globalPromo / globalTotal : 0.0;
    double allCpuParseMs = indexBuildMs + buildMs + timing.parseMs;
    printf("TPC-H Q14 Result: Promo Revenue = %.2f%%\n", promo_pct);
    printf("\nSF100 Q14 | %zu chunks | %zu rows\n", timing.chunkCount, liRows);
    printTimingSummary(allCpuParseMs, timing.gpuMs, timing.postMs);

    releaseAll(bitmapPSO, s1PSO, s2PSO, bitmapBuf, partials_promo, partials_total, finalBuf);
    for (int s = 0; s < NUM_SLOTS; s++)
        releaseAll(slots[s].partkey, slots[s].shipdate, slots[s].extprice, slots[s].discount);
}
