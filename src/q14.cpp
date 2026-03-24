#include "infra.h"

// ===================================================================
// TPC-H Q14 — Promotion Effect
// ===================================================================

// Helper: build promo bitmap on CPU from part table
static std::pair<std::vector<uint>, int> buildPromoBitmap(
    const std::string& partPath) {
    // Load p_partkey (col 0) and p_type first 5 chars (col 4)
    auto p_partkey = loadIntColumn(partPath, 0);
    auto p_type = loadCharColumn(partPath, 4, 5);  // first 5 chars

    int max_partkey = 0;
    for (int k : p_partkey) max_partkey = std::max(max_partkey, k);

    uint bitmap_ints = (max_partkey + 31) / 32 + 1;
    std::vector<uint> bitmap(bitmap_ints, 0);

    for (size_t i = 0; i < p_partkey.size(); i++) {
        const char* t = &p_type[i * 5];
        if (t[0] == 'P' && t[1] == 'R' && t[2] == 'O' && t[3] == 'M' && t[4] == 'O') {
            int k = p_partkey[i];
            bitmap[k / 32] |= (1u << (k % 32));
        }
    }
    return {std::move(bitmap), max_partkey};
}

// --- Standard (SF1/SF10) ---
void runQ14Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "--- Running TPC-H Query 14 Benchmark ---" << std::endl;

    auto q14ParseStart = std::chrono::high_resolution_clock::now();
    const std::string sf_path = g_dataset_path;

    // Build promo bitmap from part table
    auto [promo_bitmap, max_partkey] = buildPromoBitmap(sf_path + "part.tbl");

    // Load lineitem columns
    auto l_partkey       = loadIntColumn(sf_path + "lineitem.tbl", 1);
    auto l_shipdate      = loadDateColumn(sf_path + "lineitem.tbl", 10);
    auto l_extendedprice = loadFloatColumn(sf_path + "lineitem.tbl", 5);
    auto l_discount      = loadFloatColumn(sf_path + "lineitem.tbl", 6);
    auto q14ParseEnd = std::chrono::high_resolution_clock::now();
    double cpuParseMs = std::chrono::duration<double, std::milli>(q14ParseEnd - q14ParseStart).count();

    uint dataSize = (uint)l_partkey.size();
    if (dataSize == 0) { std::cerr << "Q14: no data loaded" << std::endl; return; }
    std::cout << "Loaded " << dataSize << " lineitem rows for Q14." << std::endl;

    auto s1PSO = createPipeline(device, library, "q14_filter_and_sum_stage1");
    auto s2PSO = createPipeline(device, library, "q14_final_sum_stage2");
    if (!s1PSO || !s2PSO) return;

    const int numTG = 2048;
    int start_date = 19950901, end_date = 19951001;

    MTL::Buffer* partkeyBuf  = device->newBuffer(l_partkey.data(), dataSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* shipdateBuf = device->newBuffer(l_shipdate.data(), dataSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* priceBuf    = device->newBuffer(l_extendedprice.data(), dataSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* discBuf     = device->newBuffer(l_discount.data(), dataSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* bitmapBuf   = device->newBuffer(promo_bitmap.data(), promo_bitmap.size() * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* partialPromo = device->newBuffer(numTG * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* partialTotal = device->newBuffer(numTG * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* finalBuf     = device->newBuffer(2 * sizeof(float), MTL::ResourceStorageModeShared);

    double gpuSec = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

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

    releaseAll(s1PSO, s2PSO, partkeyBuf, shipdateBuf, priceBuf, discBuf,
               bitmapBuf, partialPromo, partialTotal, finalBuf);
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

    // Build promo bitmap on CPU from part table
    auto buildT0 = std::chrono::high_resolution_clock::now();
    std::vector<int> p_partkey(partRows);
    parseIntColumnChunk(partFile, partIndex, 0, partRows, 0, p_partkey.data());
    std::vector<char> p_type(partRows * 5);
    parseCharColumnChunkFixed(partFile, partIndex, 0, partRows, 4, 5, p_type.data());

    int max_partkey = 0;
    for (auto k : p_partkey) max_partkey = std::max(max_partkey, k);
    uint bitmap_ints = (max_partkey + 31) / 32 + 1;
    std::vector<uint> promo_bitmap(bitmap_ints, 0);
    for (size_t i = 0; i < partRows; i++) {
        const char* t = &p_type[i * 5];
        if (t[0] == 'P' && t[1] == 'R' && t[2] == 'O' && t[3] == 'M' && t[4] == 'O') {
            int k = p_partkey[i];
            promo_bitmap[k / 32] |= (1u << (k % 32));
        }
    }
    auto buildT1 = std::chrono::high_resolution_clock::now();
    double buildMs = std::chrono::duration<double, std::milli>(buildT1 - buildT0).count();

    auto s1PSO = createPipeline(device, library, "q14_chunked_stage1");
    auto s2PSO = createPipeline(device, library, "q14_chunked_stage2");
    if (!s1PSO || !s2PSO) return;

    // Upload promo bitmap (persists across chunks)
    MTL::Buffer* bitmapBuf = device->newBuffer(promo_bitmap.data(), bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);

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

    releaseAll(s1PSO, s2PSO, bitmapBuf, partials_promo, partials_total, finalBuf);
    for (int s = 0; s < NUM_SLOTS; s++)
        releaseAll(slots[s].partkey, slots[s].shipdate, slots[s].extprice, slots[s].discount);
}
