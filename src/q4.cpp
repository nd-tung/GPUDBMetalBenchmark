#include "infra.h"

// ===================================================================
// TPC-H Q4 — Order Priority Checking
// ===================================================================

// --- Standard (SF1/SF10) ---
void runQ4Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "--- Running TPC-H Query 4 Benchmark ---" << std::endl;

    auto parseStart = std::chrono::high_resolution_clock::now();
    const std::string sf_path = g_dataset_path;

    // Load lineitem columns for late-delivery bitmap
    auto l_orderkey    = loadIntColumn(sf_path + "lineitem.tbl", 0);
    auto l_commitdate  = loadDateColumn(sf_path + "lineitem.tbl", 11);
    auto l_receiptdate = loadDateColumn(sf_path + "lineitem.tbl", 12);

    // Load orders columns
    auto o_orderkey      = loadIntColumn(sf_path + "orders.tbl", 0);
    auto o_orderdate     = loadDateColumn(sf_path + "orders.tbl", 4);
    auto o_orderpriority = loadCharColumn(sf_path + "orders.tbl", 5);  // first char
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double cpuParseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    uint liSize = (uint)l_orderkey.size();
    uint ordSize = (uint)o_orderkey.size();
    if (liSize == 0 || ordSize == 0) { std::cerr << "Q4: no data loaded" << std::endl; return; }
    std::cout << "Loaded " << liSize << " lineitem, " << ordSize << " orders rows for Q4." << std::endl;

    auto bitmapPSO = createPipeline(device, library, "q4_build_late_bitmap");
    auto s1PSO     = createPipeline(device, library, "q4_count_by_priority_stage1");
    auto s2PSO     = createPipeline(device, library, "q4_final_count_stage2");
    if (!bitmapPSO || !s1PSO || !s2PSO) return;

    // Bitmap size based on max orderkey
    int max_orderkey = 0;
    for (int k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    for (int k : l_orderkey) max_orderkey = std::max(max_orderkey, k);

    const int numTG = 2048;
    int date_start = 19930701, date_end = 19931001;

    MTL::Buffer* liOrderkeyBuf = device->newBuffer(l_orderkey.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* commitBuf     = device->newBuffer(l_commitdate.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* receiptBuf    = device->newBuffer(l_receiptdate.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* lateBitmapBuf = createBitmapBuffer(device, max_orderkey);

    MTL::Buffer* ordKeyBuf  = device->newBuffer(o_orderkey.data(), ordSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* ordDateBuf = device->newBuffer(o_orderdate.data(), ordSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* ordPrioBuf = device->newBuffer(o_orderpriority.data(), ordSize * sizeof(char), MTL::ResourceStorageModeShared);
    MTL::Buffer* partialBuf = device->newBuffer(numTG * 5 * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* finalBuf   = device->newBuffer(5 * sizeof(uint), MTL::ResourceStorageModeShared);

    double gpuSec = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        // Reset late-delivery bitmap
        memset(lateBitmapBuf->contents(), 0, lateBitmapBuf->length());

        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

        // Phase 1: Build late-delivery bitmap from lineitem
        enc->setComputePipelineState(bitmapPSO);
        enc->setBuffer(liOrderkeyBuf, 0, 0);
        enc->setBuffer(commitBuf, 0, 1);
        enc->setBuffer(receiptBuf, 0, 2);
        enc->setBuffer(lateBitmapBuf, 0, 3);
        enc->setBytes(&liSize, sizeof(liSize), 4);
        {
            NS::UInteger tgSize = bitmapPSO->maxTotalThreadsPerThreadgroup();
            if (tgSize > 256) tgSize = 256;
            MTL::Size grid = MTL::Size::Make((liSize + tgSize - 1) / tgSize, 1, 1);
            enc->dispatchThreadgroups(grid, MTL::Size::Make(tgSize, 1, 1));
        }

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // Phase 2: Count orders by priority
        enc->setComputePipelineState(s1PSO);
        enc->setBuffer(ordKeyBuf, 0, 0);
        enc->setBuffer(ordDateBuf, 0, 1);
        enc->setBuffer(ordPrioBuf, 0, 2);
        enc->setBuffer(lateBitmapBuf, 0, 3);
        enc->setBuffer(partialBuf, 0, 4);
        enc->setBytes(&ordSize, sizeof(ordSize), 5);
        enc->setBytes(&date_start, sizeof(date_start), 6);
        enc->setBytes(&date_end, sizeof(date_end), 7);
        {
            NS::UInteger tgSize = s1PSO->maxTotalThreadsPerThreadgroup();
            if (tgSize > 1024) tgSize = 1024;
            enc->dispatchThreadgroups(MTL::Size::Make(numTG, 1, 1), MTL::Size::Make(tgSize, 1, 1));
        }

        enc->memoryBarrier(MTL::BarrierScopeBuffers);
        const uint numTGs = numTG;
        enc->setComputePipelineState(s2PSO);
        enc->setBuffer(partialBuf, 0, 0);
        enc->setBuffer(finalBuf, 0, 1);
        enc->setBytes(&numTGs, sizeof(numTGs), 2);
        enc->dispatchThreads(MTL::Size::Make(1,1,1), MTL::Size::Make(1,1,1));
        enc->endEncoding();

        cb->commit();
        cb->waitUntilCompleted();
        if (iter == 2) gpuSec = cb->GPUEndTime() - cb->GPUStartTime();
    }

    auto cpuPostStart = std::chrono::high_resolution_clock::now();
    uint* res = (uint*)finalBuf->contents();
    auto cpuPostEnd = std::chrono::high_resolution_clock::now();
    double cpuPostMs = std::chrono::duration<double, std::milli>(cpuPostEnd - cpuPostStart).count();

    const char* prio_names[] = {"1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"};
    printf("\nTPC-H Q4 Results:\n");
    printf("+------------------+-------------+\n");
    printf("| o_orderpriority  | order_count |\n");
    printf("+------------------+-------------+\n");
    for (int i = 0; i < 5; i++) {
        printf("| %-16s | %11u |\n", prio_names[i], res[i]);
    }
    printf("+------------------+-------------+\n");

    printf("\nQ4 | %u lineitem, %u orders\n", liSize, ordSize);
    printTimingSummary(cpuParseMs, gpuSec * 1000.0, cpuPostMs);

    releaseAll(bitmapPSO, s1PSO, s2PSO, liOrderkeyBuf, commitBuf, receiptBuf,
               lateBitmapBuf, ordKeyBuf, ordDateBuf, ordPrioBuf, partialBuf, finalBuf);
}

// --- SF100 Chunked ---
void runQ4BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q4 Benchmark (SF100 Chunked) ===" << std::endl;

    MappedFile liFile, ordFile;
    if (!liFile.open(g_dataset_path + "lineitem.tbl") || !ordFile.open(g_dataset_path + "orders.tbl")) {
        std::cerr << "Q4 SF100: Cannot open required TBL files" << std::endl;
        return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto liIndex  = buildLineIndex(liFile);
    auto ordIndex = buildLineIndex(ordFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    size_t liRows = liIndex.size(), ordRows = ordIndex.size();
    printf("Q4 SF100: lineitem=%zu, orders=%zu rows (index %.1f ms)\n", liRows, ordRows, indexBuildMs);

    auto bitmapPSO = createPipeline(device, library, "q4_chunked_build_late_bitmap");
    auto s1PSO     = createPipeline(device, library, "q4_chunked_count_stage1");
    auto s2PSO     = createPipeline(device, library, "q4_chunked_final_stage2");
    if (!bitmapPSO || !s1PSO || !s2PSO) return;

    // Find max orderkey from orders to size bitmap
    auto buildT0 = std::chrono::high_resolution_clock::now();
    std::vector<int> o_orderkey(ordRows);
    parseIntColumnChunk(ordFile, ordIndex, 0, ordRows, 0, o_orderkey.data());
    int max_orderkey = 0;
    for (auto k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    auto buildT1 = std::chrono::high_resolution_clock::now();
    double buildMs = std::chrono::duration<double, std::milli>(buildT1 - buildT0).count();

    uint bitmap_ints = (max_orderkey + 31) / 32 + 1;
    printf("Late bitmap: %u ints (%.1f MB), max_orderkey=%d\n",
           bitmap_ints, bitmap_ints * sizeof(uint) / (1024.0*1024.0), max_orderkey);

    MTL::Buffer* lateBitmapBuf = createFilledBuffer(device, bitmap_ints * sizeof(uint), 0);

    // ===== Phase 1: Stream lineitem to build late-delivery bitmap =====
    size_t liChunkRows = ChunkConfig::adaptiveChunkSize(device, 12, liRows); // 3 int cols × 4 bytes
    printf("Phase 1 lineitem chunk: %zu rows\n", liChunkRows);

    const int NUM_SLOTS = 2;
    struct LISlot { MTL::Buffer* orderkey; MTL::Buffer* commitdate; MTL::Buffer* receiptdate; };
    LISlot liSlots[NUM_SLOTS];
    for (int s = 0; s < NUM_SLOTS; s++) {
        liSlots[s].orderkey    = device->newBuffer(liChunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        liSlots[s].commitdate  = device->newBuffer(liChunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        liSlots[s].receiptdate = device->newBuffer(liChunkRows * sizeof(int), MTL::ResourceStorageModeShared);
    }

    auto phase1Timing = chunkedStreamLoop(
        commandQueue, liSlots, NUM_SLOTS, liRows, liChunkRows,
        // Parse
        [&](LISlot& slot, size_t startRow, size_t rowCount) {
            parseIntColumnChunk(liFile, liIndex, startRow, rowCount, 0, (int*)slot.orderkey->contents());
            parseDateColumnChunk(liFile, liIndex, startRow, rowCount, 11, (int*)slot.commitdate->contents());
            parseDateColumnChunk(liFile, liIndex, startRow, rowCount, 12, (int*)slot.receiptdate->contents());
        },
        // Dispatch
        [&](LISlot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(bitmapPSO);
            enc->setBuffer(slot.orderkey, 0, 0);
            enc->setBuffer(slot.commitdate, 0, 1);
            enc->setBuffer(slot.receiptdate, 0, 2);
            enc->setBuffer(lateBitmapBuf, 0, 3);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 4);
            NS::UInteger tgSize = bitmapPSO->maxTotalThreadsPerThreadgroup();
            if (tgSize > 256) tgSize = 256;
            enc->dispatchThreadgroups(MTL::Size::Make((chunkSize + tgSize - 1) / tgSize, 1, 1),
                                      MTL::Size::Make(tgSize, 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        // No accumulate for bitmap build
        [&]([[maybe_unused]] uint cs, [[maybe_unused]] size_t cn) {}
    );

    // Free lineitem chunk buffers
    for (int s = 0; s < NUM_SLOTS; s++)
        releaseAll(liSlots[s].orderkey, liSlots[s].commitdate, liSlots[s].receiptdate);

    printf("Phase 1 done: bitmap built in %.1f ms parse + %.1f ms GPU\n",
           phase1Timing.parseMs, phase1Timing.gpuMs);

    // ===== Phase 2: Load orders fully, scan with date filter + bitmap probe =====
    auto ordParseT0 = std::chrono::high_resolution_clock::now();
    std::vector<int> o_orderdate(ordRows);
    std::vector<char> o_orderpriority(ordRows);
    parseDateColumnChunk(ordFile, ordIndex, 0, ordRows, 4, o_orderdate.data());
    parseCharColumnChunk(ordFile, ordIndex, 0, ordRows, 5, o_orderpriority.data());
    auto ordParseT1 = std::chrono::high_resolution_clock::now();
    double ordParseMs = std::chrono::duration<double, std::milli>(ordParseT1 - ordParseT0).count();

    const uint num_tg = 2048;
    int date_start = 19930701, date_end = 19931001;

    MTL::Buffer* ordKeyBuf  = device->newBuffer(o_orderkey.data(), ordRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* ordDateBuf = device->newBuffer(o_orderdate.data(), ordRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* ordPrioBuf = device->newBuffer(o_orderpriority.data(), ordRows * sizeof(char), MTL::ResourceStorageModeShared);
    MTL::Buffer* partialBuf = device->newBuffer(num_tg * 5 * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* finalBuf   = device->newBuffer(5 * sizeof(uint), MTL::ResourceStorageModeShared);

    uint ordSizeU = (uint)ordRows;
    {
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

        enc->setComputePipelineState(s1PSO);
        enc->setBuffer(ordKeyBuf, 0, 0);
        enc->setBuffer(ordDateBuf, 0, 1);
        enc->setBuffer(ordPrioBuf, 0, 2);
        enc->setBuffer(lateBitmapBuf, 0, 3);
        enc->setBuffer(partialBuf, 0, 4);
        enc->setBytes(&ordSizeU, sizeof(ordSizeU), 5);
        enc->setBytes(&date_start, sizeof(date_start), 6);
        enc->setBytes(&date_end, sizeof(date_end), 7);
        NS::UInteger tgSize = s1PSO->maxTotalThreadsPerThreadgroup();
        if (tgSize > 1024) tgSize = 1024;
        enc->dispatchThreadgroups(MTL::Size::Make(num_tg, 1, 1), MTL::Size::Make(tgSize, 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);
        enc->setComputePipelineState(s2PSO);
        enc->setBuffer(partialBuf, 0, 0);
        enc->setBuffer(finalBuf, 0, 1);
        enc->setBytes(&num_tg, sizeof(num_tg), 2);
        enc->dispatchThreads(MTL::Size::Make(1,1,1), MTL::Size::Make(1,1,1));
        enc->endEncoding();

        cb->commit();
        cb->waitUntilCompleted();
        double phase2GpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
        printf("Phase 2 done: orders scan GPU %.1f ms\n", phase2GpuMs);

        uint* res = (uint*)finalBuf->contents();
        const char* prio_names[] = {"1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"};
        printf("\nTPC-H Q4 Results:\n");
        printf("+------------------+-------------+\n");
        printf("| o_orderpriority  | order_count |\n");
        printf("+------------------+-------------+\n");
        for (int i = 0; i < 5; i++) printf("| %-16s | %11u |\n", prio_names[i], res[i]);
        printf("+------------------+-------------+\n");

        double allCpuParseMs = indexBuildMs + buildMs + phase1Timing.parseMs + ordParseMs;
        double allGpuMs = phase1Timing.gpuMs + phase2GpuMs;
        printf("\nSF100 Q4 | lineitem=%zu, orders=%zu\n", liRows, ordRows);
        printTimingSummary(allCpuParseMs, allGpuMs, 0.0);
    }

    releaseAll(bitmapPSO, s1PSO, s2PSO, lateBitmapBuf, ordKeyBuf, ordDateBuf, ordPrioBuf,
               partialBuf, finalBuf);
}
