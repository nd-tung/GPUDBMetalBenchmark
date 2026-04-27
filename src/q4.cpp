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
    auto lCols = loadQueryColumns(device, sf_path + "lineitem.tbl", {{0, ColType::INT}, {11, ColType::DATE}, {12, ColType::DATE}});

    // Load orders columns
    auto oCols = loadQueryColumns(device, sf_path + "orders.tbl", {{0, ColType::INT}, {4, ColType::DATE}, {5, ColType::CHAR1}});
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double cpuParseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    uint liSize = (uint)lCols.rows();
    uint ordSize = (uint)oCols.rows();
    if (liSize == 0 || ordSize == 0) { std::cerr << "Q4: no data loaded" << std::endl; return; }
    std::cout << "Loaded " << liSize << " lineitem, " << ordSize << " orders rows for Q4." << std::endl;

    auto bitmapPSO = createPipeline(device, library, "q4_build_late_bitmap");
    auto s1PSO     = createPipeline(device, library, "q4_count_by_priority_stage1");
    auto s2PSO     = createPipeline(device, library, "q4_final_count_stage2");
    if (!bitmapPSO || !s1PSO || !s2PSO) return;

    // Bitmap size based on max orderkey
    int max_orderkey = 0;
    for (int k : oCols.intSpan(0)) max_orderkey = std::max(max_orderkey, k);
    for (int k : lCols.intSpan(0)) max_orderkey = std::max(max_orderkey, k);
    uint bitmapInts = (max_orderkey + 31) / 32 + 1;

    // Cap dispatch by input size; pick larger of the two scan inputs (lineitem/orders).
    const int numTG = std::min(2048, (int)((std::max(liSize, ordSize) + 1023u) / 1024u));
    int date_start = 19930701, date_end = 19931001;

    MTL::Buffer* liOrderkeyBuf  = lCols.buffer(0);
    MTL::Buffer* liCommitBuf    = lCols.buffer(11);
    MTL::Buffer* liReceiptBuf   = lCols.buffer(12);
    MTL::Buffer* lateBitmapBuf  = device->newBuffer(bitmapInts * sizeof(uint), MTL::ResourceStorageModeShared);
    memset(lateBitmapBuf->contents(), 0, bitmapInts * sizeof(uint));
    MTL::Buffer* ordKeyBuf  = oCols.buffer(0);
    MTL::Buffer* ordDateBuf = oCols.buffer(4);
    MTL::Buffer* ordPrioBuf = oCols.buffer(5);
    MTL::Buffer* partialBuf = device->newBuffer(numTG * 5 * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* finalBuf   = device->newBuffer(5 * sizeof(uint), MTL::ResourceStorageModeShared);

    double gpuSec = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        // Reset bitmap for each iteration
        memset(lateBitmapBuf->contents(), 0, bitmapInts * sizeof(uint));

        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

        // Phase 1: Build late-delivery bitmap on GPU
        enc->setComputePipelineState(bitmapPSO);
        enc->setBuffer(liOrderkeyBuf, 0, 0);
        enc->setBuffer(liCommitBuf, 0, 1);
        enc->setBuffer(liReceiptBuf, 0, 2);
        enc->setBuffer(lateBitmapBuf, 0, 3);
        enc->setBytes(&liSize, sizeof(liSize), 4);
        {
            NS::UInteger tgSize = bitmapPSO->maxTotalThreadsPerThreadgroup();
            if (tgSize > 1024) tgSize = 1024;
            uint numGroups = (liSize + (uint)tgSize - 1) / (uint)tgSize;
            enc->dispatchThreadgroups(MTL::Size::Make(numGroups, 1, 1), MTL::Size::Make(tgSize, 1, 1));
        }

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // Phase 2: Count orders by priority (bitmap built on GPU)
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

    releaseAll(bitmapPSO, s1PSO, s2PSO,
               lateBitmapBuf, partialBuf, finalBuf);
    // Input buffers owned by lCols/oCols (QueryColumns).
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

    uint bitmap_ints = (max_orderkey + 31) / 32 + 1;
    printf("Late bitmap: %u ints (%.1f MB), max_orderkey=%d\n",
           bitmap_ints, bitmap_ints * sizeof(uint) / (1024.0*1024.0), max_orderkey);
    auto buildT1 = std::chrono::high_resolution_clock::now();
    double buildMs = std::chrono::duration<double, std::milli>(buildT1 - buildT0).count();

    MTL::Buffer* lateBitmapBuf = device->newBuffer(bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);
    memset(lateBitmapBuf->contents(), 0, bitmap_ints * sizeof(uint));

    // ===== Phase 1: Build late-delivery bitmap on GPU via chunked streaming =====
    size_t liChunkRows = ChunkConfig::adaptiveChunkSize(device, 12, liRows); // orderkey(4)+commit(4)+receipt(4)
    const int LI_NUM_SLOTS = 2;
    struct LISlot { MTL::Buffer* orderkey; MTL::Buffer* commitdate; MTL::Buffer* receiptdate; };
    LISlot liSlots[LI_NUM_SLOTS];
    for (int s = 0; s < LI_NUM_SLOTS; s++) {
        liSlots[s].orderkey    = device->newBuffer(liChunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        liSlots[s].commitdate  = device->newBuffer(liChunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        liSlots[s].receiptdate = device->newBuffer(liChunkRows * sizeof(int), MTL::ResourceStorageModeShared);
    }

    auto bitmapTiming = chunkedStreamLoop(
        commandQueue, liSlots, LI_NUM_SLOTS, liRows, liChunkRows,
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
            if (tgSize > 1024) tgSize = 1024;
            uint numGroups = (chunkSize + (uint)tgSize - 1) / (uint)tgSize;
            enc->dispatchThreadgroups(MTL::Size::Make(numGroups, 1, 1), MTL::Size::Make(tgSize, 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        // No accumulate needed
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {}
    );

    printf("Phase 1 done: bitmap built on GPU in %.1f ms (parse %.1f ms + gpu %.1f ms)\n",
           bitmapTiming.parseMs + bitmapTiming.gpuMs, bitmapTiming.parseMs, bitmapTiming.gpuMs);

    for (int s = 0; s < LI_NUM_SLOTS; s++)
        releaseAll(liSlots[s].orderkey, liSlots[s].commitdate, liSlots[s].receiptdate);

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

        double allCpuParseMs = indexBuildMs + buildMs + bitmapTiming.parseMs + ordParseMs;
        double allGpuMs = bitmapTiming.gpuMs + phase2GpuMs;
        printf("\nSF100 Q4 | lineitem=%zu, orders=%zu\n", liRows, ordRows);
        printTimingSummary(allCpuParseMs, allGpuMs, 0.0);
    }

    releaseAll(bitmapPSO, s1PSO, s2PSO, lateBitmapBuf, ordKeyBuf, ordDateBuf, ordPrioBuf,
               partialBuf, finalBuf);
}
