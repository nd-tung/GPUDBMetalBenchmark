#include "infra.h"

// ===================================================================
// TPC-H Q12 — Shipping Modes and Order Priority
// ===================================================================

// --- Standard (SF1/SF10) ---
void runQ12Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "--- Running TPC-H Query 12 Benchmark ---" << std::endl;

    auto parseStart = std::chrono::high_resolution_clock::now();
    const std::string sf_path = g_dataset_path;

    // Load orders columns for priority bitmap (pure I/O)
    auto o_orderkey      = loadIntColumn(sf_path + "orders.tbl", 0);
    auto o_orderpriority = loadCharColumn(sf_path + "orders.tbl", 5);

    // Load lineitem columns
    auto l_orderkey    = loadIntColumn(sf_path + "lineitem.tbl", 0);
    auto l_shipmode    = loadCharColumn(sf_path + "lineitem.tbl", 14);  // first char
    auto l_shipdate    = loadDateColumn(sf_path + "lineitem.tbl", 10);
    auto l_commitdate  = loadDateColumn(sf_path + "lineitem.tbl", 11);
    auto l_receiptdate = loadDateColumn(sf_path + "lineitem.tbl", 12);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double cpuParseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    uint dataSize = (uint)l_orderkey.size();
    uint ordSize = (uint)o_orderkey.size();
    if (dataSize == 0) { std::cerr << "Q12: no data loaded" << std::endl; return; }
    std::cout << "Loaded " << dataSize << " lineitem rows for Q12." << std::endl;

    auto bitmapPSO = createPipeline(device, library, "q12_build_priority_bitmap");
    auto s1PSO = createPipeline(device, library, "q12_filter_and_count_stage1");
    auto s2PSO = createPipeline(device, library, "q12_final_count_stage2");
    if (!bitmapPSO || !s1PSO || !s2PSO) return;

    int max_orderkey = 0;
    for (int k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    uint bitmapInts = (max_orderkey + 31) / 32 + 1;

    const int numTG = 2048;
    int receipt_start = 19940101, receipt_end = 19950101;

    MTL::Buffer* ordKeyBuf     = device->newBuffer(o_orderkey.data(), ordSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* ordPrioBuf    = device->newBuffer(o_orderpriority.data(), ordSize * sizeof(char), MTL::ResourceStorageModeShared);
    MTL::Buffer* bitmapBuf     = device->newBuffer(bitmapInts * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* orderkeyBuf   = device->newBuffer(l_orderkey.data(), dataSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* shipmodeBuf   = device->newBuffer(l_shipmode.data(), dataSize * sizeof(char), MTL::ResourceStorageModeShared);
    MTL::Buffer* shipdateBuf   = device->newBuffer(l_shipdate.data(), dataSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* commitdateBuf = device->newBuffer(l_commitdate.data(), dataSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* receiptBuf    = device->newBuffer(l_receiptdate.data(), dataSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* partialBuf    = device->newBuffer(numTG * 4 * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* finalBuf      = device->newBuffer(4 * sizeof(uint), MTL::ResourceStorageModeShared);

    double gpuSec = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(bitmapBuf->contents(), 0, bitmapInts * sizeof(uint));

        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

        // Phase 1: Build priority bitmap on GPU
        enc->setComputePipelineState(bitmapPSO);
        enc->setBuffer(ordKeyBuf, 0, 0);
        enc->setBuffer(ordPrioBuf, 0, 1);
        enc->setBuffer(bitmapBuf, 0, 2);
        enc->setBytes(&ordSize, sizeof(ordSize), 3);
        {
            NS::UInteger tgSize = bitmapPSO->maxTotalThreadsPerThreadgroup();
            if (tgSize > 1024) tgSize = 1024;
            uint numGroups = (ordSize + (uint)tgSize - 1) / (uint)tgSize;
            enc->dispatchThreadgroups(MTL::Size::Make(numGroups, 1, 1), MTL::Size::Make(tgSize, 1, 1));
        }

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // Phase 2: Filter lineitem and count
        enc->setComputePipelineState(s1PSO);
        enc->setBuffer(orderkeyBuf, 0, 0);
        enc->setBuffer(shipmodeBuf, 0, 1);
        enc->setBuffer(shipdateBuf, 0, 2);
        enc->setBuffer(commitdateBuf, 0, 3);
        enc->setBuffer(receiptBuf, 0, 4);
        enc->setBuffer(bitmapBuf, 0, 5);
        enc->setBuffer(partialBuf, 0, 6);
        enc->setBytes(&dataSize, sizeof(dataSize), 7);
        enc->setBytes(&receipt_start, sizeof(receipt_start), 8);
        enc->setBytes(&receipt_end, sizeof(receipt_end), 9);
        NS::UInteger tgSize = s1PSO->maxTotalThreadsPerThreadgroup();
        if (tgSize > 1024) tgSize = 1024;
        enc->dispatchThreadgroups(MTL::Size::Make(numTG, 1, 1), MTL::Size::Make(tgSize, 1, 1));

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
    // bins: 0=MAIL-HIGH, 1=MAIL-LOW, 2=SHIP-HIGH, 3=SHIP-LOW
    auto cpuPostEnd = std::chrono::high_resolution_clock::now();
    double cpuPostMs = std::chrono::duration<double, std::milli>(cpuPostEnd - cpuPostStart).count();

    printf("\nTPC-H Q12 Results:\n");
    printf("+----------+------------------+-----------------+\n");
    printf("| shipmode | high_line_count  | low_line_count  |\n");
    printf("+----------+------------------+-----------------+\n");
    printf("| MAIL     | %16u | %15u |\n", res[0], res[1]);
    printf("| SHIP     | %16u | %15u |\n", res[2], res[3]);
    printf("+----------+------------------+-----------------+\n");

    printf("\nQ12 | %u rows\n", dataSize);
    printTimingSummary(cpuParseMs, gpuSec * 1000.0, cpuPostMs);

    releaseAll(bitmapPSO, s1PSO, s2PSO, ordKeyBuf, ordPrioBuf, orderkeyBuf, shipmodeBuf, shipdateBuf,
               commitdateBuf, receiptBuf, bitmapBuf, partialBuf, finalBuf);
}

// --- SF100 Chunked ---
void runQ12BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q12 Benchmark (SF100 Chunked) ===" << std::endl;

    MappedFile ordFile, liFile;
    if (!ordFile.open(g_dataset_path + "orders.tbl") || !liFile.open(g_dataset_path + "lineitem.tbl")) {
        std::cerr << "Q12 SF100: Cannot open required TBL files" << std::endl;
        return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto ordIndex = buildLineIndex(ordFile);
    auto liIndex  = buildLineIndex(liFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    size_t ordRows = ordIndex.size(), liRows = liIndex.size();
    printf("Q12 SF100: orders=%zu, lineitem=%zu rows (index %.1f ms)\n", ordRows, liRows, indexBuildMs);

    // Build priority bitmap on GPU from orders
    auto buildT0 = std::chrono::high_resolution_clock::now();
    std::vector<int> o_orderkey(ordRows);
    std::vector<char> o_orderpriority(ordRows);
    parseIntColumnChunk(ordFile, ordIndex, 0, ordRows, 0, o_orderkey.data());
    parseCharColumnChunk(ordFile, ordIndex, 0, ordRows, 5, o_orderpriority.data());

    int max_orderkey = 0;
    for (auto k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    uint bitmap_ints = (max_orderkey + 31) / 32 + 1;
    auto buildT1 = std::chrono::high_resolution_clock::now();
    double buildMs = std::chrono::duration<double, std::milli>(buildT1 - buildT0).count();
    printf("Priority bitmap: %u ints (%.1f MB), orders parsed in %.1f ms\n",
           bitmap_ints, bitmap_ints * sizeof(uint) / (1024.0*1024.0), buildMs);

    auto bitmapPSO = createPipeline(device, library, "q12_build_priority_bitmap");
    auto s1PSO = createPipeline(device, library, "q12_chunked_stage1");
    auto s2PSO = createPipeline(device, library, "q12_chunked_stage2");
    if (!bitmapPSO || !s1PSO || !s2PSO) return;

    // Upload orders data and build bitmap on GPU
    uint ordSizeU = (uint)ordRows;
    MTL::Buffer* ordKeyBuf  = device->newBuffer(o_orderkey.data(), ordRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* ordPrioBuf = device->newBuffer(o_orderpriority.data(), ordRows * sizeof(char), MTL::ResourceStorageModeShared);
    MTL::Buffer* bitmapBuf  = device->newBuffer(bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);
    memset(bitmapBuf->contents(), 0, bitmap_ints * sizeof(uint));

    // Dispatch bitmap kernel on GPU
    {
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(bitmapPSO);
        enc->setBuffer(ordKeyBuf, 0, 0);
        enc->setBuffer(ordPrioBuf, 0, 1);
        enc->setBuffer(bitmapBuf, 0, 2);
        enc->setBytes(&ordSizeU, sizeof(ordSizeU), 3);
        NS::UInteger tgSize = bitmapPSO->maxTotalThreadsPerThreadgroup();
        if (tgSize > 1024) tgSize = 1024;
        uint numGroups = (ordSizeU + (uint)tgSize - 1) / (uint)tgSize;
        enc->dispatchThreadgroups(MTL::Size::Make(numGroups, 1, 1), MTL::Size::Make(tgSize, 1, 1));
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        double bitmapGpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
        printf("Priority bitmap built on GPU in %.2f ms\n", bitmapGpuMs);
    }
    releaseAll(ordKeyBuf, ordPrioBuf);

    // lineitem: orderkey(4) + shipmode(1) + shipdate(4) + commitdate(4) + receiptdate(4) = 17 bytes/row
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 17, liRows);
    const uint num_tg = 2048;
    printf("Chunk size: %zu rows\n", chunkRows);

    const int NUM_SLOTS = 2;
    struct Q12Slot {
        MTL::Buffer* orderkey; MTL::Buffer* shipmode; MTL::Buffer* shipdate;
        MTL::Buffer* commitdate; MTL::Buffer* receiptdate;
    };
    Q12Slot slots[NUM_SLOTS];
    for (int s = 0; s < NUM_SLOTS; s++) {
        slots[s].orderkey    = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].shipmode    = device->newBuffer(chunkRows * sizeof(char), MTL::ResourceStorageModeShared);
        slots[s].shipdate    = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].commitdate  = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].receiptdate = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
    }

    MTL::Buffer* partialBuf = device->newBuffer(num_tg * 4 * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* finalBuf   = device->newBuffer(4 * sizeof(uint), MTL::ResourceStorageModeShared);

    int receipt_start = 19940101, receipt_end = 19950101;
    uint globalCounts[4] = {0, 0, 0, 0};

    auto timing = chunkedStreamLoop(
        commandQueue, slots, NUM_SLOTS, liRows, chunkRows,
        // Parse
        [&](Q12Slot& slot, size_t startRow, size_t rowCount) {
            parseIntColumnChunk(liFile, liIndex, startRow, rowCount, 0, (int*)slot.orderkey->contents());
            parseCharColumnChunk(liFile, liIndex, startRow, rowCount, 14, (char*)slot.shipmode->contents());
            parseDateColumnChunk(liFile, liIndex, startRow, rowCount, 10, (int*)slot.shipdate->contents());
            parseDateColumnChunk(liFile, liIndex, startRow, rowCount, 11, (int*)slot.commitdate->contents());
            parseDateColumnChunk(liFile, liIndex, startRow, rowCount, 12, (int*)slot.receiptdate->contents());
        },
        // Dispatch
        [&](Q12Slot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(s1PSO);
            enc->setBuffer(slot.orderkey, 0, 0);
            enc->setBuffer(slot.shipmode, 0, 1);
            enc->setBuffer(slot.shipdate, 0, 2);
            enc->setBuffer(slot.commitdate, 0, 3);
            enc->setBuffer(slot.receiptdate, 0, 4);
            enc->setBuffer(bitmapBuf, 0, 5);
            enc->setBuffer(partialBuf, 0, 6);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 7);
            enc->setBytes(&receipt_start, sizeof(receipt_start), 8);
            enc->setBytes(&receipt_end, sizeof(receipt_end), 9);
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
            cmdBuf->commit();
        },
        // Accumulate
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {
            uint* res = (uint*)finalBuf->contents();
            for (int b = 0; b < 4; b++) globalCounts[b] += res[b];
        }
    );

    double allCpuParseMs = indexBuildMs + buildMs + timing.parseMs;
    printf("\nTPC-H Q12 Results:\n");
    printf("+----------+------------------+-----------------+\n");
    printf("| shipmode | high_line_count  | low_line_count  |\n");
    printf("+----------+------------------+-----------------+\n");
    printf("| MAIL     | %16u | %15u |\n", globalCounts[0], globalCounts[1]);
    printf("| SHIP     | %16u | %15u |\n", globalCounts[2], globalCounts[3]);
    printf("+----------+------------------+-----------------+\n");
    printf("\nSF100 Q12 | %zu chunks | %zu rows\n", timing.chunkCount, liRows);
    printTimingSummary(allCpuParseMs, timing.gpuMs, timing.postMs);

    releaseAll(bitmapPSO, s1PSO, s2PSO, bitmapBuf, partialBuf, finalBuf);
    for (int s = 0; s < NUM_SLOTS; s++)
        releaseAll(slots[s].orderkey, slots[s].shipmode, slots[s].shipdate,
                   slots[s].commitdate, slots[s].receiptdate);
}
