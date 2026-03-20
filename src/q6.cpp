#include "infra.h"

// ===================================================================
// TPC-H Q6 — Forecasting Revenue Change
// ===================================================================

// --- Standard (SF1/SF10) ---
void runQ6Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "--- Running TPC-H Query 6 Benchmark ---" << std::endl;
    
    // Load required columns from lineitem table
    auto q6ParseStart = std::chrono::high_resolution_clock::now();
    std::vector<int> l_shipdate = loadDateColumn(g_dataset_path + "lineitem.tbl", 10);    // Column 10: l_shipdate
    std::vector<float> l_discount = loadFloatColumn(g_dataset_path + "lineitem.tbl", 6);  // Column 6: l_discount
    std::vector<float> l_quantity = loadFloatColumn(g_dataset_path + "lineitem.tbl", 4);  // Column 4: l_quantity
    std::vector<float> l_extendedprice = loadFloatColumn(g_dataset_path + "lineitem.tbl", 5); // Column 5: l_extendedprice
    auto q6ParseEnd = std::chrono::high_resolution_clock::now();
    double q6CpuParseMs = std::chrono::duration<double, std::milli>(q6ParseEnd - q6ParseStart).count();

    if (l_shipdate.empty() || l_discount.empty() || l_quantity.empty() || l_extendedprice.empty()) {
        std::cerr << "Error: Could not load required columns for Q6 benchmark" << std::endl;
        return;
    }

    uint dataSize = (uint)l_shipdate.size();
    std::cout << "Loaded " << dataSize << " rows for TPC-H Query 6." << std::endl;

    // Query parameters
    int start_date = 19940101;   // 1994-01-01
    int end_date = 19950101;     // 1995-01-01
    float min_discount = 0.05f;  // 5%
    float max_discount = 0.07f;  // 7%
    float max_quantity = 24.0f;

    auto stage1Pipeline = createPipeline(device, library, "q6_filter_and_sum_stage1");
    auto stage2Pipeline = createPipeline(device, library, "q6_final_sum_stage2");
    if (!stage1Pipeline || !stage2Pipeline) return;

    // Create GPU buffers
    const int numThreadgroups = 2048;
    MTL::Buffer* shipdateBuffer = device->newBuffer(l_shipdate.data(), dataSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* discountBuffer = device->newBuffer(l_discount.data(), dataSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* quantityBuffer = device->newBuffer(l_quantity.data(), dataSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* extendedpriceBuffer = device->newBuffer(l_extendedprice.data(), dataSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* partialRevenuesBuffer = device->newBuffer(numThreadgroups * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* finalRevenueBuffer = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    // Execute GPU kernels (2 warmup + 1 measured)
    double q6_gpu_s = 0.0;
    
    for(int iter = 0; iter < 3; ++iter) {
        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = commandBuffer->computeCommandEncoder();
        
        // Stage 1: Filter and compute partial revenue sums
        enc->setComputePipelineState(stage1Pipeline);
        enc->setBuffer(shipdateBuffer, 0, 0);
        enc->setBuffer(discountBuffer, 0, 1);
        enc->setBuffer(quantityBuffer, 0, 2);
        enc->setBuffer(extendedpriceBuffer, 0, 3);
        enc->setBuffer(partialRevenuesBuffer, 0, 4);
        enc->setBytes(&dataSize, sizeof(dataSize), 5);
        enc->setBytes(&start_date, sizeof(start_date), 6);
        enc->setBytes(&end_date, sizeof(end_date), 7);
        enc->setBytes(&min_discount, sizeof(min_discount), 8);
        enc->setBytes(&max_discount, sizeof(max_discount), 9);
        enc->setBytes(&max_quantity, sizeof(max_quantity), 10);

        NS::UInteger stage1ThreadGroupSize = stage1Pipeline->maxTotalThreadsPerThreadgroup();
        if (stage1ThreadGroupSize > 1024) stage1ThreadGroupSize = 1024; // cap to match shared_memory[1024] in kernel
        MTL::Size stage1GridSize = MTL::Size::Make(numThreadgroups, 1, 1);
        MTL::Size stage1GroupSize = MTL::Size::Make(stage1ThreadGroupSize, 1, 1);
        enc->dispatchThreadgroups(stage1GridSize, stage1GroupSize);

        enc->memoryBarrier(MTL::BarrierScopeBuffers);
        // Stage 2: Final sum reduction on the same encoder
        const uint numTGs = numThreadgroups;
        enc->setComputePipelineState(stage2Pipeline);
        enc->setBuffer(partialRevenuesBuffer, 0, 0);
        enc->setBuffer(finalRevenueBuffer, 0, 1);
        enc->setBytes(&numTGs, sizeof(numTGs), 2);
        MTL::Size stage2GridSize = MTL::Size::Make(1, 1, 1);
        MTL::Size stage2GroupSize = MTL::Size::Make(1, 1, 1);
        enc->dispatchThreads(stage2GridSize, stage2GroupSize);
        enc->endEncoding();

        // Execute and measure time
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        
        if (iter == 2) {
             q6_gpu_s = commandBuffer->GPUEndTime() - commandBuffer->GPUStartTime();
        }
    }

    // CPU post (minimal) timing: fetching result
    auto q6_cpu_post_start = std::chrono::high_resolution_clock::now();

    // Get result
    float* resultData = (float*)finalRevenueBuffer->contents();
    float totalRevenue = resultData[0];

    auto q6_cpu_post_end = std::chrono::high_resolution_clock::now();
    double q6_cpu_ms = std::chrono::duration<double, std::milli>(q6_cpu_post_end - q6_cpu_post_start).count();

    std::cout << "TPC-H Query 6 Result:" << std::endl;
    std::cout << "Total Revenue: $" << std::fixed << std::setprecision(2) << totalRevenue << std::endl;

    double q6GpuMs = q6_gpu_s * 1000.0;
    size_t totalDataBytes = dataSize * (sizeof(int) + 3 * sizeof(float));
    double bandwidth = (totalDataBytes / (1024.0 * 1024.0 * 1024.0)) / q6_gpu_s;
    printf("\nQ6 | %u rows\n", dataSize);
    printTimingSummary(q6CpuParseMs, q6GpuMs, q6_cpu_ms);
    printf("  Bandwidth:          %10.2f GB/s\n", bandwidth);

    releaseAll(stage1Pipeline, stage2Pipeline,
              shipdateBuffer, discountBuffer, quantityBuffer, extendedpriceBuffer,
              partialRevenuesBuffer, finalRevenueBuffer);
}


// --- SF100 Chunked ---
void runQ6BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q6 Benchmark (SF100 Chunked) ===" << std::endl;

    MappedFile mf;
    if (!mf.open(g_dataset_path + "lineitem.tbl")) {
        std::cerr << "Q6 SF100: Cannot mmap lineitem.tbl" << std::endl;
        return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto lineIndex = buildLineIndex(mf);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    size_t totalRows = lineIndex.size();
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 16, totalRows); // 4 cols * 4 bytes
    const uint num_tg = 2048;
    printf("Q6 SF100: %zu rows, chunk size: %zu (index %.1f ms)\n", totalRows, chunkRows, indexBuildMs);

    auto s1PSO = createPipeline(device, library, "q6_chunked_stage1");
    auto s2PSO = createPipeline(device, library, "q6_chunked_stage2");
    if (!s1PSO || !s2PSO) return;

    // Double-buffered column buffers
    const int NUM_SLOTS = 2;
    struct Q6Slot { MTL::Buffer* shipdate; MTL::Buffer* discount; MTL::Buffer* quantity; MTL::Buffer* extprice; };
    Q6Slot slots[NUM_SLOTS];
    for (int s = 0; s < NUM_SLOTS; s++) {
        slots[s].shipdate = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].discount = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].quantity = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].extprice = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
    }

    MTL::Buffer* partials = device->newBuffer(num_tg * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* finalBuf = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    int start_date = 19940101, end_date = 19950101;
    float min_discount = 0.05f, max_discount = 0.07f, max_quantity = 24.0f;

    double globalRevenue = 0.0;

    auto timing = chunkedStreamLoop(
        commandQueue, slots, NUM_SLOTS, totalRows, chunkRows,
        // Parse
        [&](Q6Slot& slot, size_t startRow, size_t rowCount) {
            parseDateColumnChunk(mf, lineIndex, startRow, rowCount, 10, (int*)slot.shipdate->contents());
            parseFloatColumnChunk(mf, lineIndex, startRow, rowCount, 6, (float*)slot.discount->contents());
            parseFloatColumnChunk(mf, lineIndex, startRow, rowCount, 4, (float*)slot.quantity->contents());
            parseFloatColumnChunk(mf, lineIndex, startRow, rowCount, 5, (float*)slot.extprice->contents());
        },
        // Dispatch
        [&](Q6Slot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(s1PSO);
            enc->setBuffer(slot.shipdate, 0, 0); enc->setBuffer(slot.discount, 0, 1);
            enc->setBuffer(slot.quantity, 0, 2); enc->setBuffer(slot.extprice, 0, 3);
            enc->setBuffer(partials, 0, 4);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 5);
            enc->setBytes(&start_date, sizeof(start_date), 6);
            enc->setBytes(&end_date, sizeof(end_date), 7);
            enc->setBytes(&min_discount, sizeof(min_discount), 8);
            enc->setBytes(&max_discount, sizeof(max_discount), 9);
            enc->setBytes(&max_quantity, sizeof(max_quantity), 10);
            NS::UInteger tgSize = s1PSO->maxTotalThreadsPerThreadgroup();
            if (tgSize > 1024) tgSize = 1024;
            enc->dispatchThreadgroups(MTL::Size::Make(num_tg, 1, 1), MTL::Size::Make(tgSize, 1, 1));
            enc->memoryBarrier(MTL::BarrierScopeBuffers);
            enc->setComputePipelineState(s2PSO);
            enc->setBuffer(partials, 0, 0); enc->setBuffer(finalBuf, 0, 1);
            enc->setBytes(&num_tg, sizeof(num_tg), 2);
            enc->dispatchThreads(MTL::Size::Make(1,1,1), MTL::Size::Make(1,1,1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        // Accumulate
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {
            globalRevenue += *(float*)finalBuf->contents();
        }
    );

    double allCpuParseMs = indexBuildMs + timing.parseMs;
    printf("TPC-H Q6 Result: Revenue = $%.2f\n", globalRevenue);
    printf("\nSF100 Q6 | %zu chunks | %zu rows\n", timing.chunkCount, totalRows);
    printTimingSummary(allCpuParseMs, timing.gpuMs, timing.postMs);

    releaseAll(s1PSO, s2PSO);
    for (int s = 0; s < NUM_SLOTS; s++)
        releaseAll(slots[s].shipdate, slots[s].discount, slots[s].quantity, slots[s].extprice);
    releaseAll(partials, finalBuf);
}
