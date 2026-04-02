#include "infra.h"

// ===================================================================
// TPC-H Q3 — Shipping Priority
// ===================================================================

// --- Standard (SF1/SF10) ---
void runQ3Benchmark(MTL::Device* pDevice, MTL::CommandQueue* pCommandQueue, MTL::Library* pLibrary) {
    std::cout << "\n--- Running TPC-H Query 3 Benchmark ---" << std::endl;

    // 1. Load data for all three tables
    auto q3ParseStart = std::chrono::high_resolution_clock::now();
    const std::string sf_path = g_dataset_path;
    auto cCols = loadColumnsMulti(sf_path + "customer.tbl", {{0, ColType::INT}, {6, ColType::CHAR1}});
    auto& c_custkey = cCols.ints(0); auto& c_mktsegment = cCols.chars(6);

    auto oCols = loadColumnsMulti(sf_path + "orders.tbl", {{0, ColType::INT}, {1, ColType::INT}, {4, ColType::DATE}, {7, ColType::INT}});
    auto& o_orderkey = oCols.ints(0); auto& o_custkey = oCols.ints(1);
    auto& o_orderdate = oCols.ints(4); auto& o_shippriority = oCols.ints(7);

    auto lCols = loadColumnsMulti(sf_path + "lineitem.tbl", {{0, ColType::INT}, {5, ColType::FLOAT}, {6, ColType::FLOAT}, {10, ColType::DATE}});
    auto& l_orderkey = lCols.ints(0); auto& l_shipdate = lCols.ints(10);
    auto& l_extendedprice = lCols.floats(5); auto& l_discount = lCols.floats(6);
    auto q3ParseEnd = std::chrono::high_resolution_clock::now();
    double q3CpuParseMs = std::chrono::duration<double, std::milli>(q3ParseEnd - q3ParseStart).count();
    
    const uint customer_size = (uint)c_custkey.size();
    const uint orders_size = (uint)o_orderkey.size();
    const uint lineitem_size = (uint)l_orderkey.size();
    std::cout << "Loaded " << customer_size << " customers, " << orders_size << " orders, " << lineitem_size << " lineitem rows." << std::endl;

    // 2. Setup all kernels
    auto pCustBuildPipe = createPipeline(pDevice, pLibrary, "q3_build_customer_bitmap_kernel");
    auto pOrdersBuildPipe = createPipeline(pDevice, pLibrary, "q3_build_orders_map_kernel");
    auto pFusedProbeAggPipe = createPipeline(pDevice, pLibrary, "q3_probe_and_aggregate_direct_kernel");
    auto pCompactPipe = createPipeline(pDevice, pLibrary, "q3_compact_results_kernel");
    if (!pCustBuildPipe || !pOrdersBuildPipe || !pFusedProbeAggPipe || !pCompactPipe) return;

    // 3. Create Buffers
    // Optimization 1: Bitmap for Customer (filter 'BUILDING')
    int max_custkey = 0;
    for(int k : c_custkey) max_custkey = std::max(max_custkey, k);
    MTL::Buffer* pCustomerBitmapBuffer = createBitmapBuffer(pDevice, max_custkey);

    MTL::Buffer* pCustKeyBuffer = pDevice->newBuffer(c_custkey.data(), customer_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCustMktBuffer = pDevice->newBuffer(c_mktsegment.data(), customer_size * sizeof(char), MTL::ResourceStorageModeShared);

    // Optimization 2: Direct Map for Orders
    int max_orderkey = 0;
    for(int k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    const uint orders_map_size = max_orderkey + 1;
    MTL::Buffer* pOrdersMapBuffer = pDevice->newBuffer(orders_map_size * sizeof(int), MTL::ResourceStorageModeShared);
    // Initialize with -1
    std::memset(pOrdersMapBuffer->contents(), -1, orders_map_size * sizeof(int));

    MTL::Buffer* pOrdKeyBuffer = pDevice->newBuffer(o_orderkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdCustKeyBuffer = pDevice->newBuffer(o_custkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdDateBuffer = pDevice->newBuffer(o_orderdate.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdPrioBuffer = pDevice->newBuffer(o_shippriority.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    
    MTL::Buffer* pLineOrdKeyBuffer = pDevice->newBuffer(l_orderkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineShipDateBuffer = pDevice->newBuffer(l_shipdate.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLinePriceBuffer = pDevice->newBuffer(l_extendedprice.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineDiscBuffer = pDevice->newBuffer(l_discount.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);
    
    const uint num_threadgroups = 2048;
    const uint final_ht_size = nextPow2(orders_size * 2);
    MTL::Buffer* pFinalHTBuffer = pDevice->newBuffer(final_ht_size * sizeof(Q3Aggregates_CPU), MTL::ResourceStorageModeShared);
    std::memset(pFinalHTBuffer->contents(), 0, final_ht_size * sizeof(Q3Aggregates_CPU));

    // Dense output buffer for GPU compaction
    MTL::Buffer* pDenseBuffer = pDevice->newBuffer(final_ht_size * sizeof(Q3Aggregates_CPU), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCountBuffer = pDevice->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared);

    const int cutoff_date = 19950315;

    // 4. Dispatch full pipeline (2 warmup + 1 measured)
    double gpuExecutionTime = 0.0;
    
    for(int iter = 0; iter < 3; ++iter) {
        // Reset final HT for each iteration
        std::memset(pFinalHTBuffer->contents(), 0, final_ht_size * sizeof(Q3Aggregates_CPU));
        
        MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = pCommandBuffer->computeCommandEncoder();
        
        // Customer HT build (Bitmap)
        enc->setComputePipelineState(pCustBuildPipe);
        enc->setBuffer(pCustKeyBuffer, 0, 0);
        enc->setBuffer(pCustMktBuffer, 0, 1);
        enc->setBuffer(pCustomerBitmapBuffer, 0, 2);
        enc->setBytes(&customer_size, sizeof(customer_size), 3);
        {
            NS::UInteger threadGroupSize = pCustBuildPipe->maxTotalThreadsPerThreadgroup();
            if (threadGroupSize > 256) threadGroupSize = 256;
            MTL::Size threadgroupSize = MTL::Size(threadGroupSize, 1, 1);
            MTL::Size threadgroups = MTL::Size((customer_size + threadGroupSize - 1) / threadGroupSize, 1, 1);
            enc->dispatchThreadgroups(threadgroups, threadgroupSize);
        }

        // Orders HT build (Direct Map) — filtered by customer bitmap
        enc->memoryBarrier(MTL::BarrierScopeBuffers); // ensure customer bitmap writes are visible
        enc->setComputePipelineState(pOrdersBuildPipe);
        enc->setBuffer(pOrdKeyBuffer, 0, 0);
        enc->setBuffer(pOrdDateBuffer, 0, 1);
        enc->setBuffer(pOrdersMapBuffer, 0, 2);
        enc->setBytes(&orders_size, sizeof(orders_size), 3);
        enc->setBytes(&cutoff_date, sizeof(cutoff_date), 4);
        enc->setBuffer(pOrdCustKeyBuffer, 0, 5);
        enc->setBuffer(pCustomerBitmapBuffer, 0, 6);
        {
            NS::UInteger threadGroupSize = pOrdersBuildPipe->maxTotalThreadsPerThreadgroup();
            if (threadGroupSize > 256) threadGroupSize = 256;
            MTL::Size threadgroupSize = MTL::Size(threadGroupSize, 1, 1);
            MTL::Size threadgroups = MTL::Size((orders_size + threadGroupSize - 1) / threadGroupSize, 1, 1);
            enc->dispatchThreadgroups(threadgroups, threadgroupSize);
        }

        // Fused probe + direct aggregation into final HT
        enc->setComputePipelineState(pFusedProbeAggPipe);
        enc->setBuffer(pLineOrdKeyBuffer, 0, 0);
        enc->setBuffer(pLineShipDateBuffer, 0, 1);
        enc->setBuffer(pLinePriceBuffer, 0, 2);
        enc->setBuffer(pLineDiscBuffer, 0, 3);
        enc->setBuffer(pOrdersMapBuffer, 0, 4);
        enc->setBuffer(pOrdCustKeyBuffer, 0, 5);
        enc->setBuffer(pOrdDateBuffer, 0, 6);
        enc->setBuffer(pOrdPrioBuffer, 0, 7);
        enc->setBuffer(pFinalHTBuffer, 0, 8);
        enc->setBytes(&lineitem_size, sizeof(lineitem_size), 9);
        enc->setBytes(&cutoff_date, sizeof(cutoff_date), 10);
        enc->setBytes(&final_ht_size, sizeof(final_ht_size), 11);
        enc->dispatchThreadgroups(MTL::Size(num_threadgroups, 1, 1), MTL::Size(1024, 1, 1));
        enc->endEncoding();
        
        pCommandBuffer->commit();
        pCommandBuffer->waitUntilCompleted();
        
        if (iter == 2) {
             gpuExecutionTime = pCommandBuffer->GPUEndTime() - pCommandBuffer->GPUStartTime();
        }
    }
    
    // 6. GPU compaction: extract non-empty HT entries into dense buffer
    *(uint*)pCountBuffer->contents() = 0;
    MTL::CommandBuffer* compactCB = pCommandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* compactEnc = compactCB->computeCommandEncoder();
    compactEnc->setComputePipelineState(pCompactPipe);
    compactEnc->setBuffer(pFinalHTBuffer, 0, 0);
    compactEnc->setBuffer(pDenseBuffer, 0, 1);
    compactEnc->setBuffer(pCountBuffer, 0, 2);
    compactEnc->setBytes(&final_ht_size, sizeof(final_ht_size), 3);
    compactEnc->dispatchThreadgroups(MTL::Size((final_ht_size + 255) / 256, 1, 1), MTL::Size(256, 1, 1));
    compactEnc->endEncoding();
    compactCB->commit();
    compactCB->waitUntilCompleted();
    double compactGpuMs = (compactCB->GPUEndTime() - compactCB->GPUStartTime()) * 1000.0;

    uint resultCount = *(uint*)pCountBuffer->contents();
    Q3Aggregates_CPU* dense = (Q3Aggregates_CPU*)pDenseBuffer->contents();
    double cpuMergeMs = sortAndPrintQ3(dense, resultCount);

    double q3GpuMs = gpuExecutionTime * 1000.0 + compactGpuMs;
    printf("\nQ3 | %u rows (lineitem)\n", lineitem_size);
    printTimingSummary(q3CpuParseMs, q3GpuMs, cpuMergeMs);
    
    //Cleanup
    releaseAll(pCustBuildPipe, pOrdersBuildPipe, pFusedProbeAggPipe, pCompactPipe,
              pCustKeyBuffer, pCustMktBuffer, pCustomerBitmapBuffer,
              pOrdKeyBuffer, pOrdCustKeyBuffer, pOrdDateBuffer, pOrdPrioBuffer, pOrdersMapBuffer,
              pLineOrdKeyBuffer, pLineShipDateBuffer, pLinePriceBuffer, pLineDiscBuffer,
              pFinalHTBuffer, pDenseBuffer, pCountBuffer);
}


// --- SF100 Chunked ---
void runQ3BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q3 Benchmark (SF100 Chunked) ===" << std::endl;

    // For Q3, we need customer (small), orders (medium), lineitem (huge).
    // Strategy: Load customer + orders fully (they fit ~22GB at SF100).
    // Stream lineitem in chunks.
    
    // Check if we have enough memory for the build side
    size_t maxMem = device->recommendedMaxWorkingSetSize();
    printf("GPU max working set: %zu MB\n", maxMem / (1024*1024));

    // Load small tables fully
    MappedFile custFile, ordFile, liFile;
    if (!custFile.open(g_dataset_path + "customer.tbl") ||
        !ordFile.open(g_dataset_path + "orders.tbl") ||
        !liFile.open(g_dataset_path + "lineitem.tbl")) {
        std::cerr << "Q3 SF100: Cannot open required TBL files" << std::endl;
        return;
    }

    // Build full customer/orders data (they fit in memory at SF100)
    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto custIndex = buildLineIndex(custFile);
    auto ordIndex = buildLineIndex(ordFile);
    auto liIndex = buildLineIndex(liFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();
    size_t custRows = custIndex.size(), ordRows = ordIndex.size(), liRows = liIndex.size();
    printf("Q3 SF100: customer=%zu, orders=%zu, lineitem=%zu rows (index %.1f ms)\n", custRows, ordRows, liRows, indexBuildMs);

    // Load customer: c_custkey(col 0), c_mktsegment(col 6, first char)
    double buildParseCpuMs = 0;
    auto bpT0 = std::chrono::high_resolution_clock::now();
    std::vector<int> c_custkey(custRows);
    std::vector<char> c_mktsegment(custRows);
    parseIntColumnChunk(custFile, custIndex, 0, custRows, 0, c_custkey.data());
    parseCharColumnChunk(custFile, custIndex, 0, custRows, 6, c_mktsegment.data());

    // Load orders: o_orderkey(col 0), o_custkey(col 1), o_orderdate(col 4), o_shippriority(col 7)
    std::vector<int> o_orderkey(ordRows), o_custkey(ordRows), o_orderdate(ordRows), o_shippriority(ordRows);
    parseIntColumnChunk(ordFile, ordIndex, 0, ordRows, 0, o_orderkey.data());
    parseIntColumnChunk(ordFile, ordIndex, 0, ordRows, 1, o_custkey.data());
    parseDateColumnChunk(ordFile, ordIndex, 0, ordRows, 4, o_orderdate.data());
    parseIntColumnChunk(ordFile, ordIndex, 0, ordRows, 7, o_shippriority.data());
    auto bpT1 = std::chrono::high_resolution_clock::now();
    buildParseCpuMs = std::chrono::duration<double, std::milli>(bpT1 - bpT0).count();

    // Use existing Q3 GPU pipeline for build phase (bitmap + direct map)
    auto pCustBuildPipe = createPipeline(device, library, "q3_build_customer_bitmap_kernel");
    auto pOrdersBuildPipe = createPipeline(device, library, "q3_build_orders_map_kernel");
    auto pFusedProbeAggPipe = createPipeline(device, library, "q3_probe_and_aggregate_direct_kernel");
    auto pCompactPipe = createPipeline(device, library, "q3_compact_results_kernel");
    if (!pCustBuildPipe || !pOrdersBuildPipe || !pFusedProbeAggPipe || !pCompactPipe) return;

    // Build phase buffers (loaded once)
    const uint customer_size = (uint)custRows, orders_size = (uint)ordRows;
    int max_custkey = 0;
    for (auto k : c_custkey) max_custkey = std::max(max_custkey, k);
    MTL::Buffer* pCustBitmapBuf = createBitmapBuffer(device, max_custkey);
    MTL::Buffer* pCustKeyBuf = device->newBuffer(c_custkey.data(), customer_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCustMktBuf = device->newBuffer(c_mktsegment.data(), customer_size * sizeof(char), MTL::ResourceStorageModeShared);

    int max_orderkey = 0;
    for (auto k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    const uint orders_map_size = max_orderkey + 1;
    
    // Check if orders map fits in memory — if not, use hash table join path
    size_t ordersMapBytes = (size_t)orders_map_size * sizeof(int);
    printf("Orders map: %u entries (%.1f MB)\n", orders_map_size, ordersMapBytes / (1024.0*1024.0));
    if (ordersMapBytes > maxMem * 0.3) {
        // ============================================================
        // HASH TABLE JOIN PATH — compact open-addressing hash table
        // instead of direct-map array when max_orderkey is too large.
        // ============================================================
        printf("Using hash table join (direct map too large: %zu MB, limit: %zu MB)\n",
               ordersMapBytes/(1024*1024), (size_t)(maxMem * 0.3)/(1024*1024));

        const int cutoff_date = 19950315;

        // Count qualifying orders on CPU to size the hash table
        uint qualifyingOrders = 0;
        for (size_t i = 0; i < ordRows; i++) {
            if (o_orderdate[i] < cutoff_date) qualifyingOrders++;
        }

        // Hash table capacity: next power of 2 >= qualifying * 1.5 (~67% max load)
        uint ht_capacity = 1;
        uint target = (uint)((double)qualifyingOrders * 1.5);
        if (target < 1024) target = 1024;
        while (ht_capacity < target) ht_capacity <<= 1;
        size_t htBytes = (size_t)ht_capacity * 16; // sizeof(Q3OrdersHTEntry) = 16 bytes
        printf("Orders HT: %u qualifying orders, capacity=%u (%.1f MB, %.1f%% load)\n",
               qualifyingOrders, ht_capacity, htBytes / (1024.0*1024.0),
               100.0 * qualifyingOrders / ht_capacity);

        // Create hash-table-specific pipeline states
        auto pBuildHTPipe = createPipeline(device, library, "q3_build_orders_ht_kernel");
        auto pProbeHTPipe = createPipeline(device, library, "q3_probe_and_aggregate_ht_kernel");
        if (!pBuildHTPipe || !pProbeHTPipe) {
            std::cerr << "Q3 SF100: Failed to create HT pipeline states" << std::endl;
            releaseAll(pCustBuildPipe, pOrdersBuildPipe, pFusedProbeAggPipe, pCompactPipe,
                      pCustKeyBuf, pCustMktBuf, pCustBitmapBuf);
            if (pBuildHTPipe) pBuildHTPipe->release();
            if (pProbeHTPipe) pProbeHTPipe->release();
            return;
        }

        // Allocate orders hash table buffer (0xFF fills key=-1 = empty)
        MTL::Buffer* pHTBuf = device->newBuffer(htBytes, MTL::ResourceStorageModeShared);
        memset(pHTBuf->contents(), 0xFF, htBytes);

        // Upload orders columns for GPU build
        MTL::Buffer* pOrdKeyBuf = device->newBuffer(o_orderkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
        MTL::Buffer* pOrdCustBuf = device->newBuffer(o_custkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
        MTL::Buffer* pOrdDateBuf = device->newBuffer(o_orderdate.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
        MTL::Buffer* pOrdPrioBuf = device->newBuffer(o_shippriority.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);

        // Build phase: customer bitmap + orders hash table
        double buildGpuMs = 0;
        {
            MTL::CommandBuffer* cb = commandQueue->commandBuffer();
            MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

            // Customer bitmap build
            enc->setComputePipelineState(pCustBuildPipe);
            enc->setBuffer(pCustKeyBuf, 0, 0); enc->setBuffer(pCustMktBuf, 0, 1);
            enc->setBuffer(pCustBitmapBuf, 0, 2); enc->setBytes(&customer_size, sizeof(customer_size), 3);
            enc->dispatchThreadgroups(MTL::Size((customer_size + 255)/256, 1, 1), MTL::Size(256, 1, 1));

            // Orders hash table build — filtered by customer bitmap
            enc->memoryBarrier(MTL::BarrierScopeBuffers); // ensure customer bitmap writes are visible
            enc->setComputePipelineState(pBuildHTPipe);
            enc->setBuffer(pOrdKeyBuf, 0, 0); enc->setBuffer(pOrdCustBuf, 0, 1);
            enc->setBuffer(pOrdDateBuf, 0, 2); enc->setBuffer(pOrdPrioBuf, 0, 3);
            enc->setBuffer(pHTBuf, 0, 4);
            enc->setBytes(&orders_size, sizeof(orders_size), 5);
            enc->setBytes(&cutoff_date, sizeof(cutoff_date), 6);
            enc->setBytes(&ht_capacity, sizeof(ht_capacity), 7);
            enc->setBuffer(pCustBitmapBuf, 0, 8);
            enc->dispatchThreadgroups(MTL::Size((orders_size + 255)/256, 1, 1), MTL::Size(256, 1, 1));

            enc->endEncoding();
            cb->commit(); cb->waitUntilCompleted();
            buildGpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
            printf("Build phase done (GPU: %.2f ms)\n", buildGpuMs);
        }

        // Release build-side column buffers (data now embedded in HT)
        pCustKeyBuf->release(); pCustMktBuf->release();
        pOrdKeyBuf->release(); pOrdCustBuf->release();
        pOrdDateBuf->release(); pOrdPrioBuf->release();

        // Stream lineitem in chunks: probe HT + aggregate directly into final HT
        size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 20, liRows);
        printf("Lineitem chunk size: %zu rows\n", chunkRows);

        Q3ChunkSlot liSlots[2];
        for (int s = 0; s < 2; s++) {
            liSlots[s].orderkey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
            liSlots[s].shipdate = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
            liSlots[s].extprice = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
            liSlots[s].discount = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        }

        // Final hash table for GPU aggregation (persists across chunks)
        const uint q3_final_ht_size = nextPow2((uint)std::max((size_t)(ordRows / 64), (size_t)(1 << 20)));
        MTL::Buffer* pFinalHTBuf = device->newBuffer((size_t)q3_final_ht_size * sizeof(Q3Aggregates_CPU), MTL::ResourceStorageModeShared);
        std::memset(pFinalHTBuf->contents(), 0, (size_t)q3_final_ht_size * sizeof(Q3Aggregates_CPU));

        // Dense output buffer for GPU compaction
        MTL::Buffer* pDenseBuf = device->newBuffer((size_t)q3_final_ht_size * sizeof(Q3Aggregates_CPU), MTL::ResourceStorageModeShared);
        MTL::Buffer* pCountBuf = device->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared);

        auto timing = chunkedStreamLoop(
            commandQueue, liSlots, 2, liRows, chunkRows,
            // Parse
            [&](Q3ChunkSlot& slot, size_t startRow, size_t rowCount) {
                parseIntColumnChunk(liFile, liIndex, startRow, rowCount, 0, (int*)slot.orderkey->contents());
                parseDateColumnChunk(liFile, liIndex, startRow, rowCount, 10, (int*)slot.shipdate->contents());
                parseFloatColumnChunk(liFile, liIndex, startRow, rowCount, 5, (float*)slot.extprice->contents());
                parseFloatColumnChunk(liFile, liIndex, startRow, rowCount, 6, (float*)slot.discount->contents());
            },
            // Dispatch
            [&](Q3ChunkSlot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
                auto enc = cmdBuf->computeCommandEncoder();
                enc->setComputePipelineState(pProbeHTPipe);
                enc->setBuffer(slot.orderkey, 0, 0); enc->setBuffer(slot.shipdate, 0, 1);
                enc->setBuffer(slot.extprice, 0, 2); enc->setBuffer(slot.discount, 0, 3);
                enc->setBuffer(pHTBuf, 0, 4);
                enc->setBuffer(pFinalHTBuf, 0, 5);
                enc->setBytes(&chunkSize, sizeof(chunkSize), 6);
                enc->setBytes(&cutoff_date, sizeof(cutoff_date), 7);
                enc->setBytes(&ht_capacity, sizeof(ht_capacity), 8);
                enc->setBytes(&q3_final_ht_size, sizeof(q3_final_ht_size), 9);
                enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));
                enc->endEncoding();
                cmdBuf->commit();
            },
            // No per-chunk accumulation
            [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {}
        );

        // GPU compaction: extract non-empty HT entries into dense buffer
        *(uint*)pCountBuf->contents() = 0;
        {
            MTL::CommandBuffer* cb = commandQueue->commandBuffer();
            MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
            enc->setComputePipelineState(pCompactPipe);
            enc->setBuffer(pFinalHTBuf, 0, 0);
            enc->setBuffer(pDenseBuf, 0, 1);
            enc->setBuffer(pCountBuf, 0, 2);
            enc->setBytes(&q3_final_ht_size, sizeof(q3_final_ht_size), 3);
            enc->dispatchThreadgroups(MTL::Size((q3_final_ht_size + 255) / 256, 1, 1), MTL::Size(256, 1, 1));
            enc->endEncoding();
            cb->commit();
            cb->waitUntilCompleted();
            timing.gpuMs += (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
        }

        uint resultCount = *(uint*)pCountBuf->contents();
        Q3Aggregates_CPU* dense = (Q3Aggregates_CPU*)pDenseBuf->contents();
        double sortMs = sortAndPrintQ3(dense, resultCount);

        double allGpuMs = timing.gpuMs + buildGpuMs;
        double allCpuParseMs = indexBuildMs + buildParseCpuMs + timing.parseMs;
        printf("\nSF100 Q3 | %zu chunks | %zu rows | HT Join\n", timing.chunkCount, liRows);
        printTimingSummary(allCpuParseMs, allGpuMs, sortMs);

        // Cleanup
        releaseAll(pBuildHTPipe, pProbeHTPipe,
                  pCustBuildPipe, pOrdersBuildPipe, pFusedProbeAggPipe, pCompactPipe,
                  pCustBitmapBuf, pHTBuf);
        for (int s = 0; s < 2; s++) {
            releaseAll(liSlots[s].orderkey, liSlots[s].shipdate,
                      liSlots[s].extprice, liSlots[s].discount);
        }
        releaseAll(pFinalHTBuf, pDenseBuf, pCountBuf);
        return;
    }

    MTL::Buffer* pOrdersMapBuf = device->newBuffer(ordersMapBytes, MTL::ResourceStorageModeShared);
    memset(pOrdersMapBuf->contents(), -1, ordersMapBytes);

    MTL::Buffer* pOrdKeyBuf = device->newBuffer(o_orderkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdCustBuf = device->newBuffer(o_custkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdDateBuf = device->newBuffer(o_orderdate.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdPrioBuf = device->newBuffer(o_shippriority.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);

    const int cutoff_date = 19950315;

    // Execute build phase once
    double buildGpuMs = 0;
    {
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

        enc->setComputePipelineState(pCustBuildPipe);
        enc->setBuffer(pCustKeyBuf, 0, 0); enc->setBuffer(pCustMktBuf, 0, 1);
        enc->setBuffer(pCustBitmapBuf, 0, 2); enc->setBytes(&customer_size, sizeof(customer_size), 3);
        enc->dispatchThreadgroups(MTL::Size((customer_size + 255)/256, 1, 1), MTL::Size(256, 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers); // ensure customer bitmap writes are visible
        enc->setComputePipelineState(pOrdersBuildPipe);
        enc->setBuffer(pOrdKeyBuf, 0, 0); enc->setBuffer(pOrdDateBuf, 0, 1);
        enc->setBuffer(pOrdersMapBuf, 0, 2); enc->setBytes(&orders_size, sizeof(orders_size), 3);
        enc->setBytes(&cutoff_date, sizeof(cutoff_date), 4);
        enc->setBuffer(pOrdCustBuf, 0, 5); enc->setBuffer(pCustBitmapBuf, 0, 6);
        enc->dispatchThreadgroups(MTL::Size((orders_size + 255)/256, 1, 1), MTL::Size(256, 1, 1));

        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
        buildGpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
        printf("Build phase done (GPU: %.2f ms)\n", buildGpuMs);
    }

    // Stream lineitem in chunks: probe + aggregate directly into final HT
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 20, liRows); // 4 cols ~20B/row
    printf("Lineitem chunk size: %zu rows\n", chunkRows);

    Q3ChunkSlot liSlots[2];
    for (int s = 0; s < 2; s++) {
        liSlots[s].orderkey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        liSlots[s].shipdate = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        liSlots[s].extprice = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        liSlots[s].discount = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
    }

    // Final hash table for GPU aggregation (persists across chunks)
    const uint q3_final_ht_size = nextPow2((uint)std::max((size_t)(ordRows / 64), (size_t)(1 << 20)));
    MTL::Buffer* pFinalHTBuf = device->newBuffer((size_t)q3_final_ht_size * sizeof(Q3Aggregates_CPU), MTL::ResourceStorageModeShared);
    std::memset(pFinalHTBuf->contents(), 0, (size_t)q3_final_ht_size * sizeof(Q3Aggregates_CPU));

    // Dense output buffer for GPU compaction
    MTL::Buffer* pDenseBuf = device->newBuffer((size_t)q3_final_ht_size * sizeof(Q3Aggregates_CPU), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCountBuf = device->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared);

    auto timing = chunkedStreamLoop(
        commandQueue, liSlots, 2, liRows, chunkRows,
        // Parse
        [&](Q3ChunkSlot& slot, size_t startRow, size_t rowCount) {
            parseIntColumnChunk(liFile, liIndex, startRow, rowCount, 0, (int*)slot.orderkey->contents());
            parseDateColumnChunk(liFile, liIndex, startRow, rowCount, 10, (int*)slot.shipdate->contents());
            parseFloatColumnChunk(liFile, liIndex, startRow, rowCount, 5, (float*)slot.extprice->contents());
            parseFloatColumnChunk(liFile, liIndex, startRow, rowCount, 6, (float*)slot.discount->contents());
        },
        // Dispatch
        [&](Q3ChunkSlot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(pFusedProbeAggPipe);
            enc->setBuffer(slot.orderkey, 0, 0); enc->setBuffer(slot.shipdate, 0, 1);
            enc->setBuffer(slot.extprice, 0, 2); enc->setBuffer(slot.discount, 0, 3);
            enc->setBuffer(pOrdersMapBuf, 0, 4);
            enc->setBuffer(pOrdCustBuf, 0, 5); enc->setBuffer(pOrdDateBuf, 0, 6);
            enc->setBuffer(pOrdPrioBuf, 0, 7);
            enc->setBuffer(pFinalHTBuf, 0, 8);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 9);
            enc->setBytes(&cutoff_date, sizeof(cutoff_date), 10);
            enc->setBytes(&q3_final_ht_size, sizeof(q3_final_ht_size), 11);
            enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        // No per-chunk accumulation
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {}
    );

    // GPU compaction: extract non-empty HT entries into dense buffer
    *(uint*)pCountBuf->contents() = 0;
    {
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pCompactPipe);
        enc->setBuffer(pFinalHTBuf, 0, 0);
        enc->setBuffer(pDenseBuf, 0, 1);
        enc->setBuffer(pCountBuf, 0, 2);
        enc->setBytes(&q3_final_ht_size, sizeof(q3_final_ht_size), 3);
        enc->dispatchThreadgroups(MTL::Size((q3_final_ht_size + 255) / 256, 1, 1), MTL::Size(256, 1, 1));
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        timing.gpuMs += (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
    }

    uint resultCount = *(uint*)pCountBuf->contents();
    Q3Aggregates_CPU* dense = (Q3Aggregates_CPU*)pDenseBuf->contents();
    double sortMs = sortAndPrintQ3(dense, resultCount);

    double allGpuMs = timing.gpuMs + buildGpuMs;
    double allCpuParseMs = indexBuildMs + buildParseCpuMs + timing.parseMs;
    printf("\nSF100 Q3 | %zu chunks | %zu rows\n", timing.chunkCount, liRows);
    printTimingSummary(allCpuParseMs, allGpuMs, sortMs);

    // Cleanup
    releaseAll(pCustBuildPipe, pOrdersBuildPipe, pFusedProbeAggPipe, pCompactPipe,
              pCustKeyBuf, pCustMktBuf, pCustBitmapBuf, pOrdersMapBuf,
              pOrdKeyBuf, pOrdCustBuf, pOrdDateBuf, pOrdPrioBuf);
    for (int s = 0; s < 2; s++) {
        releaseAll(liSlots[s].orderkey, liSlots[s].shipdate,
                  liSlots[s].extprice, liSlots[s].discount);
    }
    releaseAll(pFinalHTBuf, pDenseBuf, pCountBuf);
}
