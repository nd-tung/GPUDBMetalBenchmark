#include "infra.h"

// ===================================================================
// TPC-H Q13 — Customer Distribution
// ===================================================================

// --- Standard (SF1/SF10) ---
void runQ13Benchmark(MTL::Device* pDevice, MTL::CommandQueue* pCommandQueue, MTL::Library* pLibrary) {
    std::cout << "\n--- Running TPC-H Query 13 Benchmark ---" << std::endl;

    const std::string sf_path = g_dataset_path;
    
    // 1. Load data
    auto q13ParseStart = std::chrono::high_resolution_clock::now();
    auto o_custkey = loadIntColumn(sf_path + "orders.tbl", 1);
    auto o_comment = loadCharColumn(sf_path + "orders.tbl", 8, 100);
    auto c_custkey = loadIntColumn(sf_path + "customer.tbl", 0);
    auto q13ParseEnd = std::chrono::high_resolution_clock::now();
    double q13CpuParseMs = std::chrono::duration<double, std::milli>(q13ParseEnd - q13ParseStart).count();

    const uint orders_size = (uint)o_custkey.size();
    const uint customer_size = (uint)c_custkey.size();
    std::cout << "Loaded " << orders_size << " orders and " << customer_size << " customers." << std::endl;

    // 2. Setup kernels
    auto pFusedCountPipe = createPipeline(pDevice, pLibrary, "q13_fused_direct_count_kernel");
    auto pHistPipe = createPipeline(pDevice, pLibrary, "q13_build_histogram_kernel");
    if (!pFusedCountPipe || !pHistPipe) return;

    // 3. Create Buffers
    const uint num_threadgroups = 2048;
    MTL::Buffer* pOrdCustKeyBuffer = pDevice->newBuffer(o_custkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdCommentBuffer = pDevice->newBuffer(o_comment.data(), o_comment.size() * sizeof(char), MTL::ResourceStorageModeShared);

    // Direct mapping output: per-customer order counts (index = custkey - 1).
    std::vector<uint> cpu_counts_per_customer(customer_size, 0u);
    MTL::Buffer* pCountsPerCustomerBuffer = pDevice->newBuffer(cpu_counts_per_customer.data(), customer_size * sizeof(uint), MTL::ResourceStorageModeShared);

    // Histogram buffer for GPU histogram kernel
    const uint hist_max_bins = 256;
    MTL::Buffer* pHistogramBuf = pDevice->newBuffer(hist_max_bins * sizeof(uint), MTL::ResourceStorageModeShared);

    // 4. Dispatch the fused GPU stage (2 warmup + 1 measured)
    double gpuExecutionTime = 0.0;
    
    for(int iter = 0; iter < 3; ++iter) {
        // Reset output buffers
        std::memset(pCountsPerCustomerBuffer->contents(), 0, customer_size * sizeof(uint));
        std::memset(pHistogramBuf->contents(), 0, hist_max_bins * sizeof(uint));
        
        MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
        
        // Encoder 1: count kernel
        MTL::ComputeCommandEncoder* enc1 = pCommandBuffer->computeCommandEncoder();
        enc1->setComputePipelineState(pFusedCountPipe);
        enc1->setBuffer(pOrdCustKeyBuffer, 0, 0);
        enc1->setBuffer(pOrdCommentBuffer, 0, 1);
        enc1->setBuffer(pCountsPerCustomerBuffer, 0, 2);
        enc1->setBytes(&orders_size, sizeof(orders_size), 3);
        enc1->setBytes(&customer_size, sizeof(customer_size), 4);
        enc1->dispatchThreadgroups(MTL::Size(num_threadgroups, 1, 1), MTL::Size(1024, 1, 1));
        enc1->endEncoding();

        // Encoder 2: histogram kernel (encoder boundary provides memory barrier)
        MTL::ComputeCommandEncoder* enc2 = pCommandBuffer->computeCommandEncoder();
        enc2->setComputePipelineState(pHistPipe);
        enc2->setBuffer(pCountsPerCustomerBuffer, 0, 0);
        enc2->setBuffer(pHistogramBuf, 0, 1);
        enc2->setBytes(&customer_size, sizeof(customer_size), 2);
        enc2->setBytes(&hist_max_bins, sizeof(hist_max_bins), 3);
        enc2->dispatchThreadgroups(MTL::Size(num_threadgroups, 1, 1), MTL::Size(1024, 1, 1));
        enc2->endEncoding();

        // 5. Execute GPU work
        pCommandBuffer->commit();
        pCommandBuffer->waitUntilCompleted();
        
        if (iter == 2) {
            gpuExecutionTime = pCommandBuffer->GPUEndTime() - pCommandBuffer->GPUStartTime();
        }
    }

    // 6. Read back histogram from GPU
    auto q13_cpu_merge_start = std::chrono::high_resolution_clock::now();
    postProcessQ13(pHistogramBuf->contents(), hist_max_bins);
    auto q13_cpu_merge_end = std::chrono::high_resolution_clock::now();
    double q13_cpu_merge_time = std::chrono::duration<double>(q13_cpu_merge_end - q13_cpu_merge_start).count();

    double q13GpuMs = gpuExecutionTime * 1000.0;
    double q13CpuPostMs = q13_cpu_merge_time * 1000.0;
    printf("\nQ13 | %u orders | %u customers\n", orders_size, customer_size);
    printTimingSummary(q13CpuParseMs, q13GpuMs, q13CpuPostMs);

    releaseAll(pFusedCountPipe, pHistPipe,
              pOrdCustKeyBuffer, pOrdCommentBuffer, pCountsPerCustomerBuffer, pHistogramBuf);
}


// --- SF100 Chunked ---
void runQ13BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q13 Benchmark (SF100 Chunked) ===" << std::endl;

    MappedFile custFile, ordFile;
    if (!custFile.open(g_dataset_path + "customer.tbl") || !ordFile.open(g_dataset_path + "orders.tbl")) {
        std::cerr << "Q13 SF100: Cannot open files" << std::endl;
        return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto custIdx = buildLineIndex(custFile);
    auto ordIdx = buildLineIndex(ordFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();
    size_t custRows = custIdx.size(), ordRows = ordIdx.size();
    printf("Q13 SF100: customer=%zu, orders=%zu (index %.1f ms)\n", custRows, ordRows, indexBuildMs);

    // Output: per-customer order counts (persistent across chunks, atomics accumulate)
    MTL::Buffer* pCountsBuf = device->newBuffer(custRows * sizeof(uint), MTL::ResourceStorageModeShared);
    memset(pCountsBuf->contents(), 0, custRows * sizeof(uint));

    auto pPipe = createPipeline(device, library, "q13_fused_direct_count_kernel");
    auto pHistPipe = createPipeline(device, library, "q13_build_histogram_kernel");
    if (!pPipe || !pHistPipe) return;

    // Histogram buffer for GPU histogram kernel
    const uint hist_max_bins = 256;
    MTL::Buffer* pHistogramBuf = device->newBuffer(hist_max_bins * sizeof(uint), MTL::ResourceStorageModeShared);

    // Stream orders in chunks: custkey(4 bytes) + comment(100 bytes) = 104 bytes/row
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 104, ordRows);
    size_t numChunks = (ordRows + chunkRows - 1) / chunkRows;
    printf("Q13 chunk size: %zu rows, %zu chunks\n", chunkRows, numChunks);

    // Allocate triple-buffered chunk buffers for custkey + comment
    const int Q13_NUM_SLOTS = 3;
    struct Q13ChunkSlot { MTL::Buffer* custKey; MTL::Buffer* comment; };
    Q13ChunkSlot q13Slots[Q13_NUM_SLOTS];
    for (int s = 0; s < Q13_NUM_SLOTS; s++) {
        q13Slots[s].custKey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        q13Slots[s].comment = device->newBuffer(chunkRows * 100 * sizeof(char), MTL::ResourceStorageModeShared);
    }

    const uint num_threadgroups = 2048;
    uint cs = (uint)custRows;

    auto timing = chunkedStreamLoop(
        commandQueue, q13Slots, Q13_NUM_SLOTS, ordRows, chunkRows,
        // Parse
        [&](Q13ChunkSlot& slot, size_t startRow, size_t rowCount) {
            parseIntColumnChunk(ordFile, ordIdx, startRow, rowCount, 1, (int*)slot.custKey->contents());
            parseCharColumnChunkFixed(ordFile, ordIdx, startRow, rowCount, 8, 100, (char*)slot.comment->contents());
        },
        // Dispatch
        [&](Q13ChunkSlot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(pPipe);
            enc->setBuffer(slot.custKey, 0, 0);
            enc->setBuffer(slot.comment, 0, 1);
            enc->setBuffer(pCountsBuf, 0, 2);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 3);
            enc->setBytes(&cs, sizeof(cs), 4);
            enc->dispatchThreadgroups(MTL::Size(num_threadgroups, 1, 1), MTL::Size(1024, 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        // Progress
        [&]([[maybe_unused]] uint chunkSize, size_t chunkNum) {
            if ((chunkNum + 1) % 5 == 0 || chunkNum + 1 == numChunks) {
                printf("  Chunk %zu/%zu done\n", chunkNum + 1, numChunks);
            }
        }
    );

    // GPU post-processing: dispatch histogram kernel on per-customer counts
    std::memset(pHistogramBuf->contents(), 0, hist_max_bins * sizeof(uint));
    uint cs_all = (uint)custRows;
    MTL::CommandBuffer* histCb = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* histEnc = histCb->computeCommandEncoder();
    histEnc->setComputePipelineState(pHistPipe);
    histEnc->setBuffer(pCountsBuf, 0, 0);
    histEnc->setBuffer(pHistogramBuf, 0, 1);
    histEnc->setBytes(&cs_all, sizeof(cs_all), 2);
    histEnc->setBytes(&hist_max_bins, sizeof(hist_max_bins), 3);
    histEnc->dispatchThreadgroups(MTL::Size(num_threadgroups, 1, 1), MTL::Size(1024, 1, 1));
    histEnc->endEncoding();
    histCb->commit();
    histCb->waitUntilCompleted();
    double histGpuMs = (histCb->GPUEndTime() - histCb->GPUStartTime()) * 1000.0;

    // CPU post-processing: read back histogram from GPU + sort
    auto postStart = std::chrono::high_resolution_clock::now();
    postProcessQ13(pHistogramBuf->contents(), hist_max_bins);
    auto postEnd = std::chrono::high_resolution_clock::now();
    double cpuPostMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    double allCpuParseMs = indexBuildMs + timing.parseMs;
    double allGpuMs = timing.gpuMs + histGpuMs;
    printf("\nSF100 Q13 | %zu chunks | %zu rows\n", timing.chunkCount, ordRows);
    printTimingSummary(allCpuParseMs, allGpuMs, cpuPostMs);

    // Cleanup
    releaseAll(pPipe, pHistPipe);
    for (int s = 0; s < Q13_NUM_SLOTS; s++) {
        releaseAll(q13Slots[s].custKey, q13Slots[s].comment);
    }
    releaseAll(pCountsBuf, pHistogramBuf);
}
