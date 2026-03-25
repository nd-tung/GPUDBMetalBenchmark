#include "infra.h"
#include <cstring>

// ===================================================================
// TPC-H Q13 — Customer Distribution
// ===================================================================

// --- Standard (SF1/SF10) ---
void runQ13Benchmark(MTL::Device* pDevice, MTL::CommandQueue* pCommandQueue, MTL::Library* pLibrary) {
    std::cout << "\n--- Running TPC-H Query 13 Benchmark ---" << std::endl;

    const std::string sf_path = g_dataset_path;
    
    // 1. Load data (pure I/O)
    auto q13ParseStart = std::chrono::high_resolution_clock::now();
    auto o_custkey = loadIntColumn(sf_path + "orders.tbl", 1);
    auto o_comment = loadCharColumn(sf_path + "orders.tbl", 8, 80);  // fixed 80 chars for GPU
    auto c_custkey = loadIntColumn(sf_path + "customer.tbl", 0);
    auto q13ParseEnd = std::chrono::high_resolution_clock::now();
    double q13CpuParseMs = std::chrono::duration<double, std::milli>(q13ParseEnd - q13ParseStart).count();

    const uint orders_size = (uint)o_custkey.size();
    const uint customer_size = (uint)c_custkey.size();
    const uint comment_stride = 80;
    std::cout << "Loaded " << orders_size << " orders and " << customer_size << " customers." << std::endl;

    // 2. Setup kernels
    auto pPatternPipe = createPipeline(pDevice, pLibrary, "q13_pattern_match_kernel");
    auto pCountPipe = createPipeline(pDevice, pLibrary, "q13_count_prefiltered_kernel");
    auto pHistPipe = createPipeline(pDevice, pLibrary, "q13_build_histogram_kernel");
    if (!pPatternPipe || !pCountPipe || !pHistPipe) return;

    // 3. Create Buffers
    const uint num_threadgroups = 2048;
    MTL::Buffer* pOrdCustKeyBuffer = pDevice->newBuffer(o_custkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdCommentBuffer = pDevice->newBuffer(o_comment.data(), (size_t)orders_size * comment_stride * sizeof(char), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdQualifiesBuffer = pDevice->newBuffer(orders_size * sizeof(uint8_t), MTL::ResourceStorageModeShared);

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
        
        // Encoder 0: pattern match kernel (GPU-side)
        MTL::ComputeCommandEncoder* enc0 = pCommandBuffer->computeCommandEncoder();
        enc0->setComputePipelineState(pPatternPipe);
        enc0->setBuffer(pOrdCommentBuffer, 0, 0);
        enc0->setBuffer(pOrdQualifiesBuffer, 0, 1);
        enc0->setBytes(&orders_size, sizeof(orders_size), 2);
        enc0->setBytes(&comment_stride, sizeof(comment_stride), 3);
        {
            NS::UInteger tgSize = pPatternPipe->maxTotalThreadsPerThreadgroup();
            if (tgSize > 1024) tgSize = 1024;
            uint numGroups = (orders_size + (uint)tgSize - 1) / (uint)tgSize;
            enc0->dispatchThreadgroups(MTL::Size(numGroups, 1, 1), MTL::Size(tgSize, 1, 1));
        }
        enc0->endEncoding();

        // Encoder 1: count kernel (prefiltered — no comment processing)
        MTL::ComputeCommandEncoder* enc1 = pCommandBuffer->computeCommandEncoder();
        enc1->setComputePipelineState(pCountPipe);
        enc1->setBuffer(pOrdCustKeyBuffer, 0, 0);
        enc1->setBuffer(pOrdQualifiesBuffer, 0, 1);
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

    releaseAll(pPatternPipe, pCountPipe, pHistPipe,
              pOrdCustKeyBuffer, pOrdCommentBuffer, pOrdQualifiesBuffer, pCountsPerCustomerBuffer, pHistogramBuf);
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

    auto pPatternPipe = createPipeline(device, library, "q13_chunked_pattern_match_kernel");
    auto pPipe = createPipeline(device, library, "q13_count_prefiltered_kernel");
    auto pHistPipe = createPipeline(device, library, "q13_build_histogram_kernel");
    if (!pPatternPipe || !pPipe || !pHistPipe) return;

    // Histogram buffer for GPU histogram kernel
    const uint hist_max_bins = 256;
    MTL::Buffer* pHistogramBuf = device->newBuffer(hist_max_bins * sizeof(uint), MTL::ResourceStorageModeShared);

    // Stream orders in chunks: custkey(4) + comment(80) = 84 bytes/row for parse, GPU reduces to 5 bytes/row
    const uint comment_stride = 80;
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 84, ordRows);
    size_t numChunks = (ordRows + chunkRows - 1) / chunkRows;
    printf("Q13 chunk size: %zu rows, %zu chunks\n", chunkRows, numChunks);

    // Allocate triple-buffered chunk buffers for custkey + comment + qualifies
    const int Q13_NUM_SLOTS = 3;
    struct Q13ChunkSlot { MTL::Buffer* custKey; MTL::Buffer* comment; MTL::Buffer* qualifies; };
    Q13ChunkSlot q13Slots[Q13_NUM_SLOTS];
    for (int s = 0; s < Q13_NUM_SLOTS; s++) {
        q13Slots[s].custKey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        q13Slots[s].comment = device->newBuffer(chunkRows * comment_stride * sizeof(char), MTL::ResourceStorageModeShared);
        q13Slots[s].qualifies = device->newBuffer(chunkRows * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    }

    const uint num_threadgroups = 2048;
    uint cs = (uint)custRows;

    auto timing = chunkedStreamLoop(
        commandQueue, q13Slots, Q13_NUM_SLOTS, ordRows, chunkRows,
        // Parse (pure I/O)
        [&](Q13ChunkSlot& slot, size_t startRow, size_t rowCount) {
            parseIntColumnChunk(ordFile, ordIdx, startRow, rowCount, 1, (int*)slot.custKey->contents());
            parseCharColumnChunkFixed(ordFile, ordIdx, startRow, rowCount, 8, comment_stride, (char*)slot.comment->contents());
        },
        // Dispatch: pattern match on GPU then count
        [&](Q13ChunkSlot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto enc = cmdBuf->computeCommandEncoder();
            // Pattern match kernel
            enc->setComputePipelineState(pPatternPipe);
            enc->setBuffer(slot.comment, 0, 0);
            enc->setBuffer(slot.qualifies, 0, 1);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 2);
            enc->setBytes(&comment_stride, sizeof(comment_stride), 3);
            {
                NS::UInteger tgSize = pPatternPipe->maxTotalThreadsPerThreadgroup();
                if (tgSize > 1024) tgSize = 1024;
                uint numGroups = (chunkSize + (uint)tgSize - 1) / (uint)tgSize;
                enc->dispatchThreadgroups(MTL::Size(numGroups, 1, 1), MTL::Size(tgSize, 1, 1));
            }
            enc->memoryBarrier(MTL::BarrierScopeBuffers);
            // Count kernel
            enc->setComputePipelineState(pPipe);
            enc->setBuffer(slot.custKey, 0, 0);
            enc->setBuffer(slot.qualifies, 0, 1);
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
    releaseAll(pPatternPipe, pPipe, pHistPipe);
    for (int s = 0; s < Q13_NUM_SLOTS; s++) {
        releaseAll(q13Slots[s].custKey, q13Slots[s].comment, q13Slots[s].qualifies);
    }
    releaseAll(pCountsBuf, pHistogramBuf);
}
