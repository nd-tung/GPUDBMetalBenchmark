#include "infra.h"

// ===================================================================
// Microbenchmarks — Selection, Aggregation, Join
// ===================================================================

// --- Selection Benchmark Test Function ---
void runSingleSelectionTest([[maybe_unused]] MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::ComputePipelineState* pipelineState,
                            MTL::Buffer* inBuffer, MTL::Buffer* resultBuffer,
                            const std::vector<int>& cpuData, int filterValue) {
    
    double gpuExecutionTime = 0.0;
    for(int iter = 0; iter < 3; ++iter) {
        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
        commandEncoder->setComputePipelineState(pipelineState);
        commandEncoder->setBuffer(inBuffer, 0, 0);
        commandEncoder->setBuffer(resultBuffer, 0, 1);
        commandEncoder->setBytes(&filterValue, sizeof(filterValue), 2);
        
        MTL::Size gridSize = MTL::Size::Make(cpuData.size(), 1, 1);
        NS::UInteger threadGroupSize = pipelineState->maxTotalThreadsPerThreadgroup();
        if (threadGroupSize > cpuData.size()) { threadGroupSize = cpuData.size(); }
        MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);
        commandEncoder->dispatchThreads(gridSize, threadgroupSize);
        commandEncoder->endEncoding();
        
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        if (iter == 2) {
            gpuExecutionTime = commandBuffer->GPUEndTime() - commandBuffer->GPUStartTime();
        }
    }
    double dataSizeBytes = (double)cpuData.size() * sizeof(int);
    double dataSizeGB = dataSizeBytes / (1024.0 * 1024.0 * 1024.0);
    double bandwidth = dataSizeGB / gpuExecutionTime;
    
    unsigned int *resultData = (unsigned int *)resultBuffer->contents();
    unsigned int passCount = 0;
    for (size_t i = 0; i < cpuData.size(); ++i) { if (resultData[i] == 1) { passCount++; } }
    float selectivity = 100.0f * (float)passCount / (float)cpuData.size();
    
    std::cout << "--- Filter Value: < " << filterValue << " ---" << std::endl;
    std::cout << "Selectivity: " << selectivity << "% (" << passCount << " rows matched)" << std::endl;
    std::cout << "GPU execution time: " << gpuExecutionTime * 1000.0 << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl << std::endl;
}

// --- Main Function for Selection Benchmark ---
void runSelectionBenchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "--- Running Selection Benchmark ---" << std::endl;

    //Select tpch data file
    std::vector<int> cpuData = loadIntColumn(g_dataset_path + "lineitem.tbl", 1);
    if (cpuData.empty()) { return; }
    std::cout << "Loaded " << cpuData.size() << " rows for selection." << std::endl;

    auto pipelineState = createPipeline(device, library, "selection_kernel");
    if (!pipelineState) return;

    const unsigned long dataSizeBytes = cpuData.size() * sizeof(int);
    MTL::Buffer* inBuffer = device->newBuffer(cpuData.data(), dataSizeBytes, MTL::ResourceStorageModeShared);
    MTL::Buffer* resultBuffer = device->newBuffer(cpuData.size() * sizeof(unsigned int), MTL::ResourceStorageModeShared);

    runSingleSelectionTest(device, commandQueue, pipelineState, inBuffer, resultBuffer, cpuData, 1000);
    runSingleSelectionTest(device, commandQueue, pipelineState, inBuffer, resultBuffer, cpuData, 10000);
    runSingleSelectionTest(device, commandQueue, pipelineState, inBuffer, resultBuffer, cpuData, 50000);
    
    releaseAll(pipelineState, inBuffer, resultBuffer);
}


// --- Main Function for Aggregation Benchmark ---
void runAggregationBenchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "--- Running Aggregation Benchmark ---" << std::endl;

    //Select tpch data file
    std::vector<float> cpuData = loadFloatColumn(g_dataset_path + "lineitem.tbl", 4);
    if (cpuData.empty()) return;
    std::cout << "Loaded " << cpuData.size() << " rows for aggregation." << std::endl;
    const unsigned long dataSizeBytes = cpuData.size() * sizeof(float);
    uint dataSize = (uint)cpuData.size(); // The actual number of elements

    auto stage1Pipeline = createPipeline(device, library, "sum_kernel_stage1");
    auto stage2Pipeline = createPipeline(device, library, "sum_kernel_stage2");
    if (!stage1Pipeline || !stage2Pipeline) return;

    const int numThreadgroups = 2048;
    MTL::Buffer* inBuffer = device->newBuffer(cpuData.data(), dataSizeBytes, MTL::ResourceStorageModeShared);
    MTL::Buffer* partialSumsBuffer = device->newBuffer(numThreadgroups * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* resultBuffer = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();

    // Use a single encoder for both stages to reduce encoder churn
    MTL::ComputeCommandEncoder* enc = commandBuffer->computeCommandEncoder();
    enc->setComputePipelineState(stage1Pipeline);
    enc->setBuffer(inBuffer, 0, 0);
    enc->setBuffer(partialSumsBuffer, 0, 1);
    enc->setBytes(&dataSize, sizeof(dataSize), 2);

    NS::UInteger stage1ThreadGroupSize = stage1Pipeline->maxTotalThreadsPerThreadgroup();
    if (stage1ThreadGroupSize > 1024) stage1ThreadGroupSize = 1024; // cap to match shared_memory[1024] in kernel
    MTL::Size stage1GridSize = MTL::Size::Make(numThreadgroups, 1, 1);
    MTL::Size stage1GroupSize = MTL::Size::Make(stage1ThreadGroupSize, 1, 1);
    enc->dispatchThreadgroups(stage1GridSize, stage1GroupSize);

    // Switch to stage 2 on the same encoder
    const uint numTGs = numThreadgroups;
    enc->setComputePipelineState(stage2Pipeline);
    enc->setBuffer(partialSumsBuffer, 0, 0);
    enc->setBuffer(resultBuffer, 0, 1);
    enc->setBytes(&numTGs, sizeof(numTGs), 2);
    enc->dispatchThreads(MTL::Size::Make(1, 1, 1), MTL::Size::Make(1, 1, 1));
    enc->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    double gpuExecutionTime = commandBuffer->GPUEndTime() - commandBuffer->GPUStartTime();
    double dataSizeGB = (double)dataSizeBytes / (1024.0 * 1024.0 * 1024.0);
    double bandwidth = dataSizeGB / gpuExecutionTime;

    float *finalSum = (float *)resultBuffer->contents();
    std::cout << "Final SUM(l_quantity): " << finalSum[0] << std::endl;
    std::cout << "GPU execution time: " << gpuExecutionTime * 1000.0 << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl << std::endl;
    
    releaseAll(stage1Pipeline, stage2Pipeline, inBuffer, partialSumsBuffer, resultBuffer);
}


// --- Main Function for Join Benchmark ---
void runJoinBenchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "--- Running Join Benchmark ---" << std::endl;
    
    // =================================================================
    // PHASE 1: BUILD
    // =================================================================
    
    // 1. Load Data for the build side (orders table)
    std::vector<int> buildKeys = loadIntColumn(g_dataset_path + "orders.tbl", 0);
    if (buildKeys.empty()) {
        std::cerr << "Error: Could not open 'orders.tbl'. Make sure it's in your " << g_dataset_path << " folder." << std::endl;
        return;
    }
    const uint buildDataSize = (uint)buildKeys.size();
    std::cout << "Loaded " << buildDataSize << " rows from orders.tbl for build phase." << std::endl;

    // 2. Setup Hash Table
    const uint hashTableSize = buildDataSize * 2;
    const unsigned long hashTableSizeBytes = hashTableSize * sizeof(int) * 2;
    std::vector<int> cpuHashTable(hashTableSize * 2, -1);

    // 3. Setup Build Kernel and Pipeline State
    auto buildPipeline = createPipeline(device, library, "hash_join_build");
    if (!buildPipeline) return;

    // 4. Create Build Buffers
    MTL::Buffer* buildKeysBuffer = device->newBuffer(buildKeys.data(), buildKeys.size() * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* buildValuesBuffer = device->newBuffer(buildKeys.data(), buildKeys.size() * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* hashTableBuffer = device->newBuffer(cpuHashTable.data(), hashTableSizeBytes, MTL::ResourceStorageModeShared);

    // 5. Encode and Dispatch Build Kernel
    MTL::CommandBuffer* buildCommandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* buildEncoder = buildCommandBuffer->computeCommandEncoder();
    
    buildEncoder->setComputePipelineState(buildPipeline);
    buildEncoder->setBuffer(buildKeysBuffer, 0, 0);
    buildEncoder->setBuffer(buildValuesBuffer, 0, 1);
    buildEncoder->setBuffer(hashTableBuffer, 0, 2);
    buildEncoder->setBytes(&buildDataSize, sizeof(buildDataSize), 3);
    buildEncoder->setBytes(&hashTableSize, sizeof(hashTableSize), 4);

    MTL::Size buildGridSize = MTL::Size::Make(buildDataSize, 1, 1);
    NS::UInteger buildThreadGroupSize = buildPipeline->maxTotalThreadsPerThreadgroup();
    if (buildThreadGroupSize > buildDataSize) { buildThreadGroupSize = buildDataSize; }
    MTL::Size buildGroupSize = MTL::Size::Make(buildThreadGroupSize, 1, 1);

    buildEncoder->dispatchThreads(buildGridSize, buildGroupSize);
    buildEncoder->endEncoding();
    
    // 6. Execute Build Phase
    buildCommandBuffer->commit();

    // =================================================================
    // PHASE 2: PROBE
    // =================================================================
    
    // 7. Load Data for the probe side (lineitem table)
    // l_orderkey is the 1st column (index 0)
    std::vector<int> probeKeys = loadIntColumn(g_dataset_path + "lineitem.tbl", 0);
    if (probeKeys.empty()) {
        std::cerr << "Error: Could not open 'lineitem.tbl' for probe phase." << std::endl;
        return;
    }
    const uint probeDataSize = (uint)probeKeys.size();
    std::cout << "Loaded " << probeDataSize << " rows from lineitem.tbl for probe phase." << std::endl;

    // 8. Setup Probe Kernel and Pipeline State
    auto probePipeline = createPipeline(device, library, "hash_join_probe");
    if (!probePipeline) return;

    // 9. Create Probe Buffers
    MTL::Buffer* probeKeysBuffer = device->newBuffer(probeKeys.data(), probeKeys.size() * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* matchCountBuffer = device->newBuffer(sizeof(unsigned int), MTL::ResourceStorageModeShared);
    // Clear the match count to zero
    memset(matchCountBuffer->contents(), 0, sizeof(unsigned int));

    // 10. Wait for build to finish, then start probe
    buildCommandBuffer->waitUntilCompleted(); // Ensure build is done before probe starts
    
    MTL::CommandBuffer* probeCommandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* probeEncoder = probeCommandBuffer->computeCommandEncoder();

    probeEncoder->setComputePipelineState(probePipeline);
    probeEncoder->setBuffer(probeKeysBuffer, 0, 0);
    probeEncoder->setBuffer(hashTableBuffer, 0, 1); // Reuse the hash table from build
    probeEncoder->setBuffer(matchCountBuffer, 0, 2);
    probeEncoder->setBytes(&probeDataSize, sizeof(probeDataSize), 3);
    probeEncoder->setBytes(&hashTableSize, sizeof(hashTableSize), 4);

    MTL::Size probeGridSize = MTL::Size::Make(probeDataSize, 1, 1);
    NS::UInteger probeThreadGroupSize = probePipeline->maxTotalThreadsPerThreadgroup();
    if (probeThreadGroupSize > probeDataSize) { probeThreadGroupSize = probeDataSize; }
    MTL::Size probeGroupSize = MTL::Size::Make(probeThreadGroupSize, 1, 1);
    probeEncoder->dispatchThreads(probeGridSize, probeGroupSize);
    probeEncoder->endEncoding();
    
    // 11. Execute Probe Phase
    probeCommandBuffer->commit();
    probeCommandBuffer->waitUntilCompleted();

    // =================================================================
    // FINAL RESULTS
    // =================================================================
    
    double buildTime = buildCommandBuffer->GPUEndTime() - buildCommandBuffer->GPUStartTime();
    double probeTime = probeCommandBuffer->GPUEndTime() - probeCommandBuffer->GPUStartTime();
    
    unsigned int* matchCount = (unsigned int*)matchCountBuffer->contents();

    std::cout << "Join complete. Found " << *matchCount << " total matches." << std::endl;
    std::cout << "Build Phase GPU time: " << buildTime * 1000.0 << " ms" << std::endl;
    std::cout << "Probe Phase GPU time: " << probeTime * 1000.0 << " ms" << std::endl;
    std::cout << "Total Join GPU time: " << (buildTime + probeTime) * 1000.0 << " ms" << std::endl << std::endl;
    
    releaseAll(buildPipeline, probePipeline, buildKeysBuffer, buildValuesBuffer,
              hashTableBuffer, probeKeysBuffer, matchCountBuffer);
}
