#include "infra.h"

// ===================================================================
// TPC-H Q19 — Discounted Revenue
// ===================================================================

// --- Standard (SF1/SF10) ---
void runQ19Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "--- Running TPC-H Query 19 Benchmark ---" << std::endl;

    auto parseStart = std::chrono::high_resolution_clock::now();
    const std::string sf_path = g_dataset_path;

    // Load part data (pure I/O)
    auto pCols = loadColumnsMulti(sf_path + "part.tbl", {{0, ColType::INT}, {3, ColType::CHAR_FIXED, 10}, {5, ColType::INT}, {6, ColType::CHAR_FIXED, 10}});
    auto& p_partkey = pCols.ints(0); auto& p_brand = pCols.chars(3); auto& p_size = pCols.ints(5); auto& p_container = pCols.chars(6);

    // Load lineitem columns
    auto lCols = loadColumnsMulti(sf_path + "lineitem.tbl", {{1, ColType::INT}, {4, ColType::FLOAT}, {5, ColType::FLOAT}, {6, ColType::FLOAT}, {13, ColType::CHAR_FIXED, 25}, {14, ColType::CHAR_FIXED, 10}});
    auto& l_partkey = lCols.ints(1); auto& l_quantity = lCols.floats(4);
    auto& l_extendedprice = lCols.floats(5); auto& l_discount = lCols.floats(6);
    auto& l_shipinstruct = lCols.chars(13); auto& l_shipmode = lCols.chars(14);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double cpuParseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    uint dataSize = (uint)l_partkey.size();
    uint partSize = (uint)p_partkey.size();
    if (dataSize == 0) { std::cerr << "Q19: no data loaded" << std::endl; return; }
    std::cout << "Loaded " << dataSize << " lineitem rows for Q19." << std::endl;

    int max_partkey = 0;
    for (int k : p_partkey) max_partkey = std::max(max_partkey, k);
    uint mapSize = (uint)(max_partkey + 1);
    const uint brand_stride = 10;
    const uint container_stride = 10;
    const uint shipmode_stride = 10;
    const uint shipinstruct_stride = 25;

    auto mapPSO      = createPipeline(device, library, "q19_build_part_group_map_kernel");
    auto s1PSO       = createPipeline(device, library, "q19_filter_and_sum_stage1");
    if (!mapPSO || !s1PSO) return;

    const int numTG = 2048;

    // Part group map buffers
    MTL::Buffer* pPartKeyBuf   = device->newBuffer(p_partkey.data(), partSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pBrandBuf     = device->newBuffer(p_brand.data(), (size_t)partSize * brand_stride, MTL::ResourceStorageModeShared);
    MTL::Buffer* pContainerBuf = device->newBuffer(p_container.data(), (size_t)partSize * container_stride, MTL::ResourceStorageModeShared);
    MTL::Buffer* pSizeBuf      = device->newBuffer(p_size.data(), partSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* mapBuf        = device->newBuffer(mapSize * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    memset(mapBuf->contents(), 0xFF, mapSize * sizeof(uint8_t));

    MTL::Buffer* smBuf         = device->newBuffer(l_shipmode.data(), (size_t)dataSize * shipmode_stride, MTL::ResourceStorageModeShared);
    MTL::Buffer* siBuf         = device->newBuffer(l_shipinstruct.data(), (size_t)dataSize * shipinstruct_stride, MTL::ResourceStorageModeShared);

    // Main computation buffers
    MTL::Buffer* partkeyBuf   = device->newBuffer(l_partkey.data(), dataSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* qtyBuf       = device->newBuffer(l_quantity.data(), dataSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* priceBuf     = device->newBuffer(l_extendedprice.data(), dataSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* discBuf      = device->newBuffer(l_discount.data(), dataSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* totalRevenueBuf = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    double gpuSec = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(mapBuf->contents(), 0xFF, mapSize * sizeof(uint8_t));
        *(float*)totalRevenueBuf->contents() = 0.0f;

        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

        // Phase 1: Build part group map on GPU
        enc->setComputePipelineState(mapPSO);
        enc->setBuffer(pPartKeyBuf, 0, 0);
        enc->setBuffer(pBrandBuf, 0, 1);
        enc->setBuffer(pContainerBuf, 0, 2);
        enc->setBuffer(pSizeBuf, 0, 3);
        enc->setBuffer(mapBuf, 0, 4);
        enc->setBytes(&partSize, sizeof(partSize), 5);
        enc->setBytes(&brand_stride, sizeof(brand_stride), 6);
        enc->setBytes(&container_stride, sizeof(container_stride), 7);
        {
            NS::UInteger tgSize = mapPSO->maxTotalThreadsPerThreadgroup();
            if (tgSize > 1024) tgSize = 1024;
            uint numGroups = (partSize + (uint)tgSize - 1) / (uint)tgSize;
            enc->dispatchThreadgroups(MTL::Size::Make(numGroups, 1, 1), MTL::Size::Make(tgSize, 1, 1));
        }

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // Phase 2: Main filter+sum
        enc->setComputePipelineState(s1PSO);
        enc->setBuffer(partkeyBuf, 0, 0);
        enc->setBuffer(qtyBuf, 0, 1);
        enc->setBuffer(priceBuf, 0, 2);
        enc->setBuffer(discBuf, 0, 3);
        enc->setBuffer(smBuf, 0, 4);
        enc->setBuffer(siBuf, 0, 5);
        enc->setBuffer(mapBuf, 0, 6);
        enc->setBuffer(totalRevenueBuf, 0, 7);
        enc->setBytes(&dataSize, sizeof(dataSize), 8);
        enc->setBytes(&mapSize, sizeof(mapSize), 9);
        enc->setBytes(&shipmode_stride, sizeof(shipmode_stride), 10);
        enc->setBytes(&shipinstruct_stride, sizeof(shipinstruct_stride), 11);
        NS::UInteger tgSize = s1PSO->maxTotalThreadsPerThreadgroup();
        if (tgSize > 1024) tgSize = 1024;
        enc->dispatchThreadgroups(MTL::Size::Make(numTG, 1, 1), MTL::Size::Make(tgSize, 1, 1));
        enc->endEncoding();

        cb->commit();
        cb->waitUntilCompleted();
        if (iter == 2) gpuSec = cb->GPUEndTime() - cb->GPUStartTime();
    }

    auto cpuPostStart = std::chrono::high_resolution_clock::now();
    float revenue = *(float*)totalRevenueBuf->contents();
    auto cpuPostEnd = std::chrono::high_resolution_clock::now();
    double cpuPostMs = std::chrono::duration<double, std::milli>(cpuPostEnd - cpuPostStart).count();

    printf("\nTPC-H Q19 Result: Revenue = $%.2f\n", revenue);
    printf("\nQ19 | %u rows\n", dataSize);
    printTimingSummary(cpuParseMs, gpuSec * 1000.0, cpuPostMs);

    releaseAll(mapPSO, s1PSO,
               pPartKeyBuf, pBrandBuf, pContainerBuf, pSizeBuf,
               smBuf, siBuf,
               partkeyBuf, qtyBuf, priceBuf, discBuf,
               mapBuf, totalRevenueBuf);
}

// --- SF100 Chunked ---
void runQ19BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q19 Benchmark (SF100 Chunked) ===" << std::endl;

    MappedFile partFile, liFile;
    if (!partFile.open(g_dataset_path + "part.tbl") || !liFile.open(g_dataset_path + "lineitem.tbl")) {
        std::cerr << "Q19 SF100: Cannot open required TBL files" << std::endl;
        return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto partIndex = buildLineIndex(partFile);
    auto liIndex   = buildLineIndex(liFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    size_t partRows = partIndex.size(), liRows = liIndex.size();
    printf("Q19 SF100: part=%zu, lineitem=%zu rows (index %.1f ms)\n", partRows, liRows, indexBuildMs);

    // Build part group map on GPU
    auto buildT0 = std::chrono::high_resolution_clock::now();
    std::vector<int> p_partkey(partRows);
    parseIntColumnChunk(partFile, partIndex, 0, partRows, 0, p_partkey.data());
    const uint brand_stride = 10;
    std::vector<char> p_brand(partRows * brand_stride);
    parseCharColumnChunkFixed(partFile, partIndex, 0, partRows, 3, brand_stride, p_brand.data());
    const uint container_stride = 10;
    std::vector<char> p_container(partRows * container_stride);
    parseCharColumnChunkFixed(partFile, partIndex, 0, partRows, 6, container_stride, p_container.data());
    std::vector<int> p_size(partRows);
    parseIntColumnChunk(partFile, partIndex, 0, partRows, 5, p_size.data());

    int max_partkey = 0;
    for (auto k : p_partkey) max_partkey = std::max(max_partkey, k);
    uint mapSizeU = (uint)(max_partkey + 1);
    auto buildT1 = std::chrono::high_resolution_clock::now();
    double buildMs = std::chrono::duration<double, std::milli>(buildT1 - buildT0).count();
    printf("Part group map: %d max_partkey (%.1f MB), parse %.1f ms\n",
           max_partkey, mapSizeU / (1024.0*1024.0), buildMs);

    auto mapPSO    = createPipeline(device, library, "q19_build_part_group_map_kernel");
    auto filterPSO = createPipeline(device, library, "q19_chunked_shipmode_filter_kernel");
    auto s1PSO     = createPipeline(device, library, "q19_chunked_stage1");
    auto s2PSO     = createPipeline(device, library, "q19_chunked_stage2");
    if (!mapPSO || !filterPSO || !s1PSO || !s2PSO) return;

    // Upload part data and build map on GPU
    uint partSizeU = (uint)partRows;
    MTL::Buffer* pPartKeyBuf   = device->newBuffer(p_partkey.data(), partRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pBrandBuf     = device->newBuffer(p_brand.data(), partRows * brand_stride, MTL::ResourceStorageModeShared);
    MTL::Buffer* pContainerBuf = device->newBuffer(p_container.data(), partRows * container_stride, MTL::ResourceStorageModeShared);
    MTL::Buffer* pSizeBuf      = device->newBuffer(p_size.data(), partRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* mapBuf        = device->newBuffer(mapSizeU * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    memset(mapBuf->contents(), 0xFF, mapSizeU * sizeof(uint8_t));

    // Dispatch map kernel on GPU
    {
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(mapPSO);
        enc->setBuffer(pPartKeyBuf, 0, 0);
        enc->setBuffer(pBrandBuf, 0, 1);
        enc->setBuffer(pContainerBuf, 0, 2);
        enc->setBuffer(pSizeBuf, 0, 3);
        enc->setBuffer(mapBuf, 0, 4);
        enc->setBytes(&partSizeU, sizeof(partSizeU), 5);
        enc->setBytes(&brand_stride, sizeof(brand_stride), 6);
        enc->setBytes(&container_stride, sizeof(container_stride), 7);
        NS::UInteger tgSize = mapPSO->maxTotalThreadsPerThreadgroup();
        if (tgSize > 1024) tgSize = 1024;
        uint numGroups = (partSizeU + (uint)tgSize - 1) / (uint)tgSize;
        enc->dispatchThreadgroups(MTL::Size::Make(numGroups, 1, 1), MTL::Size::Make(tgSize, 1, 1));
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        double mapGpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
        printf("Part group map built on GPU in %.2f ms\n", mapGpuMs);
    }
    releaseAll(pPartKeyBuf, pBrandBuf, pContainerBuf, pSizeBuf);

    uint mapSize = mapSizeU;

    // lineitem: partkey(4) + qty(4) + price(4) + discount(4) + shipmode(10) + shipinstruct(25) = 51 bytes/row
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 51, liRows);
    const uint shipmode_stride = 10;
    const uint shipinstruct_stride = 25;
    printf("Chunk size: %zu rows\n", chunkRows);

    const int NUM_SLOTS = 2;
    struct Q19Slot {
        MTL::Buffer* partkey; MTL::Buffer* quantity; MTL::Buffer* extprice;
        MTL::Buffer* discount; MTL::Buffer* shipmode; MTL::Buffer* shipinstruct;
    };
    Q19Slot slots[NUM_SLOTS];
    for (int s = 0; s < NUM_SLOTS; s++) {
        slots[s].partkey      = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].quantity     = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].extprice     = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].discount     = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].shipmode     = device->newBuffer(chunkRows * shipmode_stride, MTL::ResourceStorageModeShared);
        slots[s].shipinstruct = device->newBuffer(chunkRows * shipinstruct_stride, MTL::ResourceStorageModeShared);
    }

    MTL::Buffer* totalRevenueBuf = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
    *(float*)totalRevenueBuf->contents() = 0.0f;

    auto timing = chunkedStreamLoop(
        commandQueue, slots, NUM_SLOTS, liRows, chunkRows,
        // Parse (pure I/O)
        [&](Q19Slot& slot, size_t startRow, size_t rowCount) {
            parseIntColumnChunk(liFile, liIndex, startRow, rowCount, 1, (int*)slot.partkey->contents());
            parseFloatColumnChunk(liFile, liIndex, startRow, rowCount, 4, (float*)slot.quantity->contents());
            parseFloatColumnChunk(liFile, liIndex, startRow, rowCount, 5, (float*)slot.extprice->contents());
            parseFloatColumnChunk(liFile, liIndex, startRow, rowCount, 6, (float*)slot.discount->contents());
            parseCharColumnChunkFixed(liFile, liIndex, startRow, rowCount, 14, shipmode_stride, (char*)slot.shipmode->contents());
            parseCharColumnChunkFixed(liFile, liIndex, startRow, rowCount, 13, shipinstruct_stride, (char*)slot.shipinstruct->contents());
        },
        // Dispatch: shipmode filter on GPU then main filter+sum
        [&](Q19Slot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto enc = cmdBuf->computeCommandEncoder();
            // Main filter+sum kernel
            enc->setComputePipelineState(s1PSO);
            enc->setBuffer(slot.partkey, 0, 0);
            enc->setBuffer(slot.quantity, 0, 1);
            enc->setBuffer(slot.extprice, 0, 2);
            enc->setBuffer(slot.discount, 0, 3);
            enc->setBuffer(slot.shipmode, 0, 4);
            enc->setBuffer(slot.shipinstruct, 0, 5);
            enc->setBuffer(mapBuf, 0, 6);
            enc->setBuffer(totalRevenueBuf, 0, 7);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 8);
            enc->setBytes(&mapSize, sizeof(mapSize), 9);
            enc->setBytes(&shipmode_stride, sizeof(shipmode_stride), 10);
            enc->setBytes(&shipinstruct_stride, sizeof(shipinstruct_stride), 11);
            NS::UInteger tgSize = s1PSO->maxTotalThreadsPerThreadgroup();
            if (tgSize > 1024) tgSize = 1024;
            uint numGroups = (chunkSize + (uint)tgSize - 1) / (uint)tgSize;
            if (numGroups == 0) numGroups = 1;
            if (numGroups > 65535) numGroups = 65535;
            enc->dispatchThreadgroups(MTL::Size::Make(numGroups, 1, 1), MTL::Size::Make(tgSize, 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        // Accumulate
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {}
    );

    double allCpuParseMs = indexBuildMs + buildMs + timing.parseMs;
    float globalRevenue = *(float*)totalRevenueBuf->contents();
    printf("TPC-H Q19 Result: Revenue = $%.2f\n", globalRevenue);
    printf("\nSF100 Q19 | %zu chunks | %zu rows\n", timing.chunkCount, liRows);
    printTimingSummary(allCpuParseMs, timing.gpuMs, timing.postMs);

    releaseAll(mapPSO, s1PSO, mapBuf, totalRevenueBuf);
    for (int s = 0; s < NUM_SLOTS; s++)
        releaseAll(slots[s].partkey, slots[s].quantity, slots[s].extprice,
                   slots[s].discount, slots[s].shipmode, slots[s].shipinstruct);
}
