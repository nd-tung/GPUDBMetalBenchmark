#include "infra.h"

// ===================================================================
// TPC-H Q5 — Local Supplier Volume
// ===================================================================

// --- Standard (SF1/SF10) ---
void runQ5Benchmark(MTL::Device* pDevice, MTL::CommandQueue* pCommandQueue, MTL::Library* pLibrary) {
    std::cout << "\n--- Running TPC-H Query 5 Benchmark ---" << std::endl;

    const std::string sf_path = g_dataset_path;

    // 1. Load data
    auto q5ParseStart = std::chrono::high_resolution_clock::now();
    auto c_custkey = loadIntColumn(sf_path + "customer.tbl", 0);
    auto c_nationkey = loadIntColumn(sf_path + "customer.tbl", 3);

    auto s_suppkey = loadIntColumn(sf_path + "supplier.tbl", 0);
    auto s_nationkey = loadIntColumn(sf_path + "supplier.tbl", 3);

    auto o_orderkey = loadIntColumn(sf_path + "orders.tbl", 0);
    auto o_custkey = loadIntColumn(sf_path + "orders.tbl", 1);
    auto o_orderdate = loadDateColumn(sf_path + "orders.tbl", 4);

    auto l_orderkey = loadIntColumn(sf_path + "lineitem.tbl", 0);
    auto l_suppkey = loadIntColumn(sf_path + "lineitem.tbl", 2);
    auto l_extendedprice = loadFloatColumn(sf_path + "lineitem.tbl", 5);
    auto l_discount = loadFloatColumn(sf_path + "lineitem.tbl", 6);

    auto n_nationkey = loadIntColumn(sf_path + "nation.tbl", 0);
    auto n_name = loadCharColumn(sf_path + "nation.tbl", 1, 25);
    auto n_regionkey = loadIntColumn(sf_path + "nation.tbl", 2);

    auto r_regionkey = loadIntColumn(sf_path + "region.tbl", 0);
    auto r_name = loadCharColumn(sf_path + "region.tbl", 1, 25);
    auto q5ParseEnd = std::chrono::high_resolution_clock::now();
    double q5CpuParseMs = std::chrono::duration<double, std::milli>(q5ParseEnd - q5ParseStart).count();

    const uint customer_size = (uint)c_custkey.size();
    const uint supplier_size = (uint)s_suppkey.size();
    const uint orders_size = (uint)o_orderkey.size();
    const uint lineitem_size = (uint)l_orderkey.size();
    std::cout << "Loaded data. Customer: " << customer_size << ", Supplier: " << supplier_size
              << ", Orders: " << orders_size << ", Lineitem: " << lineitem_size << std::endl;

    // 2. CPU: Identify ASIA nations -> build nation bitmap
    int asia_regionkey = findRegionKey(r_regionkey, r_name.data(), 25, "ASIA");
    if (asia_regionkey == -1) {
        std::cerr << "Error: ASIA region not found" << std::endl;
        return;
    }

    // Build nation name map
    auto nation_names = buildNationNames(n_nationkey, n_name.data(), 25);
    uint cpu_nation_bitmap = buildNationBitmap(n_nationkey, n_regionkey, asia_regionkey);

    // 3. Setup GPU kernels
    auto pCustMapPipe = createPipeline(pDevice, pLibrary, "q5_build_customer_nation_map_kernel");
    auto pSuppMapPipe = createPipeline(pDevice, pLibrary, "q5_build_supplier_nation_map_kernel");
    auto pOrdersMapPipe = createPipeline(pDevice, pLibrary, "q5_build_orders_map_kernel");
    auto pProbeAggPipe = createPipeline(pDevice, pLibrary, "q5_probe_and_aggregate_kernel");
    if (!pCustMapPipe || !pSuppMapPipe || !pOrdersMapPipe || !pProbeAggPipe) return;

    // 4. Create GPU buffers
    int max_custkey = 0;
    for (int k : c_custkey) max_custkey = std::max(max_custkey, k);
    int max_suppkey = 0;
    for (int k : s_suppkey) max_suppkey = std::max(max_suppkey, k);

    const uint cust_map_size = max_custkey + 1;
    const uint supp_map_size = max_suppkey + 1;

    MTL::Buffer* pCustKeyBuf = pDevice->newBuffer(c_custkey.data(), customer_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCustNationBuf = pDevice->newBuffer(c_nationkey.data(), customer_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCustNationMapBuf = pDevice->newBuffer(cust_map_size * sizeof(int), MTL::ResourceStorageModeShared);
    std::memset(pCustNationMapBuf->contents(), -1, cust_map_size * sizeof(int));

    MTL::Buffer* pSuppKeyBuf = pDevice->newBuffer(s_suppkey.data(), supplier_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSuppNationBuf = pDevice->newBuffer(s_nationkey.data(), supplier_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSuppNationMapBuf = pDevice->newBuffer(supp_map_size * sizeof(int), MTL::ResourceStorageModeShared);
    std::memset(pSuppNationMapBuf->contents(), -1, supp_map_size * sizeof(int));

    MTL::Buffer* pNationBitmapBuf = pDevice->newBuffer(&cpu_nation_bitmap, sizeof(uint), MTL::ResourceStorageModeShared);

    MTL::Buffer* pOrdKeyBuf = pDevice->newBuffer(o_orderkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdCustKeyBuf = pDevice->newBuffer(o_custkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdDateBuf = pDevice->newBuffer(o_orderdate.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);

    // Orders direct map: orderkey -> nationkey (replaces hash table)
    int max_orderkey = 0;
    for (int k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    const uint map_size = max_orderkey + 1;
    MTL::Buffer* pOrdersMapBuf = pDevice->newBuffer((size_t)map_size * sizeof(int), MTL::ResourceStorageModeShared);

    MTL::Buffer* pLineOrdKeyBuf = pDevice->newBuffer(l_orderkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineSuppKeyBuf = pDevice->newBuffer(l_suppkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLinePriceBuf = pDevice->newBuffer(l_extendedprice.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineDiscBuf = pDevice->newBuffer(l_discount.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);

    // Revenue array: 25 nations (floats)
    MTL::Buffer* pNationRevenueBuf = pDevice->newBuffer(25 * sizeof(float), MTL::ResourceStorageModeShared);

    const int date_start = 19940101;
    const int date_end = 19950101;
    const uint num_threadgroups = 2048;

    // 5. Execute GPU pipeline (2 warmup + 1 measured)
    double q5_gpu_compute_time = 0.0;

    for (int iter = 0; iter < 3; ++iter) {
        std::memset(pCustNationMapBuf->contents(), -1, cust_map_size * sizeof(int));
        std::memset(pSuppNationMapBuf->contents(), -1, supp_map_size * sizeof(int));
        std::memset(pOrdersMapBuf->contents(), -1, (size_t)map_size * sizeof(int));
        std::memset(pNationRevenueBuf->contents(), 0, 25 * sizeof(float));

        MTL::CommandBuffer* pCmdBuf = pCommandQueue->commandBuffer();

        // Stage 1: Build customer nation map
        MTL::ComputeCommandEncoder* pBuildEnc = pCmdBuf->computeCommandEncoder();
        pBuildEnc->setComputePipelineState(pCustMapPipe);
        pBuildEnc->setBuffer(pCustKeyBuf, 0, 0);
        pBuildEnc->setBuffer(pCustNationBuf, 0, 1);
        pBuildEnc->setBuffer(pCustNationMapBuf, 0, 2);
        pBuildEnc->setBuffer(pNationBitmapBuf, 0, 3);
        pBuildEnc->setBytes(&customer_size, sizeof(customer_size), 4);
        pBuildEnc->dispatchThreads(MTL::Size(customer_size, 1, 1), MTL::Size(256, 1, 1));

        // Stage 2: Build supplier nation map
        pBuildEnc->setComputePipelineState(pSuppMapPipe);
        pBuildEnc->setBuffer(pSuppKeyBuf, 0, 0);
        pBuildEnc->setBuffer(pSuppNationBuf, 0, 1);
        pBuildEnc->setBuffer(pSuppNationMapBuf, 0, 2);
        pBuildEnc->setBuffer(pNationBitmapBuf, 0, 3);
        pBuildEnc->setBytes(&supplier_size, sizeof(supplier_size), 4);
        pBuildEnc->dispatchThreads(MTL::Size(supplier_size, 1, 1), MTL::Size(256, 1, 1));

        // Stage 3: Build orders direct map
        pBuildEnc->memoryBarrier(MTL::BarrierScopeBuffers);
        pBuildEnc->setComputePipelineState(pOrdersMapPipe);
        pBuildEnc->setBuffer(pOrdKeyBuf, 0, 0);
        pBuildEnc->setBuffer(pOrdCustKeyBuf, 0, 1);
        pBuildEnc->setBuffer(pOrdDateBuf, 0, 2);
        pBuildEnc->setBuffer(pOrdersMapBuf, 0, 3);
        pBuildEnc->setBytes(&orders_size, sizeof(orders_size), 4);
        pBuildEnc->setBytes(&date_start, sizeof(date_start), 5);
        pBuildEnc->setBytes(&date_end, sizeof(date_end), 6);
        pBuildEnc->setBytes(&map_size, sizeof(map_size), 7);
        pBuildEnc->setBuffer(pCustNationMapBuf, 0, 8);
        pBuildEnc->dispatchThreads(MTL::Size(orders_size, 1, 1), MTL::Size(256, 1, 1));

        pBuildEnc->endEncoding();
        pCmdBuf->commit();
        pCmdBuf->waitUntilCompleted();

        double buildTime = 0;
        if (iter == 2) {
            buildTime = pCmdBuf->GPUEndTime() - pCmdBuf->GPUStartTime();
        }

        // Stage 4: Probe lineitem
        pCmdBuf = pCommandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* pProbeEnc = pCmdBuf->computeCommandEncoder();
        pProbeEnc->setComputePipelineState(pProbeAggPipe);
        pProbeEnc->setBuffer(pLineOrdKeyBuf, 0, 0);
        pProbeEnc->setBuffer(pLineSuppKeyBuf, 0, 1);
        pProbeEnc->setBuffer(pLinePriceBuf, 0, 2);
        pProbeEnc->setBuffer(pLineDiscBuf, 0, 3);
        pProbeEnc->setBuffer(pOrdersMapBuf, 0, 4);
        pProbeEnc->setBuffer(pSuppNationMapBuf, 0, 5);
        pProbeEnc->setBuffer(pNationRevenueBuf, 0, 6);
        pProbeEnc->setBytes(&lineitem_size, sizeof(lineitem_size), 7);
        pProbeEnc->setBytes(&map_size, sizeof(map_size), 8);
        pProbeEnc->dispatchThreadgroups(MTL::Size(num_threadgroups, 1, 1), MTL::Size(1024, 1, 1));

        pProbeEnc->endEncoding();
        pCmdBuf->commit();
        pCmdBuf->waitUntilCompleted();

        if (iter == 2) {
            double probeTime = pCmdBuf->GPUEndTime() - pCmdBuf->GPUStartTime();
            q5_gpu_compute_time = buildTime + probeTime;
        }
    }

    // 6. CPU post-processing: read revenue per nation, sort
    auto q5CpuPostStart = std::chrono::high_resolution_clock::now();
    postProcessQ5((float*)pNationRevenueBuf->contents(), nation_names);
    auto q5CpuPostEnd = std::chrono::high_resolution_clock::now();
    double q5CpuPostMs = std::chrono::duration<double, std::milli>(q5CpuPostEnd - q5CpuPostStart).count();

    double q5GpuMs = q5_gpu_compute_time * 1000.0;
    printf("\nQ5 | %u rows (lineitem)\n", lineitem_size);
    printTimingSummary(q5CpuParseMs, q5GpuMs, q5CpuPostMs);

    releaseAll(pCustMapPipe, pSuppMapPipe, pOrdersMapPipe, pProbeAggPipe,
              pCustKeyBuf, pCustNationBuf, pCustNationMapBuf,
              pSuppKeyBuf, pSuppNationBuf, pSuppNationMapBuf,
              pNationBitmapBuf,
              pOrdKeyBuf, pOrdCustKeyBuf, pOrdDateBuf, pOrdersMapBuf,
              pLineOrdKeyBuf, pLineSuppKeyBuf, pLinePriceBuf, pLineDiscBuf,
              pNationRevenueBuf);
}


// --- SF100 Chunked ---
void runQ5BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q5 Benchmark (SF100 Chunked) ===" << std::endl;

    MappedFile custFile, suppFile, ordFile, liFile, natFile, regFile;
    if (!custFile.open(g_dataset_path + "customer.tbl") ||
        !suppFile.open(g_dataset_path + "supplier.tbl") ||
        !ordFile.open(g_dataset_path + "orders.tbl") ||
        !liFile.open(g_dataset_path + "lineitem.tbl") ||
        !natFile.open(g_dataset_path + "nation.tbl") ||
        !regFile.open(g_dataset_path + "region.tbl")) {
        std::cerr << "Q5 SF100: Cannot open required TBL files" << std::endl;
        return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto custIdx = buildLineIndex(custFile);
    auto suppIdx = buildLineIndex(suppFile);
    auto ordIdx = buildLineIndex(ordFile);
    auto liIdx = buildLineIndex(liFile);
    auto natIdx = buildLineIndex(natFile);
    auto regIdx = buildLineIndex(regFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    size_t custRows = custIdx.size(), suppRows = suppIdx.size();
    size_t ordRows = ordIdx.size(), liRows = liIdx.size();
    printf("Q5 SF100: customer=%zu, supplier=%zu, orders=%zu, lineitem=%zu (index %.1f ms)\n",
           custRows, suppRows, ordRows, liRows, indexBuildMs);

    // Load nation/region (tiny)
    auto bpT0 = std::chrono::high_resolution_clock::now();
    std::vector<int> n_nationkey, n_regionkey;
    std::vector<char> n_name_chars;
    std::vector<int> r_regionkey;
    std::vector<char> r_name_chars;
    parseNationRegionSF100(natFile, natIdx, n_nationkey, n_regionkey, n_name_chars,
                           &regFile, &regIdx, &r_regionkey, &r_name_chars);

    int asia_regionkey = findRegionKey(r_regionkey, r_name_chars.data(), 25, "ASIA");
    if (asia_regionkey == -1) { std::cerr << "ASIA not found" << std::endl; return; }

    auto nation_names = buildNationNames(n_nationkey, n_name_chars.data(), 25);
    uint cpu_nation_bitmap = buildNationBitmap(n_nationkey, n_regionkey, asia_regionkey);

    // Load customer
    std::vector<int> c_custkey(custRows), c_nationkey(custRows);
    parseIntColumnChunk(custFile, custIdx, 0, custRows, 0, c_custkey.data());
    parseIntColumnChunk(custFile, custIdx, 0, custRows, 3, c_nationkey.data());

    // Load supplier
    std::vector<int> s_suppkey(suppRows), s_nationkey(suppRows);
    parseIntColumnChunk(suppFile, suppIdx, 0, suppRows, 0, s_suppkey.data());
    parseIntColumnChunk(suppFile, suppIdx, 0, suppRows, 3, s_nationkey.data());

    // Load orders
    std::vector<int> o_orderkey(ordRows), o_custkey(ordRows), o_orderdate(ordRows);
    parseIntColumnChunk(ordFile, ordIdx, 0, ordRows, 0, o_orderkey.data());
    parseIntColumnChunk(ordFile, ordIdx, 0, ordRows, 1, o_custkey.data());
    parseDateColumnChunk(ordFile, ordIdx, 0, ordRows, 4, o_orderdate.data());
    auto bpT1 = std::chrono::high_resolution_clock::now();
    double buildParseMs = indexBuildMs + std::chrono::duration<double, std::milli>(bpT1 - bpT0).count();

    // Setup GPU kernels
    auto pCustMapPipe = createPipeline(device, library, "q5_build_customer_nation_map_kernel");
    auto pSuppMapPipe = createPipeline(device, library, "q5_build_supplier_nation_map_kernel");
    auto pOrdersMapPipe = createPipeline(device, library, "q5_build_orders_map_kernel");
    auto pProbeAggPipe = createPipeline(device, library, "q5_probe_and_aggregate_kernel");
    if (!pCustMapPipe || !pSuppMapPipe || !pOrdersMapPipe || !pProbeAggPipe) return;

    // Create build-side GPU buffers
    const uint customer_size = (uint)custRows, supplier_size = (uint)suppRows, orders_size = (uint)ordRows;
    int max_custkey = 0, max_suppkey = 0;
    for (auto k : c_custkey) max_custkey = std::max(max_custkey, k);
    for (auto k : s_suppkey) max_suppkey = std::max(max_suppkey, k);

    MTL::Buffer* pCustKeyBuf = device->newBuffer(c_custkey.data(), custRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCustNationBuf = device->newBuffer(c_nationkey.data(), custRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCustNationMapBuf = device->newBuffer((max_custkey + 1) * sizeof(int), MTL::ResourceStorageModeShared);
    memset(pCustNationMapBuf->contents(), -1, (max_custkey + 1) * sizeof(int));

    MTL::Buffer* pSuppKeyBuf = device->newBuffer(s_suppkey.data(), suppRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSuppNationBuf = device->newBuffer(s_nationkey.data(), suppRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSuppNationMapBuf = device->newBuffer((max_suppkey + 1) * sizeof(int), MTL::ResourceStorageModeShared);
    memset(pSuppNationMapBuf->contents(), -1, (max_suppkey + 1) * sizeof(int));

    MTL::Buffer* pNationBitmapBuf = device->newBuffer(&cpu_nation_bitmap, sizeof(uint), MTL::ResourceStorageModeShared);

    MTL::Buffer* pOrdKeyBuf = device->newBuffer(o_orderkey.data(), ordRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdCustKeyBuf = device->newBuffer(o_custkey.data(), ordRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdDateBuf = device->newBuffer(o_orderdate.data(), ordRows * sizeof(int), MTL::ResourceStorageModeShared);

    int max_orderkey = 0;
    for (auto k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    const uint map_size = max_orderkey + 1;
    MTL::Buffer* pOrdersMapBuf = device->newBuffer((size_t)map_size * sizeof(int), MTL::ResourceStorageModeShared);

    MTL::Buffer* pNationRevenueBuf = device->newBuffer(25 * sizeof(float), MTL::ResourceStorageModeShared);

    const int date_start = 19940101, date_end = 19950101;

    // Execute build phase
    memset(pOrdersMapBuf->contents(), -1, (size_t)map_size * sizeof(int));
    memset(pNationRevenueBuf->contents(), 0, 25 * sizeof(float));

    double buildGpuMs = 0;
    {
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

        enc->setComputePipelineState(pCustMapPipe);
        enc->setBuffer(pCustKeyBuf, 0, 0); enc->setBuffer(pCustNationBuf, 0, 1);
        enc->setBuffer(pCustNationMapBuf, 0, 2); enc->setBuffer(pNationBitmapBuf, 0, 3);
        enc->setBytes(&customer_size, sizeof(customer_size), 4);
        enc->dispatchThreads(MTL::Size(customer_size, 1, 1), MTL::Size(256, 1, 1));

        enc->setComputePipelineState(pSuppMapPipe);
        enc->setBuffer(pSuppKeyBuf, 0, 0); enc->setBuffer(pSuppNationBuf, 0, 1);
        enc->setBuffer(pSuppNationMapBuf, 0, 2); enc->setBuffer(pNationBitmapBuf, 0, 3);
        enc->setBytes(&supplier_size, sizeof(supplier_size), 4);
        enc->dispatchThreads(MTL::Size(supplier_size, 1, 1), MTL::Size(256, 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);
        enc->setComputePipelineState(pOrdersMapPipe);
        enc->setBuffer(pOrdKeyBuf, 0, 0); enc->setBuffer(pOrdCustKeyBuf, 0, 1);
        enc->setBuffer(pOrdDateBuf, 0, 2); enc->setBuffer(pOrdersMapBuf, 0, 3);
        enc->setBytes(&orders_size, sizeof(orders_size), 4);
        enc->setBytes(&date_start, sizeof(date_start), 5);
        enc->setBytes(&date_end, sizeof(date_end), 6);
        enc->setBytes(&map_size, sizeof(map_size), 7);
        enc->setBuffer(pCustNationMapBuf, 0, 8);
        enc->dispatchThreads(MTL::Size(orders_size, 1, 1), MTL::Size(256, 1, 1));

        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
        buildGpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
        printf("Q5 SF100 build phase done (GPU: %.2f ms)\n", buildGpuMs);
    }

    // Stream lineitem in chunks (double-buffered)
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 16, liRows); // 4 cols ~16B/row
    printf("Lineitem chunk size: %zu rows\n", chunkRows);

    struct Q5ChunkSlot {
        MTL::Buffer* orderkey; MTL::Buffer* suppkey; MTL::Buffer* extprice; MTL::Buffer* discount;
    };
    Q5ChunkSlot liSlots[2];
    for (int s = 0; s < 2; s++) {
        liSlots[s].orderkey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        liSlots[s].suppkey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        liSlots[s].extprice = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        liSlots[s].discount = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
    }

    auto timing = chunkedStreamLoop(
        commandQueue, liSlots, 2, liRows, chunkRows,
        // Parse
        [&](Q5ChunkSlot& slot, size_t startRow, size_t rowCount) {
            parseIntColumnChunk(liFile, liIdx, startRow, rowCount, 0, (int*)slot.orderkey->contents());
            parseIntColumnChunk(liFile, liIdx, startRow, rowCount, 2, (int*)slot.suppkey->contents());
            parseFloatColumnChunk(liFile, liIdx, startRow, rowCount, 5, (float*)slot.extprice->contents());
            parseFloatColumnChunk(liFile, liIdx, startRow, rowCount, 6, (float*)slot.discount->contents());
        },
        // Dispatch
        [&](Q5ChunkSlot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(pProbeAggPipe);
            enc->setBuffer(slot.orderkey, 0, 0); enc->setBuffer(slot.suppkey, 0, 1);
            enc->setBuffer(slot.extprice, 0, 2); enc->setBuffer(slot.discount, 0, 3);
            enc->setBuffer(pOrdersMapBuf, 0, 4); enc->setBuffer(pSuppNationMapBuf, 0, 5);
            enc->setBuffer(pNationRevenueBuf, 0, 6);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 7);
            enc->setBytes(&map_size, sizeof(map_size), 8);
            enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        // No accumulation needed
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {}
    );

    // CPU post-processing
    auto postT0 = std::chrono::high_resolution_clock::now();
    postProcessQ5((float*)pNationRevenueBuf->contents(), nation_names);
    auto postT1 = std::chrono::high_resolution_clock::now();
    double cpuPostMs = std::chrono::duration<double, std::milli>(postT1 - postT0).count();

    double allGpuMs = buildGpuMs + timing.gpuMs;
    double allParseMs = buildParseMs + timing.parseMs;
    printf("\nSF100 Q5 | %zu chunks | %zu rows (lineitem)\n", timing.chunkCount, liRows);
    printTimingSummary(allParseMs, allGpuMs, cpuPostMs);

    releaseAll(pCustMapPipe, pSuppMapPipe, pOrdersMapPipe, pProbeAggPipe,
              pCustKeyBuf, pCustNationBuf, pCustNationMapBuf,
              pSuppKeyBuf, pSuppNationBuf, pSuppNationMapBuf,
              pNationBitmapBuf,
              pOrdKeyBuf, pOrdCustKeyBuf, pOrdDateBuf, pOrdersMapBuf, pNationRevenueBuf);
    for (int s = 0; s < 2; s++) {
        releaseAll(liSlots[s].orderkey, liSlots[s].suppkey,
                  liSlots[s].extprice, liSlots[s].discount);
    }
}
