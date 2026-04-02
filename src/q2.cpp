#include "infra.h"

// ===================================================================
// TPC-H Q2 — Minimum Cost Supplier
// ===================================================================

// --- Standard (SF1/SF10) ---
void runQ2Benchmark(MTL::Device* pDevice, MTL::CommandQueue* pCommandQueue, MTL::Library* pLibrary) {
    std::cout << "\n--- Running TPC-H Query 2 Benchmark ---" << std::endl;

    const std::string sf_path = g_dataset_path;

    // 1. Load data
    auto q2ParseStart = std::chrono::high_resolution_clock::now();
    auto pCols = loadColumnsMulti(sf_path + "part.tbl", {{0, ColType::INT}, {2, ColType::CHAR_FIXED, 25}, {4, ColType::CHAR_FIXED, 25}, {5, ColType::INT}});
    auto& p_partkey = pCols.ints(0); auto& p_mfgr = pCols.chars(2); auto& p_type = pCols.chars(4); auto& p_size = pCols.ints(5);

    auto sCols = loadColumnsMulti(sf_path + "supplier.tbl", {
        {0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}, {2, ColType::CHAR_FIXED, 40},
        {3, ColType::INT}, {4, ColType::CHAR_FIXED, 15}, {5, ColType::FLOAT}, {6, ColType::CHAR_FIXED, 101}
    });
    auto& s_suppkey = sCols.ints(0); auto& s_name = sCols.chars(1); auto& s_address = sCols.chars(2);
    auto& s_nationkey = sCols.ints(3); auto& s_phone = sCols.chars(4); auto& s_acctbal = sCols.floats(5);
    auto& s_comment = sCols.chars(6);

    auto psCols = loadColumnsMulti(sf_path + "partsupp.tbl", {{0, ColType::INT}, {1, ColType::INT}, {3, ColType::FLOAT}});
    auto& ps_partkey = psCols.ints(0); auto& ps_suppkey = psCols.ints(1); auto& ps_supplycost = psCols.floats(3);

    auto nCols = loadColumnsMulti(sf_path + "nation.tbl", {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}, {2, ColType::INT}});
    auto& n_nationkey = nCols.ints(0); auto& n_name = nCols.chars(1); auto& n_regionkey = nCols.ints(2);

    auto rCols = loadColumnsMulti(sf_path + "region.tbl", {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}});
    auto& r_regionkey = rCols.ints(0); auto& r_name = rCols.chars(1);
    auto q2ParseEnd = std::chrono::high_resolution_clock::now();
    double q2CpuParseMs = std::chrono::duration<double, std::milli>(q2ParseEnd - q2ParseStart).count();

    const uint part_size = (uint)p_partkey.size();
    const uint supplier_size = (uint)s_suppkey.size();
    const uint partsupp_size = (uint)ps_partkey.size();
    std::cout << "Loaded data. Part: " << part_size << ", Supplier: " << supplier_size
              << ", PartSupp: " << partsupp_size << std::endl;

    // 2. CPU: Build EUROPE nation set and supplier bitmap
    // Find EUROPE region key
    int europe_regionkey = findRegionKey(r_regionkey, r_name.data(), 25, "EUROPE");
    if (europe_regionkey == -1) {
        std::cerr << "Error: EUROPE region not found" << std::endl;
        return;
    }

    // Build nation name map and europe nation set
    auto nation_names = buildNationNames(n_nationkey, n_name.data(), 25);
    auto europe_nation_keys = filterNationsByRegion(n_nationkey, n_regionkey, europe_regionkey);
    std::cout << "EUROPE nations: " << europe_nation_keys.size() << std::endl;

    // Build supplier bitmap (suppliers in EUROPE nations)
    auto suppBitmap = buildSuppBitmapAndIndex(s_suppkey.data(), s_nationkey.data(),
                                              supplier_size, europe_nation_keys);
    auto& cpu_supp_bitmap = suppBitmap.bitmap;
    auto supp_bitmap_ints = suppBitmap.bitmap_ints;
    auto& supp_index = suppBitmap.index;

    // 3. Setup GPU kernels
    auto pFilterPartPipe = createPipeline(pDevice, pLibrary, "q2_filter_part_kernel");
    auto pMinCostPipe = createPipeline(pDevice, pLibrary, "q2_find_min_cost_kernel");
    auto pMatchPipe = createPipeline(pDevice, pLibrary, "q2_match_suppliers_kernel");
    if (!pFilterPartPipe || !pMinCostPipe || !pMatchPipe) return;

    // 4. Create GPU buffers
    int max_partkey = 0;
    for (int k : p_partkey) max_partkey = std::max(max_partkey, k);
    const uint part_bitmap_ints = (max_partkey + 31) / 32 + 1;

    MTL::Buffer* pPartKeyBuf = pDevice->newBuffer(p_partkey.data(), part_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartSizeBuf = pDevice->newBuffer(p_size.data(), part_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartTypeBuf = pDevice->newBuffer(p_type.data(), p_type.size() * sizeof(char), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartBitmapBuf = pDevice->newBuffer(part_bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);

    MTL::Buffer* pSuppBitmapBuf = pDevice->newBuffer(cpu_supp_bitmap.data(), supp_bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);

    MTL::Buffer* pPsPartKeyBuf = pDevice->newBuffer(ps_partkey.data(), partsupp_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsSuppKeyBuf = pDevice->newBuffer(ps_suppkey.data(), partsupp_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsSupplyCostBuf = pDevice->newBuffer(ps_supplycost.data(), partsupp_size * sizeof(float), MTL::ResourceStorageModeShared);

    // min_cost array: one uint per partkey, initialized to UINT_MAX
    MTL::Buffer* pMinCostBuf = pDevice->newBuffer((max_partkey + 1) * sizeof(uint), MTL::ResourceStorageModeShared);

    // Result buffer
    const uint max_results = 10000; // generous upper bound
    MTL::Buffer* pResultsBuf = pDevice->newBuffer(max_results * sizeof(Q2MatchResult_CPU), MTL::ResourceStorageModeShared);
    MTL::Buffer* pResultCountBuf = pDevice->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared);

    const int target_size = 15;

    // 5. Execute GPU pipeline (2 warmup + 1 measured)
    double q2_gpu_compute_time = 0.0;

    for (int iter = 0; iter < 3; ++iter) {
        // Reset buffers
        std::memset(pPartBitmapBuf->contents(), 0, part_bitmap_ints * sizeof(uint));
        std::memset(pMinCostBuf->contents(), 0xFF, (max_partkey + 1) * sizeof(uint));
        *(uint*)pResultCountBuf->contents() = 0;

        MTL::CommandBuffer* pCmdBuf = pCommandQueue->commandBuffer();

        // Stage 1: Filter parts -> bitmap
        MTL::ComputeCommandEncoder* enc = pCmdBuf->computeCommandEncoder();
        enc->setComputePipelineState(pFilterPartPipe);
        enc->setBuffer(pPartKeyBuf, 0, 0);
        enc->setBuffer(pPartSizeBuf, 0, 1);
        enc->setBuffer(pPartTypeBuf, 0, 2);
        enc->setBuffer(pPartBitmapBuf, 0, 3);
        enc->setBytes(&part_size, sizeof(part_size), 4);
        enc->setBytes(&target_size, sizeof(target_size), 5);
        enc->dispatchThreads(MTL::Size(part_size, 1, 1), MTL::Size(256, 1, 1));

        // Stage 2: Find min cost per partkey
        enc->memoryBarrier(MTL::BarrierScopeBuffers);
        enc->setComputePipelineState(pMinCostPipe);
        enc->setBuffer(pPsPartKeyBuf, 0, 0);
        enc->setBuffer(pPsSuppKeyBuf, 0, 1);
        enc->setBuffer(pPsSupplyCostBuf, 0, 2);
        enc->setBuffer(pPartBitmapBuf, 0, 3);
        enc->setBuffer(pSuppBitmapBuf, 0, 4);
        enc->setBuffer(pMinCostBuf, 0, 5);
        enc->setBytes(&partsupp_size, sizeof(partsupp_size), 6);
        enc->dispatchThreads(MTL::Size(partsupp_size, 1, 1), MTL::Size(256, 1, 1));

        // Stage 3: Match suppliers with min cost
        enc->memoryBarrier(MTL::BarrierScopeBuffers);
        enc->setComputePipelineState(pMatchPipe);
        enc->setBuffer(pPsPartKeyBuf, 0, 0);
        enc->setBuffer(pPsSuppKeyBuf, 0, 1);
        enc->setBuffer(pPsSupplyCostBuf, 0, 2);
        enc->setBuffer(pPartBitmapBuf, 0, 3);
        enc->setBuffer(pSuppBitmapBuf, 0, 4);
        enc->setBuffer(pMinCostBuf, 0, 5);
        enc->setBuffer(pResultsBuf, 0, 6);
        enc->setBuffer(pResultCountBuf, 0, 7);
        enc->setBytes(&partsupp_size, sizeof(partsupp_size), 8);
        enc->setBytes(&max_results, sizeof(max_results), 9);
        enc->dispatchThreads(MTL::Size(partsupp_size, 1, 1), MTL::Size(256, 1, 1));

        enc->endEncoding();
        pCmdBuf->commit();
        pCmdBuf->waitUntilCompleted();

        if (iter == 2) {
            q2_gpu_compute_time = pCmdBuf->GPUEndTime() - pCmdBuf->GPUStartTime();
        }
    }

    // 6. CPU post-processing: join with string columns, sort, limit 100
    auto q2CpuPostStart = std::chrono::high_resolution_clock::now();
    uint result_count = *(uint*)pResultCountBuf->contents();
    if (result_count > max_results) result_count = max_results;
    postProcessQ2((Q2MatchResult_CPU*)pResultsBuf->contents(), result_count,
                  supp_index, s_acctbal.data(), s_nationkey.data(),
                  s_name.data(), s_address.data(), s_phone.data(), s_comment.data(),
                  nation_names, p_partkey.data(), part_size, p_mfgr.data());
    auto q2CpuPostEnd = std::chrono::high_resolution_clock::now();
    double q2CpuPostMs = std::chrono::duration<double, std::milli>(q2CpuPostEnd - q2CpuPostStart).count();

    double q2GpuMs = q2_gpu_compute_time * 1000.0;
    printf("\nQ2 | %u rows (partsupp)\n", partsupp_size);
    printTimingSummary(q2CpuParseMs, q2GpuMs, q2CpuPostMs);

    releaseAll(pFilterPartPipe, pMinCostPipe, pMatchPipe,
              pPartKeyBuf, pPartSizeBuf, pPartTypeBuf,
              pPartBitmapBuf, pSuppBitmapBuf,
              pPsPartKeyBuf, pPsSuppKeyBuf, pPsSupplyCostBuf,
              pMinCostBuf, pResultsBuf, pResultCountBuf);
}


// --- SF100 ---
void runQ2BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q2 Benchmark (SF100) ===" << std::endl;

    MappedFile partFile, suppFile, psFile, natFile, regFile;
    if (!partFile.open(g_dataset_path + "part.tbl") ||
        !suppFile.open(g_dataset_path + "supplier.tbl") ||
        !psFile.open(g_dataset_path + "partsupp.tbl") ||
        !natFile.open(g_dataset_path + "nation.tbl") ||
        !regFile.open(g_dataset_path + "region.tbl")) {
        std::cerr << "Q2 SF100: Cannot open required TBL files" << std::endl;
        return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto partIdx = buildLineIndex(partFile);
    auto suppIdx = buildLineIndex(suppFile);
    auto psIdx = buildLineIndex(psFile);
    auto natIdx = buildLineIndex(natFile);
    auto regIdx = buildLineIndex(regFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    size_t partRows = partIdx.size(), suppRows = suppIdx.size(), psRows = psIdx.size();
    printf("Q2 SF100: part=%zu, supplier=%zu, partsupp=%zu (index %.1f ms)\n", partRows, suppRows, psRows, indexBuildMs);

    // Load nation/region (tiny)
    auto bpT0 = std::chrono::high_resolution_clock::now();
    std::vector<int> n_nationkey, n_regionkey;
    std::vector<char> n_name_chars;
    std::vector<int> r_regionkey;
    std::vector<char> r_name_chars;
    parseNationRegionSF100(natFile, natIdx, n_nationkey, n_regionkey, n_name_chars,
                           &regFile, &regIdx, &r_regionkey, &r_name_chars);

    // Identify EUROPE nations
    int europe_regionkey = findRegionKey(r_regionkey, r_name_chars.data(), 25, "EUROPE");
    if (europe_regionkey == -1) { std::cerr << "EUROPE not found" << std::endl; return; }

    auto nation_names = buildNationNames(n_nationkey, n_name_chars.data(), 25);
    auto europe_nation_keys = filterNationsByRegion(n_nationkey, n_regionkey, europe_regionkey);

    // Load supplier
    std::vector<int> s_suppkey(suppRows), s_nationkey(suppRows);
    std::vector<float> s_acctbal(suppRows);
    std::vector<char> s_name_chars(suppRows * 25), s_address_chars(suppRows * 40);
    std::vector<char> s_phone_chars(suppRows * 15), s_comment_chars(suppRows * 101);
    parseIntColumnChunk(suppFile, suppIdx, 0, suppRows, 0, s_suppkey.data());
    parseCharColumnChunkFixed(suppFile, suppIdx, 0, suppRows, 1, 25, s_name_chars.data());
    parseCharColumnChunkFixed(suppFile, suppIdx, 0, suppRows, 2, 40, s_address_chars.data());
    parseIntColumnChunk(suppFile, suppIdx, 0, suppRows, 3, s_nationkey.data());
    parseCharColumnChunkFixed(suppFile, suppIdx, 0, suppRows, 4, 15, s_phone_chars.data());
    parseFloatColumnChunk(suppFile, suppIdx, 0, suppRows, 5, s_acctbal.data());
    parseCharColumnChunkFixed(suppFile, suppIdx, 0, suppRows, 6, 101, s_comment_chars.data());

    // Build supplier bitmap
    auto suppBitmap = buildSuppBitmapAndIndex(s_suppkey.data(), s_nationkey.data(),
                                              suppRows, europe_nation_keys);
    auto& cpu_supp_bitmap = suppBitmap.bitmap;
    auto supp_bitmap_ints = suppBitmap.bitmap_ints;
    auto& supp_index = suppBitmap.index;

    // Load part columns
    std::vector<int> p_partkey(partRows), p_size(partRows);
    std::vector<char> p_type_chars(partRows * 25), p_mfgr_chars(partRows * 25);
    parseIntColumnChunk(partFile, partIdx, 0, partRows, 0, p_partkey.data());
    parseIntColumnChunk(partFile, partIdx, 0, partRows, 5, p_size.data());
    parseCharColumnChunkFixed(partFile, partIdx, 0, partRows, 4, 25, p_type_chars.data());
    parseCharColumnChunkFixed(partFile, partIdx, 0, partRows, 2, 25, p_mfgr_chars.data());

    // Load partsupp columns
    std::vector<int> ps_partkey(psRows), ps_suppkey(psRows);
    std::vector<float> ps_supplycost(psRows);
    parseIntColumnChunk(psFile, psIdx, 0, psRows, 0, ps_partkey.data());
    parseIntColumnChunk(psFile, psIdx, 0, psRows, 1, ps_suppkey.data());
    parseFloatColumnChunk(psFile, psIdx, 0, psRows, 3, ps_supplycost.data());
    auto bpT1 = std::chrono::high_resolution_clock::now();
    double cpuParseMs = indexBuildMs + std::chrono::duration<double, std::milli>(bpT1 - bpT0).count();

    // Setup GPU kernels (same as standard Q2)
    auto pFilterPartPipe = createPipeline(device, library, "q2_filter_part_kernel");
    auto pMinCostPipe = createPipeline(device, library, "q2_find_min_cost_kernel");
    auto pMatchPipe = createPipeline(device, library, "q2_match_suppliers_kernel");
    if (!pFilterPartPipe || !pMinCostPipe || !pMatchPipe) return;

    // Create GPU buffers
    int max_partkey = 0;
    for (size_t i = 0; i < partRows; i++) max_partkey = std::max(max_partkey, p_partkey[i]);
    const uint part_bitmap_ints = (max_partkey + 31) / 32 + 1;
    const uint part_size = (uint)partRows, partsupp_size = (uint)psRows;

    MTL::Buffer* pPartKeyBuf = device->newBuffer(p_partkey.data(), partRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartSizeBuf = device->newBuffer(p_size.data(), partRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartTypeBuf = device->newBuffer(p_type_chars.data(), p_type_chars.size(), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartBitmapBuf = device->newBuffer(part_bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSuppBitmapBuf = device->newBuffer(cpu_supp_bitmap.data(), supp_bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsPartKeyBuf = device->newBuffer(ps_partkey.data(), psRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsSuppKeyBuf = device->newBuffer(ps_suppkey.data(), psRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsSupplyCostBuf = device->newBuffer(ps_supplycost.data(), psRows * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pMinCostBuf = device->newBuffer((max_partkey + 1) * sizeof(uint), MTL::ResourceStorageModeShared);
    const uint max_results = 100000;
    MTL::Buffer* pResultsBuf = device->newBuffer(max_results * sizeof(Q2MatchResult_CPU), MTL::ResourceStorageModeShared);
    MTL::Buffer* pResultCountBuf = device->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared);
    const int target_size = 15;

    // Execute GPU pipeline (2 warmup + 1 measured)
    double q2_gpu_time = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        std::memset(pPartBitmapBuf->contents(), 0, part_bitmap_ints * sizeof(uint));
        std::memset(pMinCostBuf->contents(), 0xFF, (max_partkey + 1) * sizeof(uint));
        *(uint*)pResultCountBuf->contents() = 0;

        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

        enc->setComputePipelineState(pFilterPartPipe);
        enc->setBuffer(pPartKeyBuf, 0, 0); enc->setBuffer(pPartSizeBuf, 0, 1);
        enc->setBuffer(pPartTypeBuf, 0, 2); enc->setBuffer(pPartBitmapBuf, 0, 3);
        enc->setBytes(&part_size, sizeof(part_size), 4);
        enc->setBytes(&target_size, sizeof(target_size), 5);
        enc->dispatchThreads(MTL::Size(part_size, 1, 1), MTL::Size(256, 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);
        enc->setComputePipelineState(pMinCostPipe);
        enc->setBuffer(pPsPartKeyBuf, 0, 0); enc->setBuffer(pPsSuppKeyBuf, 0, 1);
        enc->setBuffer(pPsSupplyCostBuf, 0, 2); enc->setBuffer(pPartBitmapBuf, 0, 3);
        enc->setBuffer(pSuppBitmapBuf, 0, 4); enc->setBuffer(pMinCostBuf, 0, 5);
        enc->setBytes(&partsupp_size, sizeof(partsupp_size), 6);
        enc->dispatchThreads(MTL::Size(partsupp_size, 1, 1), MTL::Size(256, 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);
        enc->setComputePipelineState(pMatchPipe);
        enc->setBuffer(pPsPartKeyBuf, 0, 0); enc->setBuffer(pPsSuppKeyBuf, 0, 1);
        enc->setBuffer(pPsSupplyCostBuf, 0, 2); enc->setBuffer(pPartBitmapBuf, 0, 3);
        enc->setBuffer(pSuppBitmapBuf, 0, 4); enc->setBuffer(pMinCostBuf, 0, 5);
        enc->setBuffer(pResultsBuf, 0, 6); enc->setBuffer(pResultCountBuf, 0, 7);
        enc->setBytes(&partsupp_size, sizeof(partsupp_size), 8);
        enc->setBytes(&max_results, sizeof(max_results), 9);
        enc->dispatchThreads(MTL::Size(partsupp_size, 1, 1), MTL::Size(256, 1, 1));

        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
        if (iter == 2) q2_gpu_time = cb->GPUEndTime() - cb->GPUStartTime();
    }

    // CPU post-processing
    auto postT0 = std::chrono::high_resolution_clock::now();
    uint result_count = std::min(*(uint*)pResultCountBuf->contents(), max_results);
    postProcessQ2((Q2MatchResult_CPU*)pResultsBuf->contents(), result_count,
                  supp_index, s_acctbal.data(), s_nationkey.data(),
                  s_name_chars.data(), s_address_chars.data(), s_phone_chars.data(), s_comment_chars.data(),
                  nation_names, p_partkey.data(), partRows, p_mfgr_chars.data());
    auto postT1 = std::chrono::high_resolution_clock::now();
    double cpuPostMs = std::chrono::duration<double, std::milli>(postT1 - postT0).count();

    double gpuMs = q2_gpu_time * 1000.0;
    printf("\nSF100 Q2 | %zu rows (partsupp)\n", psRows);
    printTimingSummary(cpuParseMs, gpuMs, cpuPostMs);

    releaseAll(pFilterPartPipe, pMinCostPipe, pMatchPipe,
              pPartKeyBuf, pPartSizeBuf, pPartTypeBuf,
              pPartBitmapBuf, pSuppBitmapBuf,
              pPsPartKeyBuf, pPsSuppKeyBuf, pPsSupplyCostBuf,
              pMinCostBuf, pResultsBuf, pResultCountBuf);
}
