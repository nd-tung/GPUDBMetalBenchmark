#include "infra.h"

// ===================================================================
// TPC-H Q9 — Product Type Profit Measure
// ===================================================================

// --- Standard (SF1/SF10) ---
void runQ9Benchmark(MTL::Device* pDevice, MTL::CommandQueue* pCommandQueue, MTL::Library* pLibrary) {
    std::cout << "\n--- Running TPC-H Query 9 Benchmark ---" << std::endl;

    const std::string sf_path = g_dataset_path;
    
    // 1. Load data for all SIX tables
    auto q9ParseStart = std::chrono::high_resolution_clock::now();
    auto pCols = loadColumnsMulti(sf_path + "part.tbl", {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 55}});
    auto& p_partkey = pCols.ints(0); auto& p_name = pCols.chars(1);
    auto s = loadSupplierBasic(sf_path);
    auto& s_suppkey = s.suppkey;
    auto& s_nationkey = s.nationkey;
    auto lCols = loadColumnsMulti(sf_path + "lineitem.tbl", {{0, ColType::INT}, {1, ColType::INT}, {2, ColType::INT}, {4, ColType::FLOAT}, {5, ColType::FLOAT}, {6, ColType::FLOAT}});
    auto& l_orderkey = lCols.ints(0); auto& l_partkey = lCols.ints(1); auto& l_suppkey = lCols.ints(2);
    auto& l_quantity = lCols.floats(4); auto& l_extendedprice = lCols.floats(5); auto& l_discount = lCols.floats(6);
    auto psCols = loadColumnsMulti(sf_path + "partsupp.tbl", {{0, ColType::INT}, {1, ColType::INT}, {3, ColType::FLOAT}});
    auto& ps_partkey = psCols.ints(0); auto& ps_suppkey = psCols.ints(1); auto& ps_supplycost = psCols.floats(3);
    auto oCols = loadColumnsMulti(sf_path + "orders.tbl", {{0, ColType::INT}, {4, ColType::DATE}});
    auto& o_orderkey = oCols.ints(0); auto& o_orderdate = oCols.ints(4);
    auto nat = loadNation(sf_path);
    auto nation_names = buildNationNames(nat.nationkey, nat.name.data(), NationData::NAME_WIDTH);
    auto q9ParseEnd = std::chrono::high_resolution_clock::now();
    double q9CpuParseMs = std::chrono::duration<double, std::milli>(q9ParseEnd - q9ParseStart).count();

    const uint lineitem_size = (uint)l_partkey.size();
    const uint part_size = (uint)p_partkey.size();
    std::cout << "Loaded data for all tables." << std::endl;

    // 2. CPU pre-build: part bitmap
    int max_partkey = 0;
    for (int k : p_partkey) max_partkey = std::max(max_partkey, k);
    size_t bmpInts = ((size_t)max_partkey + 32) / 32;
    std::vector<uint32_t> partBitmap(bmpInts, 0);
    int green_count = 0;
    for (size_t i = 0; i < part_size; ++i) {
        bool found = false;
        for (int c = 0; c <= 50; ++c) {
            if (p_name[i * 55 + c] == 'g' && p_name[i * 55 + c + 1] == 'r' &&
                p_name[i * 55 + c + 2] == 'e' && p_name[i * 55 + c + 3] == 'e' &&
                p_name[i * 55 + c + 4] == 'n') { found = true; break; }
        }
        if (found) {
            partBitmap[p_partkey[i] / 32] |= (1u << (p_partkey[i] % 32));
            green_count++;
        }
    }
    std::cout << "Found " << green_count << " parts with 'green' in name (CPU check)." << std::endl;

    // 3. CPU pre-build: supplier direct map (suppkey -> nationkey)
    int max_suppkey = 0;
    for (int k : s_suppkey) max_suppkey = std::max(max_suppkey, k);
    std::vector<int> suppNatMap((size_t)max_suppkey + 1, -1);
    for (size_t i = 0; i < s_suppkey.size(); i++)
        suppNatMap[s_suppkey[i]] = s_nationkey[i];

    // 4. CPU pre-build: orders direct map (orderkey -> year)
    int max_orderkey = 0;
    for (int k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    std::vector<int> oYearMap((size_t)max_orderkey + 1, 0);
    for (size_t i = 0; i < o_orderkey.size(); i++)
        oYearMap[o_orderkey[i]] = o_orderdate[i] / 10000;

    // 5. CPU pre-build: partsupp flat HT with bitmap pre-filter
    uint32_t suppMul = (uint32_t)(max_suppkey + 1);
    size_t psEntries = 0;
    for (size_t i = 0; i < ps_partkey.size(); i++) {
        int pk = ps_partkey[i];
        if (pk >= 0 && (size_t)pk / 32 < bmpInts && (partBitmap[pk / 32] >> (pk % 32)) & 1)
            psEntries++;
    }
    uint32_t htSlots = 1;
    while (htSlots < psEntries * 2) htSlots <<= 1;
    uint32_t htMask = htSlots - 1;
    std::vector<uint32_t> htKeys(htSlots, 0xFFFFFFFFu);
    std::vector<float> htVals(htSlots, 0.0f);
    for (size_t i = 0; i < ps_partkey.size(); i++) {
        int pk = ps_partkey[i];
        if (pk < 0 || (size_t)pk / 32 >= bmpInts || !((partBitmap[pk / 32] >> (pk % 32)) & 1))
            continue;
        uint32_t key = (uint32_t)pk * suppMul + (uint32_t)ps_suppkey[i];
        uint32_t h = (key * 2654435769u) & htMask;
        for (uint32_t s = 0; s <= htMask; s++) {
            uint32_t slot = (h + s) & htMask;
            if (htKeys[slot] == 0xFFFFFFFFu) {
                htKeys[slot] = key;
                htVals[slot] = ps_supplycost[i];
                break;
            }
        }
    }
    std::cout << "CPU pre-built: bitmap=" << bmpInts << " ints, suppMap=" << suppNatMap.size()
              << ", yearMap=" << oYearMap.size() << ", psHT=" << htSlots << " slots (" << psEntries << " entries)" << std::endl;

    // 6. Create GPU buffers
    auto pProbePipe = createPipeline(pDevice, pLibrary, "q9_probe_direct_maps");
    if (!pProbePipe) return;

    MTL::Buffer* pLinePartKeyBuf  = pDevice->newBuffer(l_partkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineSuppKeyBuf  = pDevice->newBuffer(l_suppkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineOrdKeyBuf   = pDevice->newBuffer(l_orderkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineQtyBuf      = pDevice->newBuffer(l_quantity.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLinePriceBuf    = pDevice->newBuffer(l_extendedprice.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineDiscBuf     = pDevice->newBuffer(l_discount.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);

    MTL::Buffer* pPartBitmapBuf   = pDevice->newBuffer(partBitmap.data(), bmpInts * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSuppNatMapBuf   = pDevice->newBuffer(suppNatMap.data(), suppNatMap.size() * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOYearMapBuf     = pDevice->newBuffer(oYearMap.data(), oYearMap.size() * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsHtKeysBuf     = pDevice->newBuffer(htKeys.data(), htSlots * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsHtValsBuf     = pDevice->newBuffer(htVals.data(), htSlots * sizeof(float), MTL::ResourceStorageModeShared);

    const uint NUM_PROFIT_BINS = 25 * 8; // 25 nations * 8 years (1992-1999)
    MTL::Buffer* pProfitBinsBuf   = pDevice->newBuffer(NUM_PROFIT_BINS * sizeof(float), MTL::ResourceStorageModeShared);

    // 7. Dispatch: single probe kernel (2 warmup + 1 measured)
    double q9_gpu_compute_time = 0.0;
    const uint num_threadgroups = 2048;

    for (int iter = 0; iter < 3; ++iter) {
        // Only need to reset the profit bins each iteration
        std::memset(pProfitBinsBuf->contents(), 0, NUM_PROFIT_BINS * sizeof(float));

        MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* pEnc = pCommandBuffer->computeCommandEncoder();

        pEnc->setComputePipelineState(pProbePipe);
        pEnc->setBuffer(pLinePartKeyBuf, 0, 0);
        pEnc->setBuffer(pLineSuppKeyBuf, 0, 1);
        pEnc->setBuffer(pLineOrdKeyBuf, 0, 2);
        pEnc->setBuffer(pLineQtyBuf, 0, 3);
        pEnc->setBuffer(pLinePriceBuf, 0, 4);
        pEnc->setBuffer(pLineDiscBuf, 0, 5);
        pEnc->setBuffer(pPartBitmapBuf, 0, 6);
        pEnc->setBuffer(pSuppNatMapBuf, 0, 7);
        pEnc->setBuffer(pOYearMapBuf, 0, 8);
        pEnc->setBuffer(pPsHtKeysBuf, 0, 9);
        pEnc->setBuffer(pPsHtValsBuf, 0, 10);
        pEnc->setBuffer(pProfitBinsBuf, 0, 11);
        pEnc->setBytes(&lineitem_size, sizeof(lineitem_size), 12);
        pEnc->setBytes(&htMask, sizeof(htMask), 13);
        pEnc->setBytes(&suppMul, sizeof(suppMul), 14);
        pEnc->dispatchThreadgroups(MTL::Size(num_threadgroups, 1, 1), MTL::Size(1024, 1, 1));
        pEnc->endEncoding();

        pCommandBuffer->commit();
        pCommandBuffer->waitUntilCompleted();
        if (iter == 2)
            q9_gpu_compute_time = pCommandBuffer->GPUEndTime() - pCommandBuffer->GPUStartTime();
    }

    // 8. CPU post-processing: read direct-mapped profit bins
    auto q9_cpu_post_start = std::chrono::high_resolution_clock::now();
    float* profitBins = (float*)pProfitBinsBuf->contents();
    std::vector<Q9Result> final_results;
    for (int n = 0; n < 25; n++) {
        for (int y = 0; y < 8; y++) {
            float p = profitBins[n * 8 + y];
            if (p != 0.0f) {
                final_results.push_back({n, 1992 + y, p});
            }
        }
    }
    std::sort(final_results.begin(), final_results.end(), [](const Q9Result& a, const Q9Result& b) {
        if (a.nationkey != b.nationkey) return a.nationkey < b.nationkey;
        return a.year > b.year;
    });
    printf("\nTPC-H Query 9 Results (Top 15):\n");
    printf("+------------+------+---------------+\n");
    printf("| Nation     | Year |        Profit |\n");
    printf("+------------+------+---------------+\n");
    for (size_t i = 0; i < 15 && i < final_results.size(); ++i) {
        printf("| %-10s | %4d | $%13.2f |\n",
               nation_names[final_results[i].nationkey].c_str(), final_results[i].year, final_results[i].profit);
    }
    printf("+------------+------+---------------+\n");
    printf("Total results found: %lu\n", final_results.size());
    std::map<int, double> year_totals;
    for (const auto& r : final_results) year_totals[r.year] += (double)r.profit;
    printf("\nComparable TPC-H Q9 (yearly sum_profit):\n");
    printf("+--------+---------------+\n");
    printf("| o_year |   sum_profit  |\n");
    printf("+--------+---------------+\n");
    for (const auto& kv : year_totals) printf("| %6d | %13.4f |\n", kv.first, kv.second);
    printf("+--------+---------------+\n");
    auto q9_cpu_post_end = std::chrono::high_resolution_clock::now();
    double q9_cpu_ms = std::chrono::duration<double, std::milli>(q9_cpu_post_end - q9_cpu_post_start).count();

    double q9GpuMs = q9_gpu_compute_time * 1000.0;
    printf("\nQ9 | %u rows (lineitem)\n", lineitem_size);
    printTimingSummary(q9CpuParseMs, q9GpuMs, q9_cpu_ms);
    
    // Release all
    releaseAll(pProbePipe,
              pLinePartKeyBuf, pLineSuppKeyBuf, pLineOrdKeyBuf,
              pLineQtyBuf, pLinePriceBuf, pLineDiscBuf,
              pPartBitmapBuf, pSuppNatMapBuf, pOYearMapBuf,
              pPsHtKeysBuf, pPsHtValsBuf, pProfitBinsBuf);
}


// --- SF100 Hybrid Streaming ---
void runQ9BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q9 Benchmark (SF100 Hybrid Streaming) ===" << std::endl;

    size_t maxMem = device->recommendedMaxWorkingSetSize();
    printf("GPU max working set: %zu MB\n", maxMem / (1024*1024));

    // ── Open all mmap files ──
    MappedFile partFile, suppFile, psFile, ordFile, natFile, liFile;
    if (!partFile.open(g_dataset_path + "part.tbl") || !suppFile.open(g_dataset_path + "supplier.tbl") ||
        !psFile.open(g_dataset_path + "partsupp.tbl") || !ordFile.open(g_dataset_path + "orders.tbl") ||
        !natFile.open(g_dataset_path + "nation.tbl") || !liFile.open(g_dataset_path + "lineitem.tbl")) {
        std::cerr << "Q9 SF100: Cannot open required files" << std::endl;
        return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto partIdx = buildLineIndex(partFile), suppIdx = buildLineIndex(suppFile);
    auto psIdx = buildLineIndex(psFile), ordIdx = buildLineIndex(ordFile);
    auto natIdx = buildLineIndex(natFile), liIdx = buildLineIndex(liFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    size_t partRows = partIdx.size(), suppRows = suppIdx.size();
    size_t psRows = psIdx.size(), ordRows = ordIdx.size();
    size_t liRows = liIdx.size();
    printf("Q9 SF100: part=%zu, supplier=%zu, partsupp=%zu, orders=%zu, lineitem=%zu (index %.1f ms)\n",
           partRows, suppRows, psRows, ordRows, liRows, indexBuildMs);

    // ── Load nation data (tiny — always fits) ──
    std::vector<int> n_nationkey, n_regionkey;
    std::vector<char> n_name_chars;
    parseNationRegionSF100(natFile, natIdx, n_nationkey, n_regionkey, n_name_chars);
    auto nation_names = buildNationNames(n_nationkey, n_name_chars.data(), 25);

    // ── Setup all kernel pipelines ──
    auto pPartBuildPipe = createPipeline(device, library, "q9_build_part_ht_kernel");
    auto pSuppBuildPipe = createPipeline(device, library, "q9_build_supplier_ht_kernel");
    auto pPartSuppBuildPipe = createPipeline(device, library, "q9_build_partsupp_ht_kernel");
    auto pOrdersBuildPipe = createPipeline(device, library, "q9_build_orders_ht_kernel");
    auto pProbeAggPipe = createPipeline(device, library, "q9_probe_and_global_agg_kernel");
    if (!pPartBuildPipe || !pSuppBuildPipe || !pPartSuppBuildPipe || !pOrdersBuildPipe || !pProbeAggPipe) return;

    // Timing accumulators
    double totalCpuParseMs = 0, totalGpuMs = 0;

    // ══════════════════════════════════════════════════════════════
    // PHASE 1: Build hash tables — load one table at a time,
    //          parse on CPU, upload to GPU, free CPU RAM immediately.
    // ══════════════════════════════════════════════════════════════
    printf("\n--- Phase 1: Build Hash Tables (sequential, memory-conscious) ---\n");

    // --- 1a. Part Bitmap ---
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<int> p_partkey(partRows);
    std::vector<char> p_name(partRows * 55);
    parseIntColumnChunk(partFile, partIdx, 0, partRows, 0, p_partkey.data());
    parseCharColumnChunkFixed(partFile, partIdx, 0, partRows, 1, 55, p_name.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    totalCpuParseMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

    int max_partkey = 0;
    for (size_t i = 0; i < partRows; i++) max_partkey = std::max(max_partkey, p_partkey[i]);
    const uint part_bitmap_ints = (max_partkey + 31) / 32 + 1;
    const uint part_ht_size = 0; // bitmap mode

    MTL::Buffer* pPartKeyBuf = device->newBuffer(p_partkey.data(), partRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartNameBuf = device->newBuffer(p_name.data(), p_name.size(), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartBitmapBuf = createBitmapBuffer(device, max_partkey);

    {
        uint ps = (uint)partRows;
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pPartBuildPipe);
        enc->setBuffer(pPartKeyBuf, 0, 0);
        enc->setBuffer(pPartNameBuf, 0, 1);
        enc->setBuffer(pPartBitmapBuf, 0, 2);
        enc->setBytes(&ps, sizeof(ps), 3);
        NS::UInteger tgs = std::min((NS::UInteger)256, pPartBuildPipe->maxTotalThreadsPerThreadgroup());
        enc->dispatchThreadgroups(MTL::Size((partRows + tgs - 1) / tgs, 1, 1), MTL::Size(tgs, 1, 1));
        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
        totalGpuMs += (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
    }
    // Free CPU-side vectors and GPU input buffers
    { std::vector<int>().swap(p_partkey); std::vector<char>().swap(p_name); }
    pPartKeyBuf->release(); pPartNameBuf->release();
    printf("  Part bitmap built (%u ints, max_partkey=%d)\n", part_bitmap_ints, max_partkey);

    // --- 1b. Supplier Direct Map ---
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<int> s_suppkey(suppRows), s_nationkey(suppRows);
    parseIntColumnChunk(suppFile, suppIdx, 0, suppRows, 0, s_suppkey.data());
    parseIntColumnChunk(suppFile, suppIdx, 0, suppRows, 3, s_nationkey.data());
    t1 = std::chrono::high_resolution_clock::now();
    totalCpuParseMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

    int max_suppkey = 0;
    for (size_t i = 0; i < suppRows; i++) max_suppkey = std::max(max_suppkey, s_suppkey[i]);
    const uint supp_map_size = max_suppkey + 1;
    const uint supplier_ht_size = 0; // direct map mode

    MTL::Buffer* pSuppKeyBuf = device->newBuffer(s_suppkey.data(), suppRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSuppNatBuf = device->newBuffer(s_nationkey.data(), suppRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSuppMapBuf = device->newBuffer(supp_map_size * sizeof(int), MTL::ResourceStorageModeShared);
    memset(pSuppMapBuf->contents(), 0xFF, supp_map_size * sizeof(int)); // -1

    {
        uint ss = (uint)suppRows;
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pSuppBuildPipe);
        enc->setBuffer(pSuppKeyBuf, 0, 0);
        enc->setBuffer(pSuppNatBuf, 0, 1);
        enc->setBuffer(pSuppMapBuf, 0, 2);
        enc->setBytes(&ss, sizeof(ss), 3);
        NS::UInteger tgs = std::min((NS::UInteger)256, pSuppBuildPipe->maxTotalThreadsPerThreadgroup());
        enc->dispatchThreadgroups(MTL::Size((suppRows + tgs - 1) / tgs, 1, 1), MTL::Size(tgs, 1, 1));
        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
        totalGpuMs += (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
    }
    { std::vector<int>().swap(s_suppkey); std::vector<int>().swap(s_nationkey); }
    pSuppKeyBuf->release(); pSuppNatBuf->release();
    printf("  Supplier direct map built (size=%u)\n", supp_map_size);

    // --- 1c. Orders Hash Table ---
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<int> o_orderkey(ordRows), o_orderdate(ordRows);
    parseIntColumnChunk(ordFile, ordIdx, 0, ordRows, 0, o_orderkey.data());
    parseDateColumnChunk(ordFile, ordIdx, 0, ordRows, 4, o_orderdate.data());
    t1 = std::chrono::high_resolution_clock::now();
    totalCpuParseMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

    const uint orders_ht_size = nextPow2((uint)ordRows * 2);
    MTL::Buffer* pOrdKeyBuf = device->newBuffer(o_orderkey.data(), ordRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdDateBuf = device->newBuffer(o_orderdate.data(), ordRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdersHTBuf = device->newBuffer(orders_ht_size * sizeof(int) * 2, MTL::ResourceStorageModeShared);
    memset(pOrdersHTBuf->contents(), 0xFF, orders_ht_size * sizeof(int) * 2);

    {
        uint os = (uint)ordRows;
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pOrdersBuildPipe);
        enc->setBuffer(pOrdKeyBuf, 0, 0);
        enc->setBuffer(pOrdDateBuf, 0, 1);
        enc->setBuffer(pOrdersHTBuf, 0, 2);
        enc->setBytes(&os, sizeof(os), 3);
        enc->setBytes(&orders_ht_size, sizeof(orders_ht_size), 4);
        NS::UInteger tgs = std::min((NS::UInteger)256, pOrdersBuildPipe->maxTotalThreadsPerThreadgroup());
        enc->dispatchThreadgroups(MTL::Size((ordRows + tgs - 1) / tgs, 1, 1), MTL::Size(tgs, 1, 1));
        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
        totalGpuMs += (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
    }
    { std::vector<int>().swap(o_orderkey); std::vector<int>().swap(o_orderdate); }
    pOrdKeyBuf->release(); pOrdDateBuf->release();
    printf("  Orders HT built (ht_size=%u)\n", orders_ht_size);

    // --- 1d. PartSupp Hash Table ---
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<int> ps_partkey(psRows), ps_suppkey(psRows);
    std::vector<float> ps_supplycost(psRows);
    parseIntColumnChunk(psFile, psIdx, 0, psRows, 0, ps_partkey.data());
    parseIntColumnChunk(psFile, psIdx, 0, psRows, 1, ps_suppkey.data());
    parseFloatColumnChunk(psFile, psIdx, 0, psRows, 3, ps_supplycost.data());
    t1 = std::chrono::high_resolution_clock::now();
    totalCpuParseMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

    const uint partsupp_ht_size = nextPow2((uint)psRows * 4);
    MTL::Buffer* pPsPartKeyBuf = device->newBuffer(ps_partkey.data(), psRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsSuppKeyBuf = device->newBuffer(ps_suppkey.data(), psRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsSupplyCostBuf = device->newBuffer(ps_supplycost.data(), psRows * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartSuppHTBuf = device->newBuffer(partsupp_ht_size * sizeof(int) * 4, MTL::ResourceStorageModeShared);
    memset(pPartSuppHTBuf->contents(), 0xFF, partsupp_ht_size * sizeof(int) * 4);

    {
        uint pss = (uint)psRows;
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pPartSuppBuildPipe);
        enc->setBuffer(pPsPartKeyBuf, 0, 0);
        enc->setBuffer(pPsSuppKeyBuf, 0, 1);
        enc->setBuffer(pPartSuppHTBuf, 0, 2);
        enc->setBytes(&pss, sizeof(pss), 3);
        enc->setBytes(&partsupp_ht_size, sizeof(partsupp_ht_size), 4);
        enc->setBuffer(pPartBitmapBuf, 0, 5); // bitmap pre-filter for green parts
        NS::UInteger tgs = std::min((NS::UInteger)256, pPartSuppBuildPipe->maxTotalThreadsPerThreadgroup());
        enc->dispatchThreadgroups(MTL::Size((psRows + tgs - 1) / tgs, 1, 1), MTL::Size(tgs, 1, 1));
        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
        totalGpuMs += (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
    }
    { std::vector<int>().swap(ps_partkey); std::vector<int>().swap(ps_suppkey); }
    // Keep ps_supplycost — needed by probe kernel
    pPsPartKeyBuf->release(); pPsSuppKeyBuf->release();
    printf("  PartSupp HT built (ht_size=%u)\n", partsupp_ht_size);

    printf("Phase 1 complete. GPU HTs resident: bitmap(%.1f MB) + suppmap(%.1f MB) + orders_ht(%.1f MB) + partsupp_ht(%.1f MB)\n",
           part_bitmap_ints * 4.0 / (1024*1024), supp_map_size * 4.0 / (1024*1024),
           orders_ht_size * 8.0 / (1024*1024), partsupp_ht_size * 16.0 / (1024*1024));

    // ══════════════════════════════════════════════════════════════
    // PHASE 2: Stream lineitem in chunks through Probe + Merge
    // ══════════════════════════════════════════════════════════════
    printf("\n--- Phase 2: Stream lineitem (%zu rows) ---\n", liRows);

    // Per-chunk lineitem columns: partkey, suppkey, orderkey, quantity, extprice, discount
    // ~28 bytes/row → use adaptive chunk size
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 28, liRows);
    size_t numChunks = (liRows + chunkRows - 1) / chunkRows;
    printf("Chunk size: %zu rows, %zu chunks\n", chunkRows, numChunks);

    // Allocate double-buffered lineitem chunk GPU buffers
    const int Q9_NUM_SLOTS = 2;
    struct Q9LiSlot {
        MTL::Buffer* partKey; MTL::Buffer* suppKey; MTL::Buffer* ordKey;
        MTL::Buffer* qty; MTL::Buffer* price; MTL::Buffer* disc;
    };
    Q9LiSlot liSlots[Q9_NUM_SLOTS];
    for (int s = 0; s < Q9_NUM_SLOTS; s++) {
        liSlots[s].partKey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        liSlots[s].suppKey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        liSlots[s].ordKey  = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        liSlots[s].qty     = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        liSlots[s].price   = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        liSlots[s].disc    = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
    }

    // Intermediate + final aggregation buffers (persistent across chunks)
    const uint num_threadgroups = 2048;
    const uint final_ht_size = nextPow2(25 * 10); // 25 nations * ~10 years, rounded to power-of-2
    MTL::Buffer* pFinalHTBuf = device->newBuffer(final_ht_size * sizeof(Q9Aggregates_CPU), MTL::ResourceStorageModeShared);
    memset(pFinalHTBuf->contents(), 0, final_ht_size * sizeof(Q9Aggregates_CPU));

    auto timing = chunkedStreamLoop(
        commandQueue, liSlots, Q9_NUM_SLOTS, liRows, chunkRows,
        // Parse
        [&](Q9LiSlot& slot, size_t startRow, size_t rowCount) {
            parseIntColumnChunk(liFile, liIdx, startRow, rowCount, 1, (int*)slot.partKey->contents());
            parseIntColumnChunk(liFile, liIdx, startRow, rowCount, 2, (int*)slot.suppKey->contents());
            parseIntColumnChunk(liFile, liIdx, startRow, rowCount, 0, (int*)slot.ordKey->contents());
            parseFloatColumnChunk(liFile, liIdx, startRow, rowCount, 4, (float*)slot.qty->contents());
            parseFloatColumnChunk(liFile, liIdx, startRow, rowCount, 5, (float*)slot.price->contents());
            parseFloatColumnChunk(liFile, liIdx, startRow, rowCount, 6, (float*)slot.disc->contents());
        },
        // Dispatch (probe + direct global aggregation, single kernel)
        [&](Q9LiSlot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(pProbeAggPipe);
            enc->setBuffer(slot.suppKey, 0, 0);
            enc->setBuffer(slot.partKey, 0, 1);
            enc->setBuffer(slot.ordKey, 0, 2);
            enc->setBuffer(slot.price, 0, 3);
            enc->setBuffer(slot.disc, 0, 4);
            enc->setBuffer(slot.qty, 0, 5);
            enc->setBuffer(pPsSupplyCostBuf, 0, 6);
            enc->setBuffer(pPartBitmapBuf, 0, 7);
            enc->setBuffer(pSuppMapBuf, 0, 8);
            enc->setBuffer(pPartSuppHTBuf, 0, 9);
            enc->setBuffer(pOrdersHTBuf, 0, 10);
            enc->setBuffer(pFinalHTBuf, 0, 11);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 12);
            enc->setBytes(&part_ht_size, sizeof(part_ht_size), 13);
            enc->setBytes(&supplier_ht_size, sizeof(supplier_ht_size), 14);
            enc->setBytes(&partsupp_ht_size, sizeof(partsupp_ht_size), 15);
            enc->setBytes(&orders_ht_size, sizeof(orders_ht_size), 16);
            enc->setBytes(&final_ht_size, sizeof(final_ht_size), 17);
            enc->dispatchThreadgroups(MTL::Size(num_threadgroups, 1, 1), MTL::Size(1024, 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        // Progress
        [&]([[maybe_unused]] uint chunkSize, size_t chunkNum) {
            if ((chunkNum + 1) % 10 == 0 || chunkNum + 1 == numChunks) {
                printf("  Chunk %zu/%zu done\n", chunkNum + 1, numChunks);
            }
        }
    );

    // ══════════════════════════════════════════════════════════════
    // PHASE 3: Read back results, map nation names, sort, print
    // ══════════════════════════════════════════════════════════════
    auto postStart = std::chrono::high_resolution_clock::now();
    postProcessQ9(pFinalHTBuf->contents(), final_ht_size, nation_names);
    auto postEnd = std::chrono::high_resolution_clock::now();
    double cpuPostMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    double allCpuParseMs = indexBuildMs + totalCpuParseMs + timing.parseMs;
    double allGpuMs = totalGpuMs + timing.gpuMs;
    printf("\nSF100 Q9 | %zu chunks | %zu rows\n", timing.chunkCount, liRows);
    printTimingSummary(allCpuParseMs, allGpuMs, cpuPostMs);

    // Cleanup
    releaseAll(pPartBuildPipe, pSuppBuildPipe, pPartSuppBuildPipe, pOrdersBuildPipe, pProbeAggPipe,
              pPartBitmapBuf, pSuppMapBuf, pOrdersHTBuf, pPartSuppHTBuf, pPsSupplyCostBuf);
    for (int s = 0; s < Q9_NUM_SLOTS; s++) {
        releaseAll(liSlots[s].partKey, liSlots[s].suppKey, liSlots[s].ordKey,
                  liSlots[s].qty, liSlots[s].price, liSlots[s].disc);
    }
    releaseAll(pFinalHTBuf);
}
