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
    auto q9ParseEnd = std::chrono::high_resolution_clock::now();
    double q9CpuParseMs = std::chrono::duration<double, std::milli>(q9ParseEnd - q9ParseStart).count();

    // Create a map for nation names
    auto nation_names = buildNationNames(nat.nationkey, nat.name.data(), NationData::NAME_WIDTH);
    
    // Get sizes
    const uint part_size = (uint)p_partkey.size(), supplier_size = (uint)s_suppkey.size(), lineitem_size = (uint)l_partkey.size();
    const uint partsupp_size = (uint)ps_partkey.size(), orders_size = (uint)o_orderkey.size();
    std::cout << "Loaded data for all tables." << std::endl;
    std::cout << "Part size: " << part_size << ", Supplier size: " << supplier_size << ", Lineitem size: " << lineitem_size << std::endl;

    // -------------------------------------------------------------------
    // CPU-SIDE BUILD PHASE (eliminates 3 GPU build kernels: part_bitmap,
    // supp_nation_map, orders_year_map). Mirrors codegen's CPU preprocessing.
    // The data is already in CPU memory from loadColumnsMulti so this is
    // essentially free (no extra I/O).
    // -------------------------------------------------------------------
    int max_partkey = 0;
    for (int k : p_partkey) max_partkey = std::max(max_partkey, k);
    int max_suppkey = 0;
    for (int k : s_suppkey) max_suppkey = std::max(max_suppkey, k);
    int max_orderkey = 0;
    for (int k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    std::cout << "Max PartKey: " << max_partkey
              << ", Max SuppKey: " << max_suppkey
              << ", Max OrderKey: " << max_orderkey << std::endl;

    // Build part bitmap (LIKE '%green%') on CPU. Same scan that previously
    // counted green_count + the GPU q9_build_part_ht_kernel.
    const uint bitmap_words = (uint)((max_partkey + 31) / 32 + 1);
    std::vector<uint32_t> cpu_part_bitmap(bitmap_words, 0u);
    int green_count = 0;
    for (size_t i = 0; i < part_size; ++i) {
        const char* nm = p_name.data() + i * 55;
        bool match = false;
        for (int j = 0; j <= 50; ++j) {
            if (nm[j]=='g' && nm[j+1]=='r' && nm[j+2]=='e' && nm[j+3]=='e' && nm[j+4]=='n') {
                match = true; break;
            }
        }
        if (match) {
            int pk = p_partkey[i];
            cpu_part_bitmap[pk >> 5] |= (1u << (pk & 31));
            ++green_count;
        }
    }
    std::cout << "Found " << green_count << " parts with 'green' in name (CPU check)." << std::endl;

    // Build supp_nation_map[suppkey] = nationkey on CPU.
    const uint supp_map_size = (uint)max_suppkey + 1u;
    std::vector<int> cpu_supp_map(supp_map_size, -1);
    for (size_t i = 0; i < supplier_size; ++i) {
        cpu_supp_map[s_suppkey[i]] = s_nationkey[i];
    }

    // Build orders_year_map[orderkey] = year on CPU.
    const uint orders_map_size = (uint)max_orderkey + 1u;
    std::vector<int> cpu_orders_year_map(orders_map_size, -1);
    for (size_t i = 0; i < orders_size; ++i) {
        cpu_orders_year_map[o_orderkey[i]] = o_orderdate[i] / 10000;
    }

    // 2. Setup remaining kernel pipelines (only partsupp build + probe).
    auto pPartSuppBuildPipe = createPipeline(pDevice, pLibrary, "q9_build_partsupp_ht_kernel");
    auto pProbeAggPipe      = createPipeline(pDevice, pLibrary, "q9_probe_directorders_dense_kernel");
    if (!pPartSuppBuildPipe || !pProbeAggPipe) return;

    // 3. Create GPU buffers. Bitmap / supp_map / orders_year_map are populated
    // directly from the CPU-built vectors (single shared-memory upload).
    MTL::Buffer* pPartBitmapBuffer = pDevice->newBuffer(
        cpu_part_bitmap.data(), bitmap_words * sizeof(uint32_t),
        MTL::ResourceStorageModeShared);

    MTL::Buffer* pSuppMapBuffer = pDevice->newBuffer(
        cpu_supp_map.data(), supp_map_size * sizeof(int),
        MTL::ResourceStorageModeShared);

    const uint partsupp_ht_size = nextPow2(partsupp_size); // power-of-2 for bitwise AND probing
    // PartSuppEntry has 4 ints (partkey, suppkey, idx, pad); initialize all to -1 to mark empty
    std::vector<int> cpu_partsupp_ht(partsupp_ht_size * 4, -1);
    MTL::Buffer* pPsPartKeyBuffer = pDevice->newBuffer(ps_partkey.data(), partsupp_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsSuppKeyBuffer = pDevice->newBuffer(ps_suppkey.data(), partsupp_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsSupplyCostBuffer = pDevice->newBuffer(ps_supplycost.data(), partsupp_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartSuppHTBuffer = pDevice->newBuffer(cpu_partsupp_ht.data(), partsupp_ht_size * sizeof(int) * 4, MTL::ResourceStorageModeShared);

    MTL::Buffer* pOrdersHTBuffer = pDevice->newBuffer(
        cpu_orders_year_map.data(), orders_map_size * sizeof(int),
        MTL::ResourceStorageModeShared);

    MTL::Buffer* pLinePartKeyBuffer = pDevice->newBuffer(l_partkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineSuppKeyBuffer = pDevice->newBuffer(l_suppkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineOrdKeyBuffer = pDevice->newBuffer(l_orderkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineQtyBuffer = pDevice->newBuffer(l_quantity.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLinePriceBuffer = pDevice->newBuffer(l_extendedprice.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineDiscBuffer = pDevice->newBuffer(l_discount.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);

    const uint num_threadgroups = 2048;
    // Dense profit array: 25 nations * 8 year-slots = 200 floats. Indexed by
    // (nationkey * 8 + (year - 1992)) inside q9_probe_directorders_dense_kernel.
    // Replaces the 256-slot CAS hashtable previously used for global aggregation.
    static constexpr uint kQ9DenseSlots = 25u * 8u;
    MTL::Buffer* pFinalHTBuffer = pDevice->newBuffer(kQ9DenseSlots * sizeof(float), MTL::ResourceStorageModeShared);

    // 4. Dispatch only partsupp build + probe (2 warmup + 1 measured).
    double q9_gpu_compute_time = 0.0;
    
    for(int iter = 0; iter < 3; ++iter) {
        // Reset only the GPU-built/aggregated buffers between iters.
        std::memset(pPartSuppHTBuffer->contents(), 0xFF, partsupp_ht_size * sizeof(int) * 4);
        std::memset(pFinalHTBuffer->contents(), 0, kQ9DenseSlots * sizeof(float));

        MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();

        // Encoder 1: PartSupp HT build (kept on GPU — largest table, scales with SF)
        MTL::ComputeCommandEncoder* pBuildEnc = pCommandBuffer->computeCommandEncoder();
        pBuildEnc->setComputePipelineState(pPartSuppBuildPipe);
        pBuildEnc->setBuffer(pPsPartKeyBuffer, 0, 0); pBuildEnc->setBuffer(pPsSuppKeyBuffer, 0, 1);
        pBuildEnc->setBuffer(pPartSuppHTBuffer, 0, 2); pBuildEnc->setBytes(&partsupp_size, sizeof(partsupp_size), 3);
        pBuildEnc->setBytes(&partsupp_ht_size, sizeof(partsupp_ht_size), 4);
        pBuildEnc->setBuffer(pPartBitmapBuffer, 0, 5); // bitmap pre-filter for green parts (CPU-built)
        {
            NS::UInteger threadGroupSize = pPartSuppBuildPipe->maxTotalThreadsPerThreadgroup();
            if (threadGroupSize > 256) threadGroupSize = 256;
            MTL::Size threadgroupSize = MTL::Size(threadGroupSize, 1, 1);
            MTL::Size threadgroups = MTL::Size((partsupp_size + threadGroupSize - 1) / threadGroupSize, 1, 1);
            pBuildEnc->dispatchThreadgroups(threadgroups, threadgroupSize);
        }
        pBuildEnc->endEncoding();

        // Encoder 2: Probe & Direct Global Aggregation (single kernel, no merge needed)
        MTL::ComputeCommandEncoder* pProbeEnc = pCommandBuffer->computeCommandEncoder();
        
        // Probe + dense global aggregation. Uses non-atomic RO reads on partsupp HT
        // (cached in L2) + direct array lookup for orders + dense [25*8] float
        // accumulator indexed by (nationkey, year).
        pProbeEnc->setComputePipelineState(pProbeAggPipe);
        pProbeEnc->setBuffer(pLineSuppKeyBuffer, 0, 0); pProbeEnc->setBuffer(pLinePartKeyBuffer, 0, 1);
        pProbeEnc->setBuffer(pLineOrdKeyBuffer, 0, 2); pProbeEnc->setBuffer(pLinePriceBuffer, 0, 3);
        pProbeEnc->setBuffer(pLineDiscBuffer, 0, 4); pProbeEnc->setBuffer(pLineQtyBuffer, 0, 5);
        pProbeEnc->setBuffer(pPsSupplyCostBuffer, 0, 6);
        pProbeEnc->setBuffer(pPartBitmapBuffer, 0, 7);
        pProbeEnc->setBuffer(pSuppMapBuffer, 0, 8);
        pProbeEnc->setBuffer(pPartSuppHTBuffer, 0, 9);
        pProbeEnc->setBuffer(pOrdersHTBuffer, 0, 10);
        pProbeEnc->setBuffer(pFinalHTBuffer, 0, 11);
        pProbeEnc->setBytes(&lineitem_size, sizeof(lineitem_size), 12);
        pProbeEnc->setBytes(&partsupp_ht_size, sizeof(partsupp_ht_size), 13);
        pProbeEnc->setBytes(&orders_map_size, sizeof(orders_map_size), 14);
        pProbeEnc->dispatchThreadgroups(MTL::Size(num_threadgroups, 1, 1), MTL::Size(1024, 1, 1));
        
        pProbeEnc->endEncoding();

        // Execute and time total Q9
        pCommandBuffer->commit();
        pCommandBuffer->waitUntilCompleted();
        
        if (iter == 2) {
            q9_gpu_compute_time = pCommandBuffer->GPUEndTime() - pCommandBuffer->GPUStartTime();
        }
    }

    // 6. CPU post-processing: read dense [25*8] profit array and emit results.
    auto q9_cpu_post_start = std::chrono::high_resolution_clock::now();
    {
        const float* profits = (const float*)pFinalHTBuffer->contents();
        struct Q9Row { int nation; int year; float profit; };
        std::vector<Q9Row> rows;
        rows.reserve(kQ9DenseSlots);
        for (uint slot = 0; slot < kQ9DenseSlots; ++slot) {
            float p = profits[slot];
            if (p == 0.0f) continue;
            int nation = (int)(slot / 8u);
            int year = 1992 + (int)(slot % 8u);
            rows.push_back({nation, year, p});
        }
        std::sort(rows.begin(), rows.end(), [](const Q9Row& a, const Q9Row& b) {
            if (a.nation != b.nation) return a.nation < b.nation;
            return a.year > b.year;
        });
        printf("\nTPC-H Query 9 Results (Top 15):\n");
        printf("+------------+------+---------------+\n");
        printf("| Nation     | Year |        Profit |\n");
        printf("+------------+------+---------------+\n");
        for (size_t i = 0; i < 15 && i < rows.size(); ++i) {
            printf("| %-10s | %4d | $%13.2f |\n",
                   nation_names[rows[i].nation].c_str(), rows[i].year, rows[i].profit);
        }
        printf("+------------+------+---------------+\n");
        printf("Total results found: %lu\n", rows.size());
        std::map<int, double> year_totals;
        for (const auto& r : rows) year_totals[r.year] += (double)r.profit;
        printf("\nComparable TPC-H Q9 (yearly sum_profit):\n");
        printf("+--------+---------------+\n");
        printf("| o_year |   sum_profit  |\n");
        printf("+--------+---------------+\n");
        for (const auto& kv : year_totals) printf("| %6d | %13.4f |\n", kv.first, kv.second);
        printf("+--------+---------------+\n");
    }
    auto q9_cpu_post_end = std::chrono::high_resolution_clock::now();
    double q9_cpu_ms = std::chrono::duration<double, std::milli>(q9_cpu_post_end - q9_cpu_post_start).count();

    double q9GpuMs = q9_gpu_compute_time * 1000.0;
    printf("\nQ9 | %u rows (lineitem)\n", lineitem_size);
    printTimingSummary(q9CpuParseMs, q9GpuMs, q9_cpu_ms);
    
    // Release all
    releaseAll(pPartSuppBuildPipe, pProbeAggPipe,
              pPartBitmapBuffer,
              pSuppMapBuffer,
              pPsPartKeyBuffer, pPsSuppKeyBuffer, pPsSupplyCostBuffer, pPartSuppHTBuffer,
              pOrdersHTBuffer,
              pLinePartKeyBuffer, pLineSuppKeyBuffer, pLineOrdKeyBuffer,
              pLineQtyBuffer, pLinePriceBuffer, pLineDiscBuffer,
              pFinalHTBuffer);
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
    auto pOrdersBuildPipe = createPipeline(device, library, "q9_build_orders_direct_kernel");
    auto pProbeAggPipe = createPipeline(device, library, "q9_probe_directorders_kernel");
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

    // Count green parts from bitmap for right-sizing partsupp HT
    uint* bitmap_data = (uint*)pPartBitmapBuf->contents();
    uint green_parts = 0;
    for (uint i = 0; i < part_bitmap_ints; i++) green_parts += __builtin_popcount(bitmap_data[i]);
    printf("  Part bitmap built (%u ints, max_partkey=%d, green_parts=%u)\n", part_bitmap_ints, max_partkey, green_parts);

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
    printf("  Supplier direct map built (size=%u, gpu=%.2f ms)\n", supp_map_size, totalGpuMs);

    // --- 1c. Orders Direct Map ---
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<int> o_orderkey(ordRows), o_orderdate(ordRows);
    parseIntColumnChunk(ordFile, ordIdx, 0, ordRows, 0, o_orderkey.data());
    parseDateColumnChunk(ordFile, ordIdx, 0, ordRows, 4, o_orderdate.data());
    t1 = std::chrono::high_resolution_clock::now();
    totalCpuParseMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

    int max_orderkey = 0;
    for (size_t i = 0; i < ordRows; i++) max_orderkey = std::max(max_orderkey, o_orderkey[i]);
    const uint orders_map_size = (uint)max_orderkey + 1;
    const uint orders_ht_size = orders_map_size; // for parameter compatibility
    MTL::Buffer* pOrdKeyBuf = device->newBuffer(o_orderkey.data(), ordRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdDateBuf = device->newBuffer(o_orderdate.data(), ordRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdersHTBuf = device->newBuffer(orders_map_size * sizeof(int), MTL::ResourceStorageModeShared);
    memset(pOrdersHTBuf->contents(), 0xFF, orders_map_size * sizeof(int)); // -1 = unset

    {
        uint os = (uint)ordRows;
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pOrdersBuildPipe);
        enc->setBuffer(pOrdKeyBuf, 0, 0);
        enc->setBuffer(pOrdDateBuf, 0, 1);
        enc->setBuffer(pOrdersHTBuf, 0, 2);
        enc->setBytes(&os, sizeof(os), 3);
        enc->dispatchThreads(MTL::Size(ordRows, 1, 1), MTL::Size(256, 1, 1));
        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
        totalGpuMs += (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
    }
    { std::vector<int>().swap(o_orderkey); std::vector<int>().swap(o_orderdate); }
    pOrdKeyBuf->release(); pOrdDateBuf->release();
    printf("  Orders direct map built (size=%u, %.1f MB, cumul gpu=%.2f ms)\n",
           orders_map_size, orders_map_size * 4.0 / (1024*1024), totalGpuMs);

    // --- 1d. PartSupp Hash Table ---
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<int> ps_partkey(psRows), ps_suppkey(psRows);
    std::vector<float> ps_supplycost(psRows);
    parseIntColumnChunk(psFile, psIdx, 0, psRows, 0, ps_partkey.data());
    parseIntColumnChunk(psFile, psIdx, 0, psRows, 1, ps_suppkey.data());
    parseFloatColumnChunk(psFile, psIdx, 0, psRows, 3, ps_supplycost.data());
    t1 = std::chrono::high_resolution_clock::now();
    totalCpuParseMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

    const uint partsupp_ht_size = nextPow2(std::max(green_parts * 4, (uint)1024) * 2); // sized for filtered rows only
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
    printf("  PartSupp HT built (ht_size=%u, cumul gpu=%.2f ms)\n", partsupp_ht_size, totalGpuMs);

    printf("Phase 1 complete (total Phase 1 GPU: %.2f ms). GPU HTs resident: bitmap(%.1f MB) + suppmap(%.1f MB) + orders_map(%.1f MB) + partsupp_ht(%.1f MB)\n",
           totalGpuMs,
           part_bitmap_ints * 4.0 / (1024*1024), supp_map_size * 4.0 / (1024*1024),
           orders_map_size * 4.0 / (1024*1024), partsupp_ht_size * 16.0 / (1024*1024));

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
            // Dispatch one thread per BATCH of 4 lineitem rows, threadgroup size 256
            uint totalThreads = (chunkSize + 3) / 4;
            enc->dispatchThreads(MTL::Size(totalThreads, 1, 1), MTL::Size(256, 1, 1));
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
    printf("  Phase 1 GPU (builds): %.2f ms | Phase 2 GPU (probe): %.2f ms\n", totalGpuMs, timing.gpuMs);
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
