#include "infra.h"

// ===================================================================
// TPC-H Q11 — Important Stock Identification
// ===================================================================

// --- Standard (SF1/SF10) ---
void runQ11Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n--- Running TPC-H Query 11 Benchmark ---" << std::endl;

    const std::string sf_path = g_dataset_path;

    // 1. Load data
    auto parseStart = std::chrono::high_resolution_clock::now();

    auto n_nationkey = loadIntColumn(sf_path + "nation.tbl", 0);
    auto n_name = loadCharColumn(sf_path + "nation.tbl", 1, 25);

    auto s_suppkey = loadIntColumn(sf_path + "supplier.tbl", 0);
    auto s_nationkey = loadIntColumn(sf_path + "supplier.tbl", 3);

    auto ps_partkey = loadIntColumn(sf_path + "partsupp.tbl", 0);
    auto ps_suppkey = loadIntColumn(sf_path + "partsupp.tbl", 1);
    auto ps_availqty = loadIntColumn(sf_path + "partsupp.tbl", 2);
    auto ps_supplycost = loadFloatColumn(sf_path + "partsupp.tbl", 3);

    auto parseEnd = std::chrono::high_resolution_clock::now();
    double cpuParseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    const uint partsupp_size = (uint)ps_partkey.size();
    const uint supplier_size = (uint)s_suppkey.size();
    std::cout << "Loaded data. PartSupp: " << partsupp_size << ", Supplier: " << supplier_size << std::endl;

    // 2. CPU: Find GERMANY nationkey, build supplier bitmap
    int germany_nationkey = -1;
    for (size_t i = 0; i < n_nationkey.size(); i++) {
        if (trimFixed(n_name.data(), i, 25) == "GERMANY") {
            germany_nationkey = n_nationkey[i];
            break;
        }
    }
    if (germany_nationkey == -1) {
        std::cerr << "Error: GERMANY not found in nation table" << std::endl;
        return;
    }

    std::vector<int> germany_keys = {germany_nationkey};
    auto suppBitmap = buildSuppBitmapAndIndex(s_suppkey.data(), s_nationkey.data(),
                                              supplier_size, germany_keys);
    auto& cpu_supp_bitmap = suppBitmap.bitmap;
    auto supp_bitmap_ints = suppBitmap.bitmap_ints;

    // 3. Setup GPU
    auto pAggregatePipe = createPipeline(device, library, "q11_aggregate_kernel");
    if (!pAggregatePipe) return;

    int max_partkey = 0;
    for (int k : ps_partkey) max_partkey = std::max(max_partkey, k);
    const uint value_map_size = (uint)(max_partkey + 1);

    // Compute number of threadgroups for partial sums
    const uint tg_size = 256;
    const uint num_threadgroups = (partsupp_size + tg_size - 1) / tg_size;

    // 4. Create GPU buffers
    MTL::Buffer* pPsPartKeyBuf = device->newBuffer(ps_partkey.data(), partsupp_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsSuppKeyBuf = device->newBuffer(ps_suppkey.data(), partsupp_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsSupplyCostBuf = device->newBuffer(ps_supplycost.data(), partsupp_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsAvailQtyBuf = device->newBuffer(ps_availqty.data(), partsupp_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSuppBitmapBuf = device->newBuffer(cpu_supp_bitmap.data(), supp_bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* pValueMapBuf = device->newBuffer(value_map_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartialSumsBuf = device->newBuffer(num_threadgroups * sizeof(float), MTL::ResourceStorageModeShared);

    // 5. Execute GPU (2 warmup + 1 measured)
    double gpu_compute_time = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        std::memset(pValueMapBuf->contents(), 0, value_map_size * sizeof(float));
        std::memset(pPartialSumsBuf->contents(), 0, num_threadgroups * sizeof(float));

        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pAggregatePipe);
        enc->setBuffer(pPsPartKeyBuf, 0, 0);
        enc->setBuffer(pPsSuppKeyBuf, 0, 1);
        enc->setBuffer(pPsSupplyCostBuf, 0, 2);
        enc->setBuffer(pPsAvailQtyBuf, 0, 3);
        enc->setBuffer(pSuppBitmapBuf, 0, 4);
        enc->setBuffer(pValueMapBuf, 0, 5);
        enc->setBuffer(pPartialSumsBuf, 0, 6);
        enc->setBytes(&partsupp_size, sizeof(partsupp_size), 7);
        enc->dispatchThreads(MTL::Size(partsupp_size, 1, 1), MTL::Size(tg_size, 1, 1));
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();

        if (iter == 2) {
            gpu_compute_time = cb->GPUEndTime() - cb->GPUStartTime();
        }
    }

    // 6. CPU post-processing: compute global sum, apply threshold, sort
    auto postStart = std::chrono::high_resolution_clock::now();

    float* partial_sums = (float*)pPartialSumsBuf->contents();
    double global_sum = 0.0;
    for (uint i = 0; i < num_threadgroups; i++) {
        global_sum += partial_sums[i];
    }
    double threshold = global_sum * 0.0001;

    float* value_map = (float*)pValueMapBuf->contents();
    struct Q11Result { int partkey; double value; };
    std::vector<Q11Result> results;
    for (uint i = 0; i < value_map_size; i++) {
        if (value_map[i] > threshold) {
            results.push_back({(int)i, (double)value_map[i]});
        }
    }

    std::sort(results.begin(), results.end(), [](const Q11Result& a, const Q11Result& b) {
        return a.value > b.value;
    });

    printf("\nTPC-H Query 11 Results (Top 20 of %zu):\n", results.size());
    printf("+-----------+------------------+\n");
    printf("| ps_partkey|            value |\n");
    printf("+-----------+------------------+\n");
    size_t limit = std::min(results.size(), (size_t)20);
    for (size_t i = 0; i < limit; i++) {
        printf("| %9d | %16.2f |\n", results[i].partkey, results[i].value);
    }
    printf("+-----------+------------------+\n");
    printf("Total qualifying rows: %zu, Global sum: %.2f, Threshold: %.2f\n",
           results.size(), global_sum, threshold);

    auto postEnd = std::chrono::high_resolution_clock::now();
    double cpuPostMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    double gpuMs = gpu_compute_time * 1000.0;
    printf("\nQ11 | %u rows (partsupp)\n", partsupp_size);
    printTimingSummary(cpuParseMs, gpuMs, cpuPostMs);

    releaseAll(pAggregatePipe, pPsPartKeyBuf, pPsSuppKeyBuf, pPsSupplyCostBuf,
              pPsAvailQtyBuf, pSuppBitmapBuf, pValueMapBuf, pPartialSumsBuf);
}


// --- SF100 ---
void runQ11BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q11 Benchmark (SF100) ===" << std::endl;

    MappedFile psFile, suppFile, natFile;
    if (!psFile.open(g_dataset_path + "partsupp.tbl") ||
        !suppFile.open(g_dataset_path + "supplier.tbl") ||
        !natFile.open(g_dataset_path + "nation.tbl")) {
        std::cerr << "Q11 SF100: Cannot open required TBL files" << std::endl;
        return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto psIdx = buildLineIndex(psFile);
    auto suppIdx = buildLineIndex(suppFile);
    auto natIdx = buildLineIndex(natFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    size_t psRows = psIdx.size(), suppRows = suppIdx.size();
    printf("Q11 SF100: partsupp=%zu, supplier=%zu (index %.1f ms)\n", psRows, suppRows, indexBuildMs);

    // Load nation (tiny)
    auto bpT0 = std::chrono::high_resolution_clock::now();
    std::vector<int> n_nationkey, n_regionkey;
    std::vector<char> n_name_chars;
    parseNationRegionSF100(natFile, natIdx, n_nationkey, n_regionkey, n_name_chars);

    // Find GERMANY nationkey
    int germany_nationkey = -1;
    for (size_t i = 0; i < n_nationkey.size(); i++) {
        if (trimFixed(n_name_chars.data(), i, 25) == "GERMANY") {
            germany_nationkey = n_nationkey[i];
            break;
        }
    }
    if (germany_nationkey == -1) {
        std::cerr << "Error: GERMANY not found" << std::endl;
        return;
    }

    // Load supplier
    std::vector<int> s_suppkey(suppRows), s_nationkey(suppRows);
    parseIntColumnChunk(suppFile, suppIdx, 0, suppRows, 0, s_suppkey.data());
    parseIntColumnChunk(suppFile, suppIdx, 0, suppRows, 3, s_nationkey.data());

    // Build supplier bitmap for GERMANY
    std::vector<int> germany_keys = {germany_nationkey};
    auto suppBitmap = buildSuppBitmapAndIndex(s_suppkey.data(), s_nationkey.data(),
                                              suppRows, germany_keys);
    auto& cpu_supp_bitmap = suppBitmap.bitmap;
    auto supp_bitmap_ints = suppBitmap.bitmap_ints;

    // Load partsupp columns
    std::vector<int> ps_partkey(psRows), ps_suppkey(psRows), ps_availqty(psRows);
    std::vector<float> ps_supplycost(psRows);
    parseIntColumnChunk(psFile, psIdx, 0, psRows, 0, ps_partkey.data());
    parseIntColumnChunk(psFile, psIdx, 0, psRows, 1, ps_suppkey.data());
    parseIntColumnChunk(psFile, psIdx, 0, psRows, 2, ps_availqty.data());
    parseFloatColumnChunk(psFile, psIdx, 0, psRows, 3, ps_supplycost.data());

    auto bpT1 = std::chrono::high_resolution_clock::now();
    double cpuParseMs = indexBuildMs + std::chrono::duration<double, std::milli>(bpT1 - bpT0).count();

    // Setup GPU
    auto pAggregatePipe = createPipeline(device, library, "q11_aggregate_kernel");
    if (!pAggregatePipe) return;

    int max_partkey = 0;
    for (size_t i = 0; i < psRows; i++) max_partkey = std::max(max_partkey, ps_partkey[i]);
    const uint value_map_size = (uint)(max_partkey + 1);
    const uint partsupp_size = (uint)psRows;
    const uint tg_size = 256;
    const uint num_threadgroups = (partsupp_size + tg_size - 1) / tg_size;

    // Create GPU buffers
    MTL::Buffer* pPsPartKeyBuf = device->newBuffer(ps_partkey.data(), psRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsSuppKeyBuf = device->newBuffer(ps_suppkey.data(), psRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsSupplyCostBuf = device->newBuffer(ps_supplycost.data(), psRows * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsAvailQtyBuf = device->newBuffer(ps_availqty.data(), psRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSuppBitmapBuf = device->newBuffer(cpu_supp_bitmap.data(), supp_bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* pValueMapBuf = device->newBuffer(value_map_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartialSumsBuf = device->newBuffer(num_threadgroups * sizeof(float), MTL::ResourceStorageModeShared);

    // Execute GPU (2 warmup + 1 measured)
    double gpu_time = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        std::memset(pValueMapBuf->contents(), 0, value_map_size * sizeof(float));
        std::memset(pPartialSumsBuf->contents(), 0, num_threadgroups * sizeof(float));

        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pAggregatePipe);
        enc->setBuffer(pPsPartKeyBuf, 0, 0);
        enc->setBuffer(pPsSuppKeyBuf, 0, 1);
        enc->setBuffer(pPsSupplyCostBuf, 0, 2);
        enc->setBuffer(pPsAvailQtyBuf, 0, 3);
        enc->setBuffer(pSuppBitmapBuf, 0, 4);
        enc->setBuffer(pValueMapBuf, 0, 5);
        enc->setBuffer(pPartialSumsBuf, 0, 6);
        enc->setBytes(&partsupp_size, sizeof(partsupp_size), 7);
        enc->dispatchThreads(MTL::Size(partsupp_size, 1, 1), MTL::Size(tg_size, 1, 1));
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();

        if (iter == 2) gpu_time = cb->GPUEndTime() - cb->GPUStartTime();
    }

    // CPU post-processing
    auto postT0 = std::chrono::high_resolution_clock::now();

    float* partial_sums = (float*)pPartialSumsBuf->contents();
    double global_sum = 0.0;
    for (uint i = 0; i < num_threadgroups; i++) {
        global_sum += partial_sums[i];
    }
    double threshold = global_sum * 0.0001;

    float* value_map = (float*)pValueMapBuf->contents();
    struct Q11Result { int partkey; double value; };
    std::vector<Q11Result> results;
    for (uint i = 0; i < value_map_size; i++) {
        if (value_map[i] > threshold) {
            results.push_back({(int)i, (double)value_map[i]});
        }
    }

    std::sort(results.begin(), results.end(), [](const Q11Result& a, const Q11Result& b) {
        return a.value > b.value;
    });

    printf("\nTPC-H Query 11 Results (Top 20 of %zu):\n", results.size());
    printf("+-----------+------------------+\n");
    printf("| ps_partkey|            value |\n");
    printf("+-----------+------------------+\n");
    size_t limit = std::min(results.size(), (size_t)20);
    for (size_t i = 0; i < limit; i++) {
        printf("| %9d | %16.2f |\n", results[i].partkey, results[i].value);
    }
    printf("+-----------+------------------+\n");
    printf("Total qualifying rows: %zu, Global sum: %.2f, Threshold: %.2f\n",
           results.size(), global_sum, threshold);

    auto postT1 = std::chrono::high_resolution_clock::now();
    double cpuPostMs = std::chrono::duration<double, std::milli>(postT1 - postT0).count();

    double gpuMs = gpu_time * 1000.0;
    printf("\nSF100 Q11 | %zu rows (partsupp)\n", psRows);
    printTimingSummary(cpuParseMs, gpuMs, cpuPostMs);

    releaseAll(pAggregatePipe, pPsPartKeyBuf, pPsSuppKeyBuf, pPsSupplyCostBuf,
              pPsAvailQtyBuf, pSuppBitmapBuf, pValueMapBuf, pPartialSumsBuf);
}
