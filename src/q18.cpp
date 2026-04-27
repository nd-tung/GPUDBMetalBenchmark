#include "infra.h"

// ===================================================================
// TPC-H Q18 — Large Volume Customer
// ===================================================================

void runQ18Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n--- Running TPC-H Query 18 Benchmark ---" << std::endl;

    const std::string sf_path = g_dataset_path;

    auto parseStart = std::chrono::high_resolution_clock::now();
    auto cCols = loadQueryColumns(device, sf_path + "customer.tbl", {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}});

    auto oCols = loadQueryColumns(device, sf_path + "orders.tbl", {{0, ColType::INT}, {1, ColType::INT}, {3, ColType::FLOAT}, {4, ColType::DATE}});

    auto lCols = loadQueryColumns(device, sf_path + "lineitem.tbl", {{0, ColType::INT}, {4, ColType::FLOAT}});
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double cpuParseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    uint liSize = (uint)lCols.rows();
    uint ordSize = (uint)oCols.rows();

    int max_orderkey = 0;
    for (int k : oCols.intSpan(0)) max_orderkey = std::max(max_orderkey, k);
    uint qty_map_size = max_orderkey + 1;

    auto pAggPipe = createPipeline(device, library, "q18_aggregate_quantity_kernel");
    auto pFilterPipe = createPipeline(device, library, "q18_filter_orders_kernel");
    if (!pAggPipe || !pFilterPipe) return;

    MTL::Buffer* pLineOrdKeyBuf = lCols.buffer(0);
    MTL::Buffer* pLineQtyBuf = lCols.buffer(4);
    MTL::Buffer* pQtyMapBuf = device->newBuffer((size_t)qty_map_size * sizeof(float), MTL::ResourceStorageModeShared);

    // Order columns for the filter kernel
    MTL::Buffer* pOrdKeyBuf = oCols.buffer(0);
    MTL::Buffer* pOrdCustKeyBuf = oCols.buffer(1);
    MTL::Buffer* pOrdDateBuf = oCols.buffer(4);
    MTL::Buffer* pOrdPriceBuf = oCols.buffer(3);

    // Output buffer for qualifying orders (worst case: all orders qualify)
    // Q18OutputRow = {int, int, int, float, float} = 20 bytes
    MTL::Buffer* pOutputBuf = device->newBuffer((size_t)ordSize * 20, MTL::ResourceStorageModeShared);
    MTL::Buffer* pOutputCountBuf = device->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared);

    float threshold = 300.0f;

    double gpuSec = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(pQtyMapBuf->contents(), 0, (size_t)qty_map_size * sizeof(float));
        *(uint*)pOutputCountBuf->contents() = 0;

        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

        // Kernel 1: aggregate quantity
        enc->setComputePipelineState(pAggPipe);
        enc->setBuffer(pLineOrdKeyBuf, 0, 0);
        enc->setBuffer(pLineQtyBuf, 0, 1);
        enc->setBuffer(pQtyMapBuf, 0, 2);
        enc->setBytes(&liSize, sizeof(liSize), 3);
        const uint q18AggTGs = std::min(2048u, (liSize + 1023u) / 1024u);
        enc->dispatchThreadgroups(MTL::Size(q18AggTGs, 1, 1), MTL::Size(1024, 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // Kernel 2: filter orders with sum > 300
        enc->setComputePipelineState(pFilterPipe);
        enc->setBuffer(pOrdKeyBuf, 0, 0);
        enc->setBuffer(pOrdCustKeyBuf, 0, 1);
        enc->setBuffer(pOrdDateBuf, 0, 2);
        enc->setBuffer(pOrdPriceBuf, 0, 3);
        enc->setBuffer(pQtyMapBuf, 0, 4);
        enc->setBuffer(pOutputBuf, 0, 5);
        enc->setBuffer(pOutputCountBuf, 0, 6);
        enc->setBytes(&ordSize, sizeof(ordSize), 7);
        enc->setBytes(&qty_map_size, sizeof(qty_map_size), 8);
        enc->setBytes(&threshold, sizeof(threshold), 9);
        enc->dispatchThreads(MTL::Size(ordSize, 1, 1), MTL::Size(256, 1, 1));

        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
        if (iter == 2) gpuSec = cb->GPUEndTime() - cb->GPUStartTime();
    }

    // CPU post: read compacted results, join with customer name, sort top-100
    auto postStart = std::chrono::high_resolution_clock::now();
    uint outputCount = *(uint*)pOutputCountBuf->contents();

    struct Q18Row { int o_orderkey; int o_custkey; int o_orderdate; float o_totalprice; float sum_qty; };
    auto* gpuRows = (Q18Row*)pOutputBuf->contents();

    // Build custkey→row index for customer
    int max_custkey = 0;
    for (int k : cCols.intSpan(0)) max_custkey = std::max(max_custkey, k);
    std::vector<int> cust_index(max_custkey + 1, -1);
    const int* c_custkey = cCols.ints(0);
    const char* c_name = cCols.chars(1);
    for (size_t i = 0; i < cCols.rows(); i++) cust_index[c_custkey[i]] = (int)i;

    struct Q18Result {
        std::string c_name; int c_custkey; int o_orderkey; int o_orderdate;
        float o_totalprice; float sum_qty;
    };
    std::vector<Q18Result> results;
    results.reserve(outputCount);
    for (uint i = 0; i < outputCount; i++) {
        int ck = gpuRows[i].o_custkey;
        std::string name;
        if (ck <= max_custkey && cust_index[ck] >= 0)
            name = trimFixed(c_name, cust_index[ck], 25);
        results.push_back({name, ck, gpuRows[i].o_orderkey, gpuRows[i].o_orderdate,
                          gpuRows[i].o_totalprice, gpuRows[i].sum_qty});
    }

    size_t topK = std::min((size_t)100, results.size());
    std::partial_sort(results.begin(), results.begin() + topK, results.end(),
        [](const Q18Result& a, const Q18Result& b) {
            if (a.o_totalprice != b.o_totalprice) return a.o_totalprice > b.o_totalprice;
            return a.o_orderdate < b.o_orderdate;
        });

    printf("\nTPC-H Q18 Results (Top 10 of LIMIT 100):\n");
    printf("+------------------+----------+----------+----------+----------+----------+\n");
    printf("| c_name           | c_custkey| o_orderkey| o_orderdate| o_totalprice| sum_qty |\n");
    printf("+------------------+----------+----------+----------+----------+----------+\n");
    size_t show = std::min((size_t)10, topK);
    for (size_t i = 0; i < show; i++) {
        printf("| %-16s | %8d | %9d | %10d | %11.2f | %7.2f |\n",
               results[i].c_name.c_str(), results[i].c_custkey, results[i].o_orderkey,
               results[i].o_orderdate, results[i].o_totalprice, results[i].sum_qty);
    }
    printf("+------------------+----------+----------+----------+----------+----------+\n");
    printf("Total qualifying orders: %zu\n", results.size());
    auto postEnd = std::chrono::high_resolution_clock::now();
    double cpuPostMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nQ18 | %u lineitem\n", liSize);
    printTimingSummary(cpuParseMs, gpuSec * 1000.0, cpuPostMs);

    releaseAll(pAggPipe, pFilterPipe, pQtyMapBuf,
              pOutputBuf, pOutputCountBuf);
    // Input buffers owned by cCols/oCols/lCols (QueryColumns).
}

// --- SF100 Chunked ---
void runQ18BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q18 Benchmark (SF100 Chunked) ===" << std::endl;

    MappedFile custFile, ordFile, liFile;
    if (!custFile.open(g_dataset_path + "customer.tbl") ||
        !ordFile.open(g_dataset_path + "orders.tbl") ||
        !liFile.open(g_dataset_path + "lineitem.tbl")) {
        std::cerr << "Q18 SF100: Cannot open required TBL files" << std::endl;
        return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto custIdx = buildLineIndex(custFile);
    auto ordIdx = buildLineIndex(ordFile);
    auto liIdx = buildLineIndex(liFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    auto bpT0 = std::chrono::high_resolution_clock::now();
    size_t custRows = custIdx.size(), ordRows = ordIdx.size(), liRows = liIdx.size();

    // Load dimension tables  
    std::vector<int> c_custkey(custRows);
    std::vector<char> c_name(custRows * 25);
    parseIntColumnChunk(custFile, custIdx, 0, custRows, 0, c_custkey.data());
    parseCharColumnChunkFixed(custFile, custIdx, 0, custRows, 1, 25, c_name.data());

    std::vector<int> o_orderkey(ordRows), o_custkey(ordRows), o_orderdate(ordRows);
    std::vector<float> o_totalprice(ordRows);
    parseIntColumnChunk(ordFile, ordIdx, 0, ordRows, 0, o_orderkey.data());
    parseIntColumnChunk(ordFile, ordIdx, 0, ordRows, 1, o_custkey.data());
    parseFloatColumnChunk(ordFile, ordIdx, 0, ordRows, 3, o_totalprice.data());
    parseDateColumnChunk(ordFile, ordIdx, 0, ordRows, 4, o_orderdate.data());
    auto bpT1 = std::chrono::high_resolution_clock::now();
    double buildParseMs = indexBuildMs + std::chrono::duration<double, std::milli>(bpT1 - bpT0).count();

    int max_orderkey = 0;
    for (size_t i = 0; i < ordRows; i++) max_orderkey = std::max(max_orderkey, o_orderkey[i]);
    uint qty_map_size = max_orderkey + 1;

    auto pAggPipe = createPipeline(device, library, "q18_aggregate_quantity_kernel");
    if (!pAggPipe) return;

    MTL::Buffer* pQtyMapBuf = device->newBuffer((size_t)qty_map_size * sizeof(float), MTL::ResourceStorageModeShared);
    memset(pQtyMapBuf->contents(), 0, (size_t)qty_map_size * sizeof(float));

    // Stream lineitem
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 8, liRows);
    struct Q18Slot { MTL::Buffer* orderkey; MTL::Buffer* quantity; };
    Q18Slot slots[2];
    for (int s = 0; s < 2; s++) {
        slots[s].orderkey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].quantity = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
    }

    auto timing = chunkedStreamLoop(
        commandQueue, slots, 2, liRows, chunkRows,
        [&](Q18Slot& slot, size_t startRow, size_t rowCount) {
            parseIntColumnChunk(liFile, liIdx, startRow, rowCount, 0, (int*)slot.orderkey->contents());
            parseFloatColumnChunk(liFile, liIdx, startRow, rowCount, 4, (float*)slot.quantity->contents());
        },
        [&](Q18Slot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(pAggPipe);
            enc->setBuffer(slot.orderkey, 0, 0);
            enc->setBuffer(slot.quantity, 0, 1);
            enc->setBuffer(pQtyMapBuf, 0, 2);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 3);
            enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {}
    );

    // CPU post
    float* qtyMap = (float*)pQtyMapBuf->contents();

    int max_custkey = 0;
    for (size_t i = 0; i < custRows; i++) max_custkey = std::max(max_custkey, c_custkey[i]);
    std::vector<int> cust_index(max_custkey + 1, -1);
    for (size_t i = 0; i < custRows; i++) cust_index[c_custkey[i]] = (int)i;

    struct Q18Result {
        std::string c_name; int c_custkey; int o_orderkey; int o_orderdate;
        float o_totalprice; float sum_qty;
    };
    std::vector<Q18Result> results;
    for (size_t i = 0; i < ordRows; i++) {
        int okey = o_orderkey[i];
        if ((uint)okey < qty_map_size && qtyMap[okey] > 300.0f) {
            int ck = o_custkey[i];
            std::string name;
            if (ck <= max_custkey && cust_index[ck] >= 0)
                name = trimFixed(c_name.data(), cust_index[ck], 25);
            results.push_back({name, ck, okey, o_orderdate[i], o_totalprice[i], qtyMap[okey]});
        }
    }

    size_t topK = std::min((size_t)100, results.size());
    std::partial_sort(results.begin(), results.begin() + topK, results.end(),
        [](const Q18Result& a, const Q18Result& b) {
            if (a.o_totalprice != b.o_totalprice) return a.o_totalprice > b.o_totalprice;
            return a.o_orderdate < b.o_orderdate;
        });

    printf("\nTPC-H Q18 Results (Top 10 of LIMIT 100):\n");
    printf("+------------------+----------+----------+----------+----------+----------+\n");
    printf("| c_name           | c_custkey| o_orderkey| o_orderdate| o_totalprice| sum_qty |\n");
    printf("+------------------+----------+----------+----------+----------+----------+\n");
    size_t show = std::min((size_t)10, topK);
    for (size_t i = 0; i < show; i++)
        printf("| %-16s | %8d | %9d | %10d | %11.2f | %7.2f |\n",
               results[i].c_name.c_str(), results[i].c_custkey, results[i].o_orderkey,
               results[i].o_orderdate, results[i].o_totalprice, results[i].sum_qty);
    printf("+------------------+----------+----------+----------+----------+----------+\n");
    printf("Total qualifying orders: %zu\n", results.size());

    printf("\nSF100 Q18 | %zu chunks | %zu lineitem\n", timing.chunkCount, liRows);
    printTimingSummary(buildParseMs + timing.parseMs, timing.gpuMs, 0.0);

    releaseAll(pAggPipe, pQtyMapBuf);
    for (int s = 0; s < 2; s++) releaseAll(slots[s].orderkey, slots[s].quantity);
}
