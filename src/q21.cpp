#include "infra.h"

// ===================================================================
// TPC-H Q21 — Suppliers Who Kept Orders Waiting
// ===================================================================

void runQ21Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n--- Running TPC-H Query 21 Benchmark ---" << std::endl;

    const std::string sf_path = g_dataset_path;

    auto parseStart = std::chrono::high_resolution_clock::now();
    auto sCols = loadColumnsMulti(sf_path + "supplier.tbl", {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}, {3, ColType::INT}});
    auto& s_suppkey = sCols.ints(0); auto& s_name = sCols.chars(1); auto& s_nationkey = sCols.ints(3);

    auto oCols = loadColumnsMulti(sf_path + "orders.tbl", {{0, ColType::INT}, {2, ColType::CHAR1}});
    auto& o_orderkey = oCols.ints(0); auto& o_orderstatus = oCols.chars(2);

    auto lCols = loadColumnsMulti(sf_path + "lineitem.tbl", {{0, ColType::INT}, {2, ColType::INT}, {11, ColType::DATE}, {12, ColType::DATE}});
    auto& l_orderkey = lCols.ints(0); auto& l_suppkey = lCols.ints(2);
    auto& l_commitdate = lCols.ints(11); auto& l_receiptdate = lCols.ints(12);

    auto nat = loadNation(sf_path);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double cpuParseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    int sa_nk = findNationKey(nat, "SAUDI ARABIA");

    // Build SAUDI ARABIA supplier bitmap
    auto sa_bm = buildCPUBitmap(s_suppkey, [&](size_t i) { return s_nationkey[i] == sa_nk; });
    int max_suppkey = sa_bm.max_key;

    // Build orders status map: orderkey → 1 if 'F', -1 otherwise
    int max_orderkey = 0;
    for (int k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    uint map_size = max_orderkey + 1;
    std::vector<int> orders_status_map(map_size, -1);
    for (size_t i = 0; i < o_orderkey.size(); i++) {
        if (o_orderstatus[i] == 'F') orders_status_map[o_orderkey[i]] = 1;
    }

    uint liSize = (uint)l_orderkey.size();
    uint bitmapInts = (map_size + 31) / 32 + 1;

    auto pBuildPipe = createPipeline(device, library, "q21_build_order_tracking_kernel");
    auto pCountPipe = createPipeline(device, library, "q21_count_qualifying_kernel");
    if (!pBuildPipe || !pCountPipe) return;

    MTL::Buffer* pLineOrdKeyBuf = device->newBuffer(l_orderkey.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineSuppKeyBuf = device->newBuffer(l_suppkey.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineReceiptBuf = device->newBuffer(l_receiptdate.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineCommitBuf = device->newBuffer(l_commitdate.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrderStatusMapBuf = device->newBuffer(orders_status_map.data(), map_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pFirstSuppBuf = device->newBuffer((size_t)map_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pMultiSuppBmBuf = device->newBuffer((size_t)bitmapInts * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLateSuppBuf = device->newBuffer((size_t)map_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pMultiLateBmBuf = device->newBuffer((size_t)bitmapInts * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSaBitmapBuf = uploadBitmap(device, sa_bm);
    MTL::Buffer* pSuppCountBuf = device->newBuffer((size_t)(max_suppkey + 1) * sizeof(uint), MTL::ResourceStorageModeShared);

    double gpuSec = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(pFirstSuppBuf->contents(), -1, (size_t)map_size * sizeof(int));
        memset(pMultiSuppBmBuf->contents(), 0, (size_t)bitmapInts * sizeof(uint));
        memset(pLateSuppBuf->contents(), -1, (size_t)map_size * sizeof(int));
        memset(pMultiLateBmBuf->contents(), 0, (size_t)bitmapInts * sizeof(uint));
        memset(pSuppCountBuf->contents(), 0, (size_t)(max_suppkey + 1) * sizeof(uint));

        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

        // Pass 1: build tracking
        enc->setComputePipelineState(pBuildPipe);
        enc->setBuffer(pLineOrdKeyBuf, 0, 0);
        enc->setBuffer(pLineSuppKeyBuf, 0, 1);
        enc->setBuffer(pLineReceiptBuf, 0, 2);
        enc->setBuffer(pLineCommitBuf, 0, 3);
        enc->setBuffer(pOrderStatusMapBuf, 0, 4);
        enc->setBuffer(pFirstSuppBuf, 0, 5);
        enc->setBuffer(pMultiSuppBmBuf, 0, 6);
        enc->setBuffer(pLateSuppBuf, 0, 7);
        enc->setBuffer(pMultiLateBmBuf, 0, 8);
        enc->setBytes(&liSize, sizeof(liSize), 9);
        enc->setBytes(&map_size, sizeof(map_size), 10);
        enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // Pass 2: count qualifying
        enc->setComputePipelineState(pCountPipe);
        enc->setBuffer(pLineOrdKeyBuf, 0, 0);
        enc->setBuffer(pLineSuppKeyBuf, 0, 1);
        enc->setBuffer(pLineReceiptBuf, 0, 2);
        enc->setBuffer(pLineCommitBuf, 0, 3);
        enc->setBuffer(pOrderStatusMapBuf, 0, 4);
        enc->setBuffer(pFirstSuppBuf, 0, 5);
        enc->setBuffer(pMultiSuppBmBuf, 0, 6);
        enc->setBuffer(pLateSuppBuf, 0, 7);
        enc->setBuffer(pMultiLateBmBuf, 0, 8);
        enc->setBuffer(pSaBitmapBuf, 0, 9);
        enc->setBuffer(pSuppCountBuf, 0, 10);
        enc->setBytes(&liSize, sizeof(liSize), 11);
        enc->setBytes(&map_size, sizeof(map_size), 12);
        enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));

        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
        if (iter == 2) gpuSec = cb->GPUEndTime() - cb->GPUStartTime();
    }

    // CPU post
    auto postStart = std::chrono::high_resolution_clock::now();
    uint* suppCounts = (uint*)pSuppCountBuf->contents();
    struct Q21Result { std::string s_name; uint numwait; };
    std::vector<Q21Result> results;

    // Build suppkey→name index
    std::vector<int> supp_idx(max_suppkey + 1, -1);
    for (size_t i = 0; i < s_suppkey.size(); i++) supp_idx[s_suppkey[i]] = (int)i;

    for (int sk = 0; sk <= max_suppkey; sk++) {
        if (suppCounts[sk] > 0 && supp_idx[sk] >= 0) {
            results.push_back({trimFixed(s_name.data(), supp_idx[sk], 25), suppCounts[sk]});
        }
    }

    size_t topK = std::min((size_t)100, results.size());
    std::partial_sort(results.begin(), results.begin() + topK, results.end(),
        [](const Q21Result& a, const Q21Result& b) {
            if (a.numwait != b.numwait) return a.numwait > b.numwait;
            return a.s_name < b.s_name;
        });

    printf("\nTPC-H Q21 Results (Top 10 of LIMIT 100):\n");
    printf("+---------------------------+----------+\n");
    printf("| s_name                    | numwait  |\n");
    printf("+---------------------------+----------+\n");
    size_t show = std::min((size_t)10, topK);
    for (size_t i = 0; i < show; i++) {
        printf("| %-25s | %8u |\n", results[i].s_name.c_str(), results[i].numwait);
    }
    printf("+---------------------------+----------+\n");
    printf("Total qualifying SA suppliers: %zu\n", results.size());
    auto postEnd = std::chrono::high_resolution_clock::now();
    double cpuPostMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nQ21 | %u lineitem\n", liSize);
    printTimingSummary(cpuParseMs, gpuSec * 1000.0, cpuPostMs);

    releaseAll(pBuildPipe, pCountPipe, pLineOrdKeyBuf, pLineSuppKeyBuf, pLineReceiptBuf,
              pLineCommitBuf, pOrderStatusMapBuf, pFirstSuppBuf, pMultiSuppBmBuf,
              pLateSuppBuf, pMultiLateBmBuf, pSaBitmapBuf, pSuppCountBuf);
}

// --- SF100 Chunked ---
void runQ21BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q21 Benchmark (SF100 Chunked) ===" << std::endl;

    MappedFile suppFile, ordFile, liFile, natFile;
    if (!suppFile.open(g_dataset_path + "supplier.tbl") ||
        !ordFile.open(g_dataset_path + "orders.tbl") ||
        !liFile.open(g_dataset_path + "lineitem.tbl") ||
        !natFile.open(g_dataset_path + "nation.tbl")) {
        std::cerr << "Q21 SF100: Cannot open required TBL files" << std::endl;
        return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto suppIdx = buildLineIndex(suppFile);
    auto ordIdx = buildLineIndex(ordFile);
    auto liIdx = buildLineIndex(liFile);
    auto natIdx = buildLineIndex(natFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    auto bpT0 = std::chrono::high_resolution_clock::now();
    size_t suppRows = suppIdx.size(), ordRows = ordIdx.size(), liRows = liIdx.size();

    // Nation
    std::vector<int> n_nationkey, n_regionkey;
    std::vector<char> n_name_chars;
    parseNationRegionSF100(natFile, natIdx, n_nationkey, n_regionkey, n_name_chars);
    int sa_nk = -1;
    for (size_t i = 0; i < n_nationkey.size(); i++) {
        if (trimFixed(n_name_chars.data(), i, 25) == "SAUDI ARABIA") { sa_nk = n_nationkey[i]; break; }
    }

    // Supplier
    std::vector<int> s_suppkey(suppRows), s_nationkey(suppRows);
    std::vector<char> s_name_chars(suppRows * 25);
    parseIntColumnChunk(suppFile, suppIdx, 0, suppRows, 0, s_suppkey.data());
    parseCharColumnChunkFixed(suppFile, suppIdx, 0, suppRows, 1, 25, s_name_chars.data());
    parseIntColumnChunk(suppFile, suppIdx, 0, suppRows, 3, s_nationkey.data());

    int max_suppkey = 0;
    for (auto k : s_suppkey) max_suppkey = std::max(max_suppkey, k);
    uint sa_bitmap_ints = (max_suppkey + 31) / 32 + 1;
    std::vector<uint> sa_bitmap(sa_bitmap_ints, 0);
    for (size_t i = 0; i < suppRows; i++) {
        if (s_nationkey[i] == sa_nk) {
            int sk = s_suppkey[i];
            sa_bitmap[sk / 32] |= (1u << (sk % 32));
        }
    }

    // Orders status map
    std::vector<int> o_orderkey(ordRows);
    std::vector<char> o_orderstatus(ordRows);
    parseIntColumnChunk(ordFile, ordIdx, 0, ordRows, 0, o_orderkey.data());
    parseCharColumnChunk(ordFile, ordIdx, 0, ordRows, 2, o_orderstatus.data());

    int max_orderkey = 0;
    for (auto k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    uint map_size = max_orderkey + 1;
    std::vector<int> orders_status_map(map_size, -1);
    for (size_t i = 0; i < ordRows; i++) {
        if (o_orderstatus[i] == 'F') orders_status_map[o_orderkey[i]] = 1;
    }
    auto bpT1 = std::chrono::high_resolution_clock::now();
    double buildParseMs = indexBuildMs + std::chrono::duration<double, std::milli>(bpT1 - bpT0).count();

    uint bitmapInts = (map_size + 31) / 32 + 1;

    auto pBuildPipe = createPipeline(device, library, "q21_build_order_tracking_kernel");
    auto pCountPipe = createPipeline(device, library, "q21_count_qualifying_kernel");
    if (!pBuildPipe || !pCountPipe) return;

    MTL::Buffer* pOrderStatusMapBuf = device->newBuffer(orders_status_map.data(), map_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pFirstSuppBuf = createFilledBuffer(device, (size_t)map_size * sizeof(int), 0xFF);
    memset(pFirstSuppBuf->contents(), -1, (size_t)map_size * sizeof(int));
    MTL::Buffer* pMultiSuppBmBuf = createFilledBuffer(device, (size_t)bitmapInts * sizeof(uint), 0);
    MTL::Buffer* pLateSuppBuf = createFilledBuffer(device, (size_t)map_size * sizeof(int), 0xFF);
    memset(pLateSuppBuf->contents(), -1, (size_t)map_size * sizeof(int));
    MTL::Buffer* pMultiLateBmBuf = createFilledBuffer(device, (size_t)bitmapInts * sizeof(uint), 0);
    MTL::Buffer* pSaBitmapBuf = device->newBuffer(sa_bitmap.data(), sa_bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSuppCountBuf = createFilledBuffer(device, (size_t)(max_suppkey + 1) * sizeof(uint), 0);

    // Pass 1: Stream lineitem for tracking
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 16, liRows);
    struct Q21Slot { MTL::Buffer* orderkey; MTL::Buffer* suppkey; MTL::Buffer* receiptdate; MTL::Buffer* commitdate; };
    Q21Slot slots[2];
    for (int s = 0; s < 2; s++) {
        slots[s].orderkey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].suppkey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].receiptdate = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].commitdate = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
    }

    auto parseChunk = [&](Q21Slot& slot, size_t startRow, size_t rowCount) {
        parseIntColumnChunk(liFile, liIdx, startRow, rowCount, 0, (int*)slot.orderkey->contents());
        parseIntColumnChunk(liFile, liIdx, startRow, rowCount, 2, (int*)slot.suppkey->contents());
        parseDateColumnChunk(liFile, liIdx, startRow, rowCount, 11, (int*)slot.commitdate->contents());
        parseDateColumnChunk(liFile, liIdx, startRow, rowCount, 12, (int*)slot.receiptdate->contents());
    };

    auto timing1 = chunkedStreamLoop(
        commandQueue, slots, 2, liRows, chunkRows,
        parseChunk,
        [&](Q21Slot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(pBuildPipe);
            enc->setBuffer(slot.orderkey, 0, 0);
            enc->setBuffer(slot.suppkey, 0, 1);
            enc->setBuffer(slot.receiptdate, 0, 2);
            enc->setBuffer(slot.commitdate, 0, 3);
            enc->setBuffer(pOrderStatusMapBuf, 0, 4);
            enc->setBuffer(pFirstSuppBuf, 0, 5);
            enc->setBuffer(pMultiSuppBmBuf, 0, 6);
            enc->setBuffer(pLateSuppBuf, 0, 7);
            enc->setBuffer(pMultiLateBmBuf, 0, 8);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 9);
            enc->setBytes(&map_size, sizeof(map_size), 10);
            enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {}
    );

    // Pass 2: Stream lineitem for counting
    auto timing2 = chunkedStreamLoop(
        commandQueue, slots, 2, liRows, chunkRows,
        parseChunk,
        [&](Q21Slot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(pCountPipe);
            enc->setBuffer(slot.orderkey, 0, 0);
            enc->setBuffer(slot.suppkey, 0, 1);
            enc->setBuffer(slot.receiptdate, 0, 2);
            enc->setBuffer(slot.commitdate, 0, 3);
            enc->setBuffer(pOrderStatusMapBuf, 0, 4);
            enc->setBuffer(pFirstSuppBuf, 0, 5);
            enc->setBuffer(pMultiSuppBmBuf, 0, 6);
            enc->setBuffer(pLateSuppBuf, 0, 7);
            enc->setBuffer(pMultiLateBmBuf, 0, 8);
            enc->setBuffer(pSaBitmapBuf, 0, 9);
            enc->setBuffer(pSuppCountBuf, 0, 10);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 11);
            enc->setBytes(&map_size, sizeof(map_size), 12);
            enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {}
    );

    // CPU post
    uint* suppCounts = (uint*)pSuppCountBuf->contents();
    std::vector<int> supp_idx(max_suppkey + 1, -1);
    for (size_t i = 0; i < suppRows; i++) supp_idx[s_suppkey[i]] = (int)i;

    struct Q21Result { std::string s_name; uint numwait; };
    std::vector<Q21Result> results;
    for (int sk = 0; sk <= max_suppkey; sk++) {
        if (suppCounts[sk] > 0 && supp_idx[sk] >= 0)
            results.push_back({trimFixed(s_name_chars.data(), supp_idx[sk], 25), suppCounts[sk]});
    }

    size_t topK = std::min((size_t)100, results.size());
    std::partial_sort(results.begin(), results.begin() + topK, results.end(),
        [](const Q21Result& a, const Q21Result& b) {
            if (a.numwait != b.numwait) return a.numwait > b.numwait;
            return a.s_name < b.s_name;
        });

    printf("\nTPC-H Q21 Results (Top 10 of LIMIT 100):\n");
    printf("+---------------------------+----------+\n");
    printf("| s_name                    | numwait  |\n");
    printf("+---------------------------+----------+\n");
    size_t show = std::min((size_t)10, topK);
    for (size_t i = 0; i < show; i++)
        printf("| %-25s | %8u |\n", results[i].s_name.c_str(), results[i].numwait);
    printf("+---------------------------+----------+\n");
    printf("Total qualifying SA suppliers: %zu\n", results.size());

    printf("\nSF100 Q21 | %zu chunks (2 passes) | %zu lineitem\n",
           timing1.chunkCount + timing2.chunkCount, liRows);
    printTimingSummary(buildParseMs + timing1.parseMs + timing2.parseMs,
                       timing1.gpuMs + timing2.gpuMs, 0.0);

    releaseAll(pBuildPipe, pCountPipe, pOrderStatusMapBuf, pFirstSuppBuf, pMultiSuppBmBuf,
              pLateSuppBuf, pMultiLateBmBuf, pSaBitmapBuf, pSuppCountBuf);
    for (int s = 0; s < 2; s++) releaseAll(slots[s].orderkey, slots[s].suppkey, slots[s].receiptdate, slots[s].commitdate);
}
