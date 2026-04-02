#include "infra.h"
#include <cstring>

// ===================================================================
// TPC-H Q20 — Potential Part Promotion
// ===================================================================

// Q20HTEntry must match GPU struct
struct Q20HTEntry_CPU {
    int key_hi;   // partkey
    int key_lo;   // suppkey
    float value;  // sum(l_quantity)
};

void runQ20Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n--- Running TPC-H Query 20 Benchmark ---" << std::endl;

    const std::string sf_path = g_dataset_path;

    auto parseStart = std::chrono::high_resolution_clock::now();
    auto pCols = loadColumnsMulti(sf_path + "part.tbl", {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 55}});
    auto& p_partkey = pCols.ints(0); auto& p_name = pCols.chars(1);

    auto sCols = loadColumnsMulti(sf_path + "supplier.tbl", {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}, {2, ColType::CHAR_FIXED, 40}, {3, ColType::INT}});
    auto& s_suppkey = sCols.ints(0); auto& s_name = sCols.chars(1); auto& s_address = sCols.chars(2); auto& s_nationkey = sCols.ints(3);

    auto psCols = loadColumnsMulti(sf_path + "partsupp.tbl", {{0, ColType::INT}, {1, ColType::INT}, {2, ColType::INT}});
    auto& ps_partkey = psCols.ints(0); auto& ps_suppkey = psCols.ints(1); auto& ps_availqty = psCols.ints(2);

    auto lCols = loadColumnsMulti(sf_path + "lineitem.tbl", {{1, ColType::INT}, {2, ColType::INT}, {4, ColType::FLOAT}, {10, ColType::DATE}});
    auto& l_partkey = lCols.ints(1); auto& l_suppkey = lCols.ints(2); auto& l_quantity = lCols.floats(4); auto& l_shipdate = lCols.ints(10);

    auto nat = loadNation(sf_path);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double cpuParseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    // Find CANADA nationkey
    int canada_nk = findNationKey(nat, "CANADA");

    // Build part bitmap: p_name LIKE 'forest%'
    auto part_bm = buildCPUBitmap(p_partkey, [&](size_t i) {
        return trimFixed(p_name.data(), i, 55).substr(0, 6) == "forest";
    });

    uint liSize = (uint)l_partkey.size();
    // Hash table sized for ~10% of lineitem qualifying
    uint htCapacity = nextPow2(std::max(liSize / 4, 1024u));
    uint htMask = htCapacity - 1;

    int date_start = 19940101, date_end = 19950101;

    auto pAggPipe = createPipeline(device, library, "q20_aggregate_lineitem_kernel");
    auto pProbePipe = createPipeline(device, library, "q20_probe_partsupp_kernel");
    if (!pAggPipe || !pProbePipe) return;

    // Build CANADA supplier bitmap before GPU dispatch (needed as GPU input)
    int max_sk = 0;
    for (size_t i = 0; i < s_suppkey.size(); i++) max_sk = std::max(max_sk, s_suppkey[i]);
    uint canada_bv_ints = (max_sk + 32) / 32;
    std::vector<uint> canada_bm(canada_bv_ints, 0);
    for (size_t i = 0; i < s_suppkey.size(); i++) {
        if (s_nationkey[i] == canada_nk) {
            int sk = s_suppkey[i];
            canada_bm[sk / 32] |= (1u << (sk % 32));
        }
    }

    uint psSize = (uint)ps_partkey.size();

    MTL::Buffer* pLinePartKeyBuf = device->newBuffer(l_partkey.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineSuppKeyBuf = device->newBuffer(l_suppkey.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineQtyBuf = device->newBuffer(l_quantity.data(), liSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineDateBuf = device->newBuffer(l_shipdate.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartBitmapBuf = uploadBitmap(device, part_bm);
    MTL::Buffer* pHTBuf = device->newBuffer((size_t)htCapacity * sizeof(Q20HTEntry_CPU), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPSPartKeyBuf = device->newBuffer(ps_partkey.data(), psSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPSSuppKeyBuf = device->newBuffer(ps_suppkey.data(), psSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPSAvailQtyBuf = device->newBuffer(ps_availqty.data(), psSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCanadaBmBuf = device->newBuffer(canada_bm.data(), canada_bv_ints * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* pQualBmBuf = device->newBuffer(canada_bv_ints * sizeof(uint), MTL::ResourceStorageModeShared);

    double gpuSec = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        // Initialize HT: key_hi = -1
        auto* ht = (Q20HTEntry_CPU*)pHTBuf->contents();
        for (uint j = 0; j < htCapacity; j++) { ht[j].key_hi = -1; ht[j].key_lo = 0; ht[j].value = 0.0f; }
        // Clear qualifying bitmap
        memset(pQualBmBuf->contents(), 0, canada_bv_ints * sizeof(uint));

        MTL::CommandBuffer* cb = commandQueue->commandBuffer();

        // Encoder 1: aggregate lineitem into HT
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pAggPipe);
        enc->setBuffer(pLinePartKeyBuf, 0, 0);
        enc->setBuffer(pLineSuppKeyBuf, 0, 1);
        enc->setBuffer(pLineQtyBuf, 0, 2);
        enc->setBuffer(pLineDateBuf, 0, 3);
        enc->setBuffer(pPartBitmapBuf, 0, 4);
        enc->setBuffer(pHTBuf, 0, 5);
        enc->setBytes(&liSize, sizeof(liSize), 6);
        enc->setBytes(&htMask, sizeof(htMask), 7);
        enc->setBytes(&date_start, sizeof(date_start), 8);
        enc->setBytes(&date_end, sizeof(date_end), 9);
        enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));
        enc->endEncoding();

        // Encoder 2: probe partsupp against HT (memory barrier via encoder boundary)
        MTL::ComputeCommandEncoder* enc2 = cb->computeCommandEncoder();
        enc2->setComputePipelineState(pProbePipe);
        enc2->setBuffer(pPSPartKeyBuf, 0, 0);
        enc2->setBuffer(pPSSuppKeyBuf, 0, 1);
        enc2->setBuffer(pPSAvailQtyBuf, 0, 2);
        enc2->setBuffer(pPartBitmapBuf, 0, 3);
        enc2->setBuffer(pCanadaBmBuf, 0, 4);
        enc2->setBuffer(pHTBuf, 0, 5);
        enc2->setBuffer(pQualBmBuf, 0, 6);
        enc2->setBytes(&psSize, sizeof(psSize), 7);
        enc2->setBytes(&htMask, sizeof(htMask), 8);
        uint tgCount = (psSize + 1023) / 1024;
        enc2->dispatchThreadgroups(MTL::Size(tgCount, 1, 1), MTL::Size(1024, 1, 1));
        enc2->endEncoding();

        cb->commit(); cb->waitUntilCompleted();
        if (iter == 2) gpuSec = cb->GPUEndTime() - cb->GPUStartTime();
    }

    // CPU post: just read qualifying bitmap and collect supplier names
    auto postStart = std::chrono::high_resolution_clock::now();
    auto* qual_bm = (uint*)pQualBmBuf->contents();

    struct Q20Result { std::string s_name; std::string s_address; };
    std::vector<Q20Result> results;
    for (size_t i = 0; i < s_suppkey.size(); i++) {
        int sk = s_suppkey[i];
        if (sk >= 0 && (uint)sk <= (uint)max_sk && ((qual_bm[sk / 32] >> (sk % 32)) & 1)) {
            results.push_back({trimFixed(s_name.data(), i, 25), trimFixed(s_address.data(), i, 40)});
        }
    }
    std::sort(results.begin(), results.end(), [](const Q20Result& a, const Q20Result& b) {
        return a.s_name < b.s_name;
    });

    printf("\nTPC-H Q20 Results (Top 10):\n");
    printf("+---------------------------+------------------------------------------+\n");
    printf("| s_name                    | s_address                                |\n");
    printf("+---------------------------+------------------------------------------+\n");
    size_t show = std::min((size_t)10, results.size());
    for (size_t i = 0; i < show; i++) {
        printf("| %-25s | %-40s |\n", results[i].s_name.c_str(), results[i].s_address.c_str());
    }
    printf("+---------------------------+------------------------------------------+\n");
    printf("Total qualifying suppliers: %zu\n", results.size());
    auto postEnd = std::chrono::high_resolution_clock::now();
    double cpuPostMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nQ20 | %u lineitem\n", liSize);
    printTimingSummary(cpuParseMs, gpuSec * 1000.0, cpuPostMs);

    releaseAll(pAggPipe, pProbePipe, pLinePartKeyBuf, pLineSuppKeyBuf, pLineQtyBuf, pLineDateBuf,
              pPartBitmapBuf, pHTBuf, pPSPartKeyBuf, pPSSuppKeyBuf, pPSAvailQtyBuf,
              pCanadaBmBuf, pQualBmBuf);
}

// --- SF100 Chunked ---
void runQ20BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q20 Benchmark (SF100 Chunked) ===" << std::endl;

    MappedFile partFile, suppFile, psFile, liFile, natFile;
    if (!partFile.open(g_dataset_path + "part.tbl") ||
        !suppFile.open(g_dataset_path + "supplier.tbl") ||
        !psFile.open(g_dataset_path + "partsupp.tbl") ||
        !liFile.open(g_dataset_path + "lineitem.tbl") ||
        !natFile.open(g_dataset_path + "nation.tbl")) {
        std::cerr << "Q20 SF100: Cannot open required TBL files" << std::endl;
        return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto partIdx = buildLineIndex(partFile);
    auto suppIdx = buildLineIndex(suppFile);
    auto psIdx = buildLineIndex(psFile);
    auto liIdx = buildLineIndex(liFile);
    auto natIdx = buildLineIndex(natFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    auto bpT0 = std::chrono::high_resolution_clock::now();
    size_t partRows = partIdx.size(), suppRows = suppIdx.size();
    size_t psRows = psIdx.size(), liRows = liIdx.size();

    // Nation
    std::vector<int> n_nationkey, n_regionkey;
    std::vector<char> n_name_chars;
    parseNationRegionSF100(natFile, natIdx, n_nationkey, n_regionkey, n_name_chars);
    int canada_nk = -1;
    for (size_t i = 0; i < n_nationkey.size(); i++) {
        if (trimFixed(n_name_chars.data(), i, 25) == "CANADA") { canada_nk = n_nationkey[i]; break; }
    }

    // Part bitmap
    std::vector<int> p_partkey(partRows);
    std::vector<char> p_name(partRows * 55);
    parseIntColumnChunk(partFile, partIdx, 0, partRows, 0, p_partkey.data());
    parseCharColumnChunkFixed(partFile, partIdx, 0, partRows, 1, 55, p_name.data());
    int max_partkey = 0;
    for (auto k : p_partkey) max_partkey = std::max(max_partkey, k);
    uint part_bitmap_ints = (max_partkey + 31) / 32 + 1;
    std::vector<uint> part_bitmap(part_bitmap_ints, 0);
    for (size_t i = 0; i < partRows; i++) {
        std::string name = trimFixed(p_name.data(), i, 55);
        if (name.substr(0, 6) == "forest") {
            int pk = p_partkey[i];
            part_bitmap[pk / 32] |= (1u << (pk % 32));
        }
    }

    // Supplier
    std::vector<int> s_suppkey(suppRows), s_nationkey(suppRows);
    std::vector<char> s_name_chars(suppRows * 25), s_address_chars(suppRows * 40);
    parseIntColumnChunk(suppFile, suppIdx, 0, suppRows, 0, s_suppkey.data());
    parseCharColumnChunkFixed(suppFile, suppIdx, 0, suppRows, 1, 25, s_name_chars.data());
    parseCharColumnChunkFixed(suppFile, suppIdx, 0, suppRows, 2, 40, s_address_chars.data());
    parseIntColumnChunk(suppFile, suppIdx, 0, suppRows, 3, s_nationkey.data());

    // Partsupp
    std::vector<int> ps_partkey(psRows), ps_suppkey(psRows), ps_availqty(psRows);
    parseIntColumnChunk(psFile, psIdx, 0, psRows, 0, ps_partkey.data());
    parseIntColumnChunk(psFile, psIdx, 0, psRows, 1, ps_suppkey.data());
    parseIntColumnChunk(psFile, psIdx, 0, psRows, 2, ps_availqty.data());
    auto bpT1 = std::chrono::high_resolution_clock::now();
    double buildParseMs = indexBuildMs + std::chrono::duration<double, std::milli>(bpT1 - bpT0).count();

    uint htCapacity = nextPow2(std::max((uint)(liRows / 4), 1024u));
    uint htMask = htCapacity - 1;
    int date_start = 19940101, date_end = 19950101;

    auto pAggPipe = createPipeline(device, library, "q20_aggregate_lineitem_kernel");
    if (!pAggPipe) return;

    MTL::Buffer* pPartBitmapBuf = device->newBuffer(part_bitmap.data(), part_bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* pHTBuf = device->newBuffer((size_t)htCapacity * sizeof(Q20HTEntry_CPU), MTL::ResourceStorageModeShared);
    auto* ht_init = (Q20HTEntry_CPU*)pHTBuf->contents();
    for (uint j = 0; j < htCapacity; j++) { ht_init[j].key_hi = -1; ht_init[j].key_lo = 0; ht_init[j].value = 0.0f; }

    // Stream lineitem
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 16, liRows);
    struct Q20Slot { MTL::Buffer* partkey; MTL::Buffer* suppkey; MTL::Buffer* quantity; MTL::Buffer* shipdate; };
    Q20Slot slots[2];
    for (int s = 0; s < 2; s++) {
        slots[s].partkey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].suppkey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].quantity = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].shipdate = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
    }

    auto timing = chunkedStreamLoop(
        commandQueue, slots, 2, liRows, chunkRows,
        [&](Q20Slot& slot, size_t startRow, size_t rowCount) {
            parseIntColumnChunk(liFile, liIdx, startRow, rowCount, 1, (int*)slot.partkey->contents());
            parseIntColumnChunk(liFile, liIdx, startRow, rowCount, 2, (int*)slot.suppkey->contents());
            parseFloatColumnChunk(liFile, liIdx, startRow, rowCount, 4, (float*)slot.quantity->contents());
            parseDateColumnChunk(liFile, liIdx, startRow, rowCount, 10, (int*)slot.shipdate->contents());
        },
        [&](Q20Slot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(pAggPipe);
            enc->setBuffer(slot.partkey, 0, 0);
            enc->setBuffer(slot.suppkey, 0, 1);
            enc->setBuffer(slot.quantity, 0, 2);
            enc->setBuffer(slot.shipdate, 0, 3);
            enc->setBuffer(pPartBitmapBuf, 0, 4);
            enc->setBuffer(pHTBuf, 0, 5);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 6);
            enc->setBytes(&htMask, sizeof(htMask), 7);
            enc->setBytes(&date_start, sizeof(date_start), 8);
            enc->setBytes(&date_end, sizeof(date_end), 9);
            enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {}
    );

    // CPU post: direct HT probe + bitmaps (same optimization as SF1/SF10 path)
    auto* ht = (Q20HTEntry_CPU*)pHTBuf->contents();

    // Build canada_suppliers bitmap
    int max_sk = 0;
    for (size_t i = 0; i < suppRows; i++) max_sk = std::max(max_sk, s_suppkey[i]);
    size_t bm_ints = (max_sk / 32) + 1;
    std::vector<uint> canada_bm(bm_ints, 0);
    for (size_t i = 0; i < suppRows; i++) {
        if (s_nationkey[i] == canada_nk) canada_bm[s_suppkey[i] / 32] |= (1u << (s_suppkey[i] % 32));
    }

    // Qualifying suppkeys bitmap
    std::vector<uint> qual_bm(bm_ints, 0);
    for (size_t i = 0; i < psRows; i++) {
        int pk = ps_partkey[i], sk = ps_suppkey[i];
        if ((uint)pk > (uint)max_partkey) continue;
        if (!((part_bitmap[pk / 32] >> (pk % 32)) & 1)) continue;
        if (sk > max_sk || !((canada_bm[sk / 32] >> (sk % 32)) & 1)) continue;
        // Direct HT probe for sum_qty
        uint h_raw = (uint)(pk * 100001 + sk);
        h_raw ^= h_raw >> 16; h_raw *= 0x45d9f3b; h_raw ^= h_raw >> 16;
        uint slot = h_raw & htMask;
        float sum_qty = 0.0f;
        bool found = false;
        for (;;) {
            if (ht[slot].key_hi == pk && ht[slot].key_lo == sk) { sum_qty = ht[slot].value; found = true; break; }
            if (ht[slot].key_hi == -1) break;
            slot = (slot + 1) & htMask;
        }
        if (found && (float)ps_availqty[i] > 0.5f * sum_qty)
            qual_bm[sk / 32] |= (1u << (sk % 32));
    }

    struct Q20Result { std::string s_name; std::string s_address; };
    std::vector<Q20Result> results;
    for (size_t i = 0; i < suppRows; i++) {
        int sk = s_suppkey[i];
        if (sk <= max_sk && ((qual_bm[sk / 32] >> (sk % 32)) & 1))
            results.push_back({trimFixed(s_name_chars.data(), i, 25), trimFixed(s_address_chars.data(), i, 40)});
    }
    std::sort(results.begin(), results.end(), [](const Q20Result& a, const Q20Result& b) { return a.s_name < b.s_name; });

    printf("\nTPC-H Q20 Results (Top 10):\n");
    printf("+---------------------------+------------------------------------------+\n");
    printf("| s_name                    | s_address                                |\n");
    printf("+---------------------------+------------------------------------------+\n");
    size_t show = std::min((size_t)10, results.size());
    for (size_t i = 0; i < show; i++)
        printf("| %-25s | %-40s |\n", results[i].s_name.c_str(), results[i].s_address.c_str());
    printf("+---------------------------+------------------------------------------+\n");
    printf("Total qualifying suppliers: %zu\n", results.size());

    printf("\nSF100 Q20 | %zu chunks | %zu lineitem\n", timing.chunkCount, liRows);
    printTimingSummary(buildParseMs + timing.parseMs, timing.gpuMs, 0.0);

    releaseAll(pAggPipe, pPartBitmapBuf, pHTBuf);
    for (int s = 0; s < 2; s++) releaseAll(slots[s].partkey, slots[s].suppkey, slots[s].quantity, slots[s].shipdate);
}
