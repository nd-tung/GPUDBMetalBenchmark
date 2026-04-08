#include "infra.h"
#include <set>
#include <cstring>

// ===================================================================
// TPC-H Q16 — Parts/Supplier Relationship
// ===================================================================

void runQ16Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n--- Running TPC-H Query 16 Benchmark ---" << std::endl;

    const std::string sf_path = g_dataset_path;

    auto parseStart = std::chrono::high_resolution_clock::now();
    auto pCols = loadColumnsMulti(sf_path + "part.tbl", {{0, ColType::INT}, {3, ColType::CHAR_FIXED, 10}, {4, ColType::CHAR_FIXED, 25}, {5, ColType::INT}});
    auto& p_partkey = pCols.ints(0); auto& p_brand = pCols.chars(3); auto& p_type = pCols.chars(4); auto& p_size = pCols.ints(5);

    auto psCols = loadColumnsMulti(sf_path + "partsupp.tbl", {{0, ColType::INT}, {1, ColType::INT}});
    auto& ps_partkey = psCols.ints(0); auto& ps_suppkey = psCols.ints(1);

    auto sCols = loadColumnsMulti(sf_path + "supplier.tbl", {{0, ColType::INT}, {6, ColType::CHAR_FIXED, 101}});
    auto& s_suppkey = sCols.ints(0); auto& s_comment = sCols.chars(6);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double cpuParseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    auto prepStart = std::chrono::high_resolution_clock::now();

    // Build supplier complaints bitmap
    auto complaint_bm = buildCPUBitmap(s_suppkey, [&](size_t i) {
        std::string comment = trimFixed(s_comment.data(), i, 101);
        auto pos1 = comment.find("Customer");
        return pos1 != std::string::npos && comment.find("Complaints", pos1) != std::string::npos;
    });

    // Build part group map: partkey → group_id  
    // Qualifying parts: brand != 'Brand#45', type NOT LIKE 'MEDIUM POLISHED%', size in {49,14,23,45,19,3,36,9}
    std::set<int> valid_sizes = {49, 14, 23, 45, 19, 3, 36, 9};

    // Group = (brand, type, size) → assign integer group_id
    struct GroupKey { std::string brand; std::string type; int size; 
        bool operator<(const GroupKey& o) const {
            if (brand != o.brand) return brand < o.brand;
            if (type != o.type) return type < o.type;
            return size < o.size;
        }
    };
    std::map<GroupKey, int> group_map;
    std::vector<GroupKey> groups;

    int max_partkey = 0;
    for (int k : p_partkey) max_partkey = std::max(max_partkey, k);
    std::vector<int> part_group_map(max_partkey + 1, -1);

    for (size_t i = 0; i < p_partkey.size(); i++) {
        std::string brand = trimFixed(p_brand.data(), i, 10);
        std::string type = trimFixed(p_type.data(), i, 25);
        int size = p_size[i];

        if (brand == "Brand#45") continue;
        if (type.substr(0, 15) == "MEDIUM POLISHED") continue;
        if (valid_sizes.find(size) == valid_sizes.end()) continue;

        GroupKey gk{brand, type, size};
        auto it = group_map.find(gk);
        int gid;
        if (it == group_map.end()) {
            gid = (int)groups.size();
            group_map[gk] = gid;
            groups.push_back(gk);
        } else {
            gid = it->second;
        }
        part_group_map[p_partkey[i]] = gid;
    }
    auto prepEnd = std::chrono::high_resolution_clock::now();
    double cpuPrepMs = std::chrono::duration<double, std::milli>(prepEnd - prepStart).count();

    uint psSize = (uint)ps_partkey.size();
    uint partMapSize = (uint)(max_partkey + 1);
    uint numGroups = (uint)groups.size();

    // Find max suppkey for bitmap sizing
    int max_sk = 0;
    for (int k : ps_suppkey) max_sk = std::max(max_sk, k);
    uint bv_ints = (max_sk + 32) / 32;

    auto pScanPipe = createPipeline(device, library, "q16_scan_and_bitmap_kernel");
    auto pPopcountPipe = createPipeline(device, library, "q16_popcount_kernel");
    if (!pScanPipe || !pPopcountPipe) return;

    MTL::Buffer* pPsPartKeyBuf = device->newBuffer(ps_partkey.data(), psSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsSuppKeyBuf = device->newBuffer(ps_suppkey.data(), psSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartGroupMapBuf = device->newBuffer(part_group_map.data(), partMapSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pComplaintBitmapBuf = uploadBitmap(device, complaint_bm);
    // Per-group suppkey bitmaps (flat array): numGroups × bv_ints uint words
    size_t bitmapBytes = (size_t)numGroups * bv_ints * sizeof(uint);
    MTL::Buffer* pGroupBitmapsBuf = device->newBuffer(bitmapBytes, MTL::ResourceStorageModeShared);
    MTL::Buffer* pGroupCountsBuf = device->newBuffer(numGroups * sizeof(uint), MTL::ResourceStorageModeShared);

    double gpuSec = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(pGroupBitmapsBuf->contents(), 0, bitmapBytes);
        memset(pGroupCountsBuf->contents(), 0, numGroups * sizeof(uint));

        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

        // Kernel 1: scan partsupp + set bits in per-group bitmaps
        enc->setComputePipelineState(pScanPipe);
        enc->setBuffer(pPsPartKeyBuf, 0, 0);
        enc->setBuffer(pPsSuppKeyBuf, 0, 1);
        enc->setBuffer(pPartGroupMapBuf, 0, 2);
        enc->setBuffer(pComplaintBitmapBuf, 0, 3);
        enc->setBuffer(pGroupBitmapsBuf, 0, 4);
        enc->setBytes(&psSize, sizeof(psSize), 5);
        enc->setBytes(&partMapSize, sizeof(partMapSize), 6);
        enc->setBytes(&bv_ints, sizeof(bv_ints), 7);
        enc->dispatchThreads(MTL::Size(psSize, 1, 1), MTL::Size(256, 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // Kernel 2: popcount each group's bitmap
        enc->setComputePipelineState(pPopcountPipe);
        enc->setBuffer(pGroupBitmapsBuf, 0, 0);
        enc->setBuffer(pGroupCountsBuf, 0, 1);
        enc->setBytes(&numGroups, sizeof(numGroups), 2);
        enc->setBytes(&bv_ints, sizeof(bv_ints), 3);
        // One threadgroup per group, up to 256 threads each
        uint tgSizePop = std::min((uint)256, bv_ints);
        if (tgSizePop < 1) tgSizePop = 1;
        enc->dispatchThreadgroups(MTL::Size(numGroups, 1, 1), MTL::Size(tgSizePop, 1, 1));

        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
        if (iter == 2) gpuSec = cb->GPUEndTime() - cb->GPUStartTime();
    }

    // CPU post: just read counts and format results
    auto postStart = std::chrono::high_resolution_clock::now();
    uint* gpuGroupCounts = (uint*)pGroupCountsBuf->contents();

    // Count qualifying pairs for reporting
    uint outputCount = 0;
    for (uint i = 0; i < numGroups; i++) outputCount += gpuGroupCounts[i];

    struct Q16Result { std::string brand; std::string type; int size; int supplier_cnt; };
    std::vector<Q16Result> results;
    for (size_t i = 0; i < groups.size(); i++) {
        if (gpuGroupCounts[i] > 0) {
            results.push_back({groups[i].brand, groups[i].type, groups[i].size, (int)gpuGroupCounts[i]});
        }
    }
    std::sort(results.begin(), results.end(), [](const Q16Result& a, const Q16Result& b) {
        if (a.supplier_cnt != b.supplier_cnt) return a.supplier_cnt > b.supplier_cnt;
        if (a.brand != b.brand) return a.brand < b.brand;
        if (a.type != b.type) return a.type < b.type;
        return a.size < b.size;
    });

    printf("\nTPC-H Q16 Results (Top 10):\n");
    printf("+----------+---------------------------+------+--------------+\n");
    printf("| p_brand  | p_type                    |p_size| supplier_cnt |\n");
    printf("+----------+---------------------------+------+--------------+\n");
    size_t show = std::min((size_t)10, results.size());
    for (size_t i = 0; i < show; i++) {
        printf("| %-8s | %-25s | %4d | %12d |\n",
               results[i].brand.c_str(), results[i].type.c_str(), results[i].size, results[i].supplier_cnt);
    }
    printf("+----------+---------------------------+------+--------------+\n");
    printf("Total groups: %zu\n", results.size());
    auto postEnd = std::chrono::high_resolution_clock::now();
    double cpuPostMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nQ16 | %u partsupp | %u qualifying pairs\n", psSize, outputCount);
    printTimingSummary(cpuParseMs + cpuPrepMs, gpuSec * 1000.0, cpuPostMs);

    releaseAll(pScanPipe, pPopcountPipe, pPsPartKeyBuf, pPsSuppKeyBuf, pPartGroupMapBuf, pComplaintBitmapBuf,
              pGroupBitmapsBuf, pGroupCountsBuf);
}

// --- SF100 Chunked ---
void runQ16BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q16 Benchmark (SF100 Chunked) ===" << std::endl;

    MappedFile partFile, psFile, suppFile;
    if (!partFile.open(g_dataset_path + "part.tbl") ||
        !psFile.open(g_dataset_path + "partsupp.tbl") ||
        !suppFile.open(g_dataset_path + "supplier.tbl")) {
        std::cerr << "Q16 SF100: Cannot open required TBL files" << std::endl;
        return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto partIdx = buildLineIndex(partFile);
    auto psIdx = buildLineIndex(psFile);
    auto suppIdx = buildLineIndex(suppFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    auto bpT0 = std::chrono::high_resolution_clock::now();
    size_t partRows = partIdx.size(), suppRows = suppIdx.size(), psRows = psIdx.size();

    // Load part columns
    std::vector<int> p_partkey(partRows), p_size(partRows);
    std::vector<char> p_brand(partRows * 10), p_type(partRows * 25);
    parseIntColumnChunk(partFile, partIdx, 0, partRows, 0, p_partkey.data());
    parseCharColumnChunkFixed(partFile, partIdx, 0, partRows, 3, 10, p_brand.data());
    parseCharColumnChunkFixed(partFile, partIdx, 0, partRows, 4, 25, p_type.data());
    parseIntColumnChunk(partFile, partIdx, 0, partRows, 5, p_size.data());

    // Load supplier comment for complaints
    std::vector<int> s_suppkey(suppRows);
    std::vector<char> s_comment(suppRows * 101);
    parseIntColumnChunk(suppFile, suppIdx, 0, suppRows, 0, s_suppkey.data());
    parseCharColumnChunkFixed(suppFile, suppIdx, 0, suppRows, 6, 101, s_comment.data());

    // Build complaints bitmap
    int max_suppkey = 0;
    for (size_t i = 0; i < suppRows; i++) max_suppkey = std::max(max_suppkey, s_suppkey[i]);
    uint complaint_bitmap_ints = (max_suppkey + 31) / 32 + 1;
    std::vector<uint> complaint_bitmap(complaint_bitmap_ints, 0);
    for (size_t i = 0; i < suppRows; i++) {
        std::string comment = trimFixed(s_comment.data(), i, 101);
        auto pos1 = comment.find("Customer");
        if (pos1 != std::string::npos && comment.find("Complaints", pos1) != std::string::npos) {
            int sk = s_suppkey[i];
            complaint_bitmap[sk / 32] |= (1u << (sk % 32));
        }
    }

    // Build part group map
    std::set<int> valid_sizes = {49, 14, 23, 45, 19, 3, 36, 9};
    struct GroupKey { std::string brand; std::string type; int size;
        bool operator<(const GroupKey& o) const {
            if (brand != o.brand) return brand < o.brand;
            if (type != o.type) return type < o.type;
            return size < o.size;
        }
    };
    std::map<GroupKey, int> group_map;
    std::vector<GroupKey> groups;

    int max_partkey = 0;
    for (size_t i = 0; i < partRows; i++) max_partkey = std::max(max_partkey, p_partkey[i]);
    std::vector<int> part_group_map(max_partkey + 1, -1);

    for (size_t i = 0; i < partRows; i++) {
        std::string brand = trimFixed(p_brand.data(), i, 10);
        std::string type = trimFixed(p_type.data(), i, 25);
        int size = p_size[i];
        if (brand == "Brand#45") continue;
        if (type.substr(0, 15) == "MEDIUM POLISHED") continue;
        if (valid_sizes.find(size) == valid_sizes.end()) continue;
        GroupKey gk{brand, type, size};
        auto it = group_map.find(gk);
        int gid;
        if (it == group_map.end()) { gid = (int)groups.size(); group_map[gk] = gid; groups.push_back(gk); }
        else { gid = it->second; }
        part_group_map[p_partkey[i]] = gid;
    }
    auto bpT1 = std::chrono::high_resolution_clock::now();
    double buildParseMs = indexBuildMs + std::chrono::duration<double, std::milli>(bpT1 - bpT0).count();

    uint partMapSize = (uint)(max_partkey + 1);
    uint numGroups = (uint)groups.size();

    // Use max_suppkey from supplier table for bitmap sizing
    uint bv_ints = (max_suppkey + 32) / 32;

    auto pScanPipe = createPipeline(device, library, "q16_scan_and_bitmap_kernel");
    auto pPopcountPipe = createPipeline(device, library, "q16_popcount_kernel");
    if (!pScanPipe || !pPopcountPipe) return;

    MTL::Buffer* pPartGroupMapBuf = device->newBuffer(part_group_map.data(), partMapSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pComplaintBitmapBuf = device->newBuffer(complaint_bitmap.data(), complaint_bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);

    // Per-group suppkey bitmaps (flat array): numGroups × bv_ints uint words
    size_t bitmapBytes = (size_t)numGroups * bv_ints * sizeof(uint);
    MTL::Buffer* pGroupBitmapsBuf = device->newBuffer(bitmapBytes, MTL::ResourceStorageModeShared);
    memset(pGroupBitmapsBuf->contents(), 0, bitmapBytes);
    MTL::Buffer* pGroupCountsBuf = device->newBuffer(numGroups * sizeof(uint), MTL::ResourceStorageModeShared);
    memset(pGroupCountsBuf->contents(), 0, numGroups * sizeof(uint));

    printf("Q16 SF100: %u groups, bv_ints=%u, bitmap=%.1f MB\n",
           numGroups, bv_ints, bitmapBytes / (1024.0 * 1024.0));

    // Stream partsupp chunks through bitmap-set kernel
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 8, psRows);
    struct Q16Slot { MTL::Buffer* partkey; MTL::Buffer* suppkey; };
    Q16Slot slots[2];
    for (int s = 0; s < 2; s++) {
        slots[s].partkey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].suppkey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
    }

    auto timing = chunkedStreamLoop(
        commandQueue, slots, 2, psRows, chunkRows,
        [&](Q16Slot& slot, size_t startRow, size_t rowCount) {
            parseIntColumnChunk(psFile, psIdx, startRow, rowCount, 0, (int*)slot.partkey->contents());
            parseIntColumnChunk(psFile, psIdx, startRow, rowCount, 1, (int*)slot.suppkey->contents());
        },
        [&](Q16Slot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(pScanPipe);
            enc->setBuffer(slot.partkey, 0, 0);
            enc->setBuffer(slot.suppkey, 0, 1);
            enc->setBuffer(pPartGroupMapBuf, 0, 2);
            enc->setBuffer(pComplaintBitmapBuf, 0, 3);
            enc->setBuffer(pGroupBitmapsBuf, 0, 4);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 5);
            enc->setBytes(&partMapSize, sizeof(partMapSize), 6);
            enc->setBytes(&bv_ints, sizeof(bv_ints), 7);
            enc->dispatchThreads(MTL::Size(chunkSize, 1, 1), MTL::Size(256, 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {}
    );

    // After all chunks: run popcount kernel to count distinct suppkeys per group
    MTL::CommandBuffer* cbPop = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encPop = cbPop->computeCommandEncoder();
    encPop->setComputePipelineState(pPopcountPipe);
    encPop->setBuffer(pGroupBitmapsBuf, 0, 0);
    encPop->setBuffer(pGroupCountsBuf, 0, 1);
    encPop->setBytes(&numGroups, sizeof(numGroups), 2);
    encPop->setBytes(&bv_ints, sizeof(bv_ints), 3);
    uint tgSizePop = std::min((uint)256, bv_ints);
    if (tgSizePop < 1) tgSizePop = 1;
    encPop->dispatchThreadgroups(MTL::Size(numGroups, 1, 1), MTL::Size(tgSizePop, 1, 1));
    encPop->endEncoding();
    cbPop->commit(); cbPop->waitUntilCompleted();
    double popcountGpuMs = (cbPop->GPUEndTime() - cbPop->GPUStartTime()) * 1000.0;

    // Read GPU results
    uint* gpuGroupCounts = (uint*)pGroupCountsBuf->contents();
    uint outputCount = 0;
    for (uint i = 0; i < numGroups; i++) outputCount += gpuGroupCounts[i];

    struct Q16Result { std::string brand; std::string type; int size; int supplier_cnt; };
    std::vector<Q16Result> results;
    for (size_t i = 0; i < groups.size(); i++) {
        if (gpuGroupCounts[i] > 0)
            results.push_back({groups[i].brand, groups[i].type, groups[i].size, (int)gpuGroupCounts[i]});
    }
    std::sort(results.begin(), results.end(), [](const Q16Result& a, const Q16Result& b) {
        if (a.supplier_cnt != b.supplier_cnt) return a.supplier_cnt > b.supplier_cnt;
        if (a.brand != b.brand) return a.brand < b.brand;
        if (a.type != b.type) return a.type < b.type;
        return a.size < b.size;
    });

    printf("\nTPC-H Q16 Results (Top 10):\n");
    printf("+----------+---------------------------+------+--------------+\n");
    printf("| p_brand  | p_type                    |p_size| supplier_cnt |\n");
    printf("+----------+---------------------------+------+--------------+\n");
    size_t show = std::min((size_t)10, results.size());
    for (size_t i = 0; i < show; i++)
        printf("| %-8s | %-25s | %4d | %12d |\n",
               results[i].brand.c_str(), results[i].type.c_str(), results[i].size, results[i].supplier_cnt);
    printf("+----------+---------------------------+------+--------------+\n");
    printf("Total groups: %zu | %u qualifying pairs\n", results.size(), outputCount);

    printTimingSummary(buildParseMs + timing.parseMs, timing.gpuMs + popcountGpuMs, 0.0);

    releaseAll(pScanPipe, pPopcountPipe, pPartGroupMapBuf, pComplaintBitmapBuf,
              pGroupBitmapsBuf, pGroupCountsBuf);
    for (int s = 0; s < 2; s++) releaseAll(slots[s].partkey, slots[s].suppkey);
}
