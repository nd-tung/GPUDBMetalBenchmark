#include "infra.h"

// ===================================================================
// TPC-H Q19 — Discounted Revenue
// ===================================================================

// Helper: build part group map on CPU
// group 0: Brand#12, SM container, size 1-5
// group 1: Brand#23, MED container, size 1-10
// group 2: Brand#34, LG container, size 1-15
// 0xFF: no match
static std::pair<std::vector<uint8_t>, int> buildPartGroupMap(
    const std::string& partPath) {
    auto p_partkey   = loadIntColumn(partPath, 0);
    auto p_brand     = loadCharColumn(partPath, 3, 10);  // CHAR(10)
    auto p_container = loadCharColumn(partPath, 6, 10);  // CHAR(10)
    auto p_size      = loadIntColumn(partPath, 5);

    int max_partkey = 0;
    for (int k : p_partkey) max_partkey = std::max(max_partkey, k);

    std::vector<uint8_t> map(max_partkey + 1, 0xFF);

    auto matchBrand = [](const char* b, const char* target, int len) {
        for (int i = 0; i < len; i++) if (b[i] != target[i]) return false;
        return true;
    };

    auto matchCont = [](const char* c, const char* target) {
        for (int i = 0; target[i]; i++) if (c[i] != target[i]) return false;
        return true;
    };

    for (size_t i = 0; i < p_partkey.size(); i++) {
        const char* brand = &p_brand[i * 10];
        const char* cont  = &p_container[i * 10];
        int sz = p_size[i];

        // Group 0: Brand#12, SM {CASE,BOX,PACK,PKG}, size 1-5
        if (matchBrand(brand, "Brand#12", 8) && sz >= 1 && sz <= 5
            && (matchCont(cont,"SM CASE") || matchCont(cont,"SM BOX") || matchCont(cont,"SM PACK") || matchCont(cont,"SM PKG"))) {
            map[p_partkey[i]] = 0;
        }
        // Group 1: Brand#23, MED {BAG,BOX,PKG,PACK}, size 1-10
        else if (matchBrand(brand, "Brand#23", 8) && sz >= 1 && sz <= 10
            && (matchCont(cont,"MED BAG") || matchCont(cont,"MED BOX") || matchCont(cont,"MED PKG") || matchCont(cont,"MED PACK"))) {
            map[p_partkey[i]] = 1;
        }
        // Group 2: Brand#34, LG {CASE,BOX,PACK,PKG}, size 1-15
        else if (matchBrand(brand, "Brand#34", 8) && sz >= 1 && sz <= 15
            && (matchCont(cont,"LG CASE") || matchCont(cont,"LG BOX") || matchCont(cont,"LG PACK") || matchCont(cont,"LG PKG"))) {
            map[p_partkey[i]] = 2;
        }
    }
    return {std::move(map), max_partkey};
}

// --- Standard (SF1/SF10) ---
void runQ19Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "--- Running TPC-H Query 19 Benchmark ---" << std::endl;

    auto parseStart = std::chrono::high_resolution_clock::now();
    const std::string sf_path = g_dataset_path;

    // Build part group map
    auto [part_map, max_partkey] = buildPartGroupMap(sf_path + "part.tbl");

    // Load lineitem columns
    auto l_partkey       = loadIntColumn(sf_path + "lineitem.tbl", 1);
    auto l_quantity      = loadFloatColumn(sf_path + "lineitem.tbl", 4);
    auto l_extendedprice = loadFloatColumn(sf_path + "lineitem.tbl", 5);
    auto l_discount      = loadFloatColumn(sf_path + "lineitem.tbl", 6);
    auto l_shipmode      = loadCharColumn(sf_path + "lineitem.tbl", 14, 7);  // up to 7 chars for "REG AIR"
    auto l_shipinstruct  = loadCharColumn(sf_path + "lineitem.tbl", 13, 1);  // first char: 'D'=DELIVER IN PERSON

    // Pre-compute qualifies flag: shipmode IN ('AIR','REG AIR') AND shipinstruct = 'DELIVER IN PERSON'
    size_t nRows = l_partkey.size();
    std::vector<uint8_t> l_qualifies(nRows);
    for (size_t i = 0; i < nRows; i++) {
        const char* sm = &l_shipmode[i * 7];
        bool isAir    = (sm[0]=='A' && sm[1]=='I' && sm[2]=='R' && sm[3]=='\0');
        bool isRegAir = (sm[0]=='R' && sm[1]=='E' && sm[2]=='G' && sm[3]==' ' && sm[4]=='A' && sm[5]=='I' && sm[6]=='R');
        bool isDIP    = (l_shipinstruct[i] == 'D');
        l_qualifies[i] = ((isAir || isRegAir) && isDIP) ? 1 : 0;
    }
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double cpuParseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    uint dataSize = (uint)l_partkey.size();
    if (dataSize == 0) { std::cerr << "Q19: no data loaded" << std::endl; return; }
    std::cout << "Loaded " << dataSize << " lineitem rows for Q19." << std::endl;

    auto s1PSO = createPipeline(device, library, "q19_filter_and_sum_stage1");
    auto s2PSO = createPipeline(device, library, "q19_final_sum_stage2");
    if (!s1PSO || !s2PSO) return;

    const int numTG = 2048;
    uint mapSize = (uint)part_map.size();

    MTL::Buffer* partkeyBuf   = device->newBuffer(l_partkey.data(), dataSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* qtyBuf       = device->newBuffer(l_quantity.data(), dataSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* priceBuf     = device->newBuffer(l_extendedprice.data(), dataSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* discBuf      = device->newBuffer(l_discount.data(), dataSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* qualifiesBuf = device->newBuffer(l_qualifies.data(), dataSize * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* mapBuf       = device->newBuffer(part_map.data(), mapSize * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* partialBuf   = device->newBuffer(numTG * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* finalBuf     = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    double gpuSec = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

        enc->setComputePipelineState(s1PSO);
        enc->setBuffer(partkeyBuf, 0, 0);
        enc->setBuffer(qtyBuf, 0, 1);
        enc->setBuffer(priceBuf, 0, 2);
        enc->setBuffer(discBuf, 0, 3);
        enc->setBuffer(qualifiesBuf, 0, 4);
        enc->setBuffer(mapBuf, 0, 5);
        enc->setBuffer(partialBuf, 0, 6);
        enc->setBytes(&dataSize, sizeof(dataSize), 7);
        enc->setBytes(&mapSize, sizeof(mapSize), 8);
        NS::UInteger tgSize = s1PSO->maxTotalThreadsPerThreadgroup();
        if (tgSize > 1024) tgSize = 1024;
        enc->dispatchThreadgroups(MTL::Size::Make(numTG, 1, 1), MTL::Size::Make(tgSize, 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);
        const uint numTGs = numTG;
        enc->setComputePipelineState(s2PSO);
        enc->setBuffer(partialBuf, 0, 0);
        enc->setBuffer(finalBuf, 0, 1);
        enc->setBytes(&numTGs, sizeof(numTGs), 2);
        enc->dispatchThreads(MTL::Size::Make(1,1,1), MTL::Size::Make(1,1,1));
        enc->endEncoding();

        cb->commit();
        cb->waitUntilCompleted();
        if (iter == 2) gpuSec = cb->GPUEndTime() - cb->GPUStartTime();
    }

    auto cpuPostStart = std::chrono::high_resolution_clock::now();
    float revenue = *(float*)finalBuf->contents();
    auto cpuPostEnd = std::chrono::high_resolution_clock::now();
    double cpuPostMs = std::chrono::duration<double, std::milli>(cpuPostEnd - cpuPostStart).count();

    printf("\nTPC-H Q19 Result: Revenue = $%.2f\n", revenue);
    printf("\nQ19 | %u rows\n", dataSize);
    printTimingSummary(cpuParseMs, gpuSec * 1000.0, cpuPostMs);

    releaseAll(s1PSO, s2PSO, partkeyBuf, qtyBuf, priceBuf, discBuf,
               qualifiesBuf, mapBuf, partialBuf, finalBuf);
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

    // Build part group map on CPU
    auto buildT0 = std::chrono::high_resolution_clock::now();
    std::vector<int> p_partkey(partRows);
    parseIntColumnChunk(partFile, partIndex, 0, partRows, 0, p_partkey.data());
    std::vector<char> p_brand(partRows * 10);
    parseCharColumnChunkFixed(partFile, partIndex, 0, partRows, 3, 10, p_brand.data());
    std::vector<char> p_container(partRows * 10);
    parseCharColumnChunkFixed(partFile, partIndex, 0, partRows, 6, 10, p_container.data());
    std::vector<int> p_size(partRows);
    parseIntColumnChunk(partFile, partIndex, 0, partRows, 5, p_size.data());

    int max_partkey = 0;
    for (auto k : p_partkey) max_partkey = std::max(max_partkey, k);
    std::vector<uint8_t> part_map(max_partkey + 1, 0xFF);

    auto matchBrand = [](const char* b, const char* target, int len) {
        for (int i = 0; i < len; i++) if (b[i] != target[i]) return false;
        return true;
    };

    auto matchCont = [](const char* c, const char* target) {
        for (int i = 0; target[i]; i++) if (c[i] != target[i]) return false;
        return true;
    };

    for (size_t i = 0; i < partRows; i++) {
        const char* brand = &p_brand[i * 10];
        const char* cont  = &p_container[i * 10];
        int sz = p_size[i];
        if (matchBrand(brand, "Brand#12", 8) && sz >= 1 && sz <= 5
            && (matchCont(cont,"SM CASE") || matchCont(cont,"SM BOX") || matchCont(cont,"SM PACK") || matchCont(cont,"SM PKG")))
            part_map[p_partkey[i]] = 0;
        else if (matchBrand(brand, "Brand#23", 8) && sz >= 1 && sz <= 10
            && (matchCont(cont,"MED BAG") || matchCont(cont,"MED BOX") || matchCont(cont,"MED PKG") || matchCont(cont,"MED PACK")))
            part_map[p_partkey[i]] = 1;
        else if (matchBrand(brand, "Brand#34", 8) && sz >= 1 && sz <= 15
            && (matchCont(cont,"LG CASE") || matchCont(cont,"LG BOX") || matchCont(cont,"LG PACK") || matchCont(cont,"LG PKG")))
            part_map[p_partkey[i]] = 2;
    }
    auto buildT1 = std::chrono::high_resolution_clock::now();
    double buildMs = std::chrono::duration<double, std::milli>(buildT1 - buildT0).count();
    printf("Part group map: %d max_partkey (%.1f MB), build %.1f ms\n",
           max_partkey, part_map.size() / (1024.0*1024.0), buildMs);

    auto s1PSO = createPipeline(device, library, "q19_chunked_stage1");
    auto s2PSO = createPipeline(device, library, "q19_chunked_stage2");
    if (!s1PSO || !s2PSO) return;

    uint mapSize = (uint)part_map.size();
    MTL::Buffer* mapBuf = device->newBuffer(part_map.data(), mapSize * sizeof(uint8_t), MTL::ResourceStorageModeShared);

    // lineitem: partkey(4) + qty(4) + price(4) + discount(4) + qualifies(1) = 17 bytes/row
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 17, liRows);
    const uint num_tg = 2048;
    printf("Chunk size: %zu rows\n", chunkRows);

    const int NUM_SLOTS = 2;
    struct Q19Slot {
        MTL::Buffer* partkey; MTL::Buffer* quantity; MTL::Buffer* extprice;
        MTL::Buffer* discount; MTL::Buffer* qualifies;
    };
    Q19Slot slots[NUM_SLOTS];
    for (int s = 0; s < NUM_SLOTS; s++) {
        slots[s].partkey      = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].quantity     = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].extprice     = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].discount     = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].qualifies    = device->newBuffer(chunkRows * sizeof(uint8_t), MTL::ResourceStorageModeShared);
    }

    MTL::Buffer* partialBuf = device->newBuffer(num_tg * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* finalBuf   = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    double globalRevenue = 0.0;

    auto timing = chunkedStreamLoop(
        commandQueue, slots, NUM_SLOTS, liRows, chunkRows,
        // Parse
        [&](Q19Slot& slot, size_t startRow, size_t rowCount) {
            parseIntColumnChunk(liFile, liIndex, startRow, rowCount, 1, (int*)slot.partkey->contents());
            parseFloatColumnChunk(liFile, liIndex, startRow, rowCount, 4, (float*)slot.quantity->contents());
            parseFloatColumnChunk(liFile, liIndex, startRow, rowCount, 5, (float*)slot.extprice->contents());
            parseFloatColumnChunk(liFile, liIndex, startRow, rowCount, 6, (float*)slot.discount->contents());
            // Pre-compute qualifies flag on CPU: shipmode IN ('AIR','REG AIR') AND shipinstruct = 'DELIVER IN PERSON'
            std::vector<char> sm_buf(rowCount * 7);
            parseCharColumnChunkFixed(liFile, liIndex, startRow, rowCount, 14, 7, sm_buf.data());
            std::vector<char> si_buf(rowCount);
            parseCharColumnChunk(liFile, liIndex, startRow, rowCount, 13, si_buf.data());
            auto* qual = (uint8_t*)slot.qualifies->contents();
            for (size_t i = 0; i < rowCount; i++) {
                const char* sm = &sm_buf[i * 7];
                bool isAir    = (sm[0]=='A' && sm[1]=='I' && sm[2]=='R' && sm[3]=='\0');
                bool isRegAir = (sm[0]=='R' && sm[1]=='E' && sm[2]=='G' && sm[3]==' ' && sm[4]=='A' && sm[5]=='I' && sm[6]=='R');
                bool isDIP    = (si_buf[i] == 'D');
                qual[i] = ((isAir || isRegAir) && isDIP) ? 1 : 0;
            }
        },
        // Dispatch
        [&](Q19Slot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(s1PSO);
            enc->setBuffer(slot.partkey, 0, 0);
            enc->setBuffer(slot.quantity, 0, 1);
            enc->setBuffer(slot.extprice, 0, 2);
            enc->setBuffer(slot.discount, 0, 3);
            enc->setBuffer(slot.qualifies, 0, 4);
            enc->setBuffer(mapBuf, 0, 5);
            enc->setBuffer(partialBuf, 0, 6);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 7);
            enc->setBytes(&mapSize, sizeof(mapSize), 8);
            NS::UInteger tgSize = s1PSO->maxTotalThreadsPerThreadgroup();
            if (tgSize > 1024) tgSize = 1024;
            enc->dispatchThreadgroups(MTL::Size::Make(num_tg, 1, 1), MTL::Size::Make(tgSize, 1, 1));

            enc->memoryBarrier(MTL::BarrierScopeBuffers);
            enc->setComputePipelineState(s2PSO);
            enc->setBuffer(partialBuf, 0, 0);
            enc->setBuffer(finalBuf, 0, 1);
            enc->setBytes(&num_tg, sizeof(num_tg), 2);
            enc->dispatchThreads(MTL::Size::Make(1,1,1), MTL::Size::Make(1,1,1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        // Accumulate
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {
            globalRevenue += *(float*)finalBuf->contents();
        }
    );

    double allCpuParseMs = indexBuildMs + buildMs + timing.parseMs;
    printf("TPC-H Q19 Result: Revenue = $%.2f\n", globalRevenue);
    printf("\nSF100 Q19 | %zu chunks | %zu rows\n", timing.chunkCount, liRows);
    printTimingSummary(allCpuParseMs, timing.gpuMs, timing.postMs);

    releaseAll(s1PSO, s2PSO, mapBuf, partialBuf, finalBuf);
    for (int s = 0; s < NUM_SLOTS; s++)
        releaseAll(slots[s].partkey, slots[s].quantity, slots[s].extprice,
                   slots[s].discount, slots[s].qualifies);
}
