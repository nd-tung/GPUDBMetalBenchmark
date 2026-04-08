#include "infra.h"

// ===================================================================
// TPC-H Q22 — Global Sales Opportunity
// ===================================================================

// Extract 2-digit phone prefix from char column
static std::vector<int> extractPhonePrefix(const std::vector<char>& phone_chars, int width, size_t count) {
    std::vector<int> prefixes(count);
    for (size_t i = 0; i < count; i++) {
        const char* p = phone_chars.data() + i * width;
        prefixes[i] = (p[0] - '0') * 10 + (p[1] - '0');
    }
    return prefixes;
}

void runQ22Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n--- Running TPC-H Query 22 Benchmark ---" << std::endl;

    const std::string sf_path = g_dataset_path;

    auto parseStart = std::chrono::high_resolution_clock::now();
    auto cCols = loadColumnsMulti(sf_path + "customer.tbl", {{0, ColType::INT}, {4, ColType::CHAR_FIXED, 15}, {5, ColType::FLOAT}});
    auto& c_custkey = cCols.ints(0); auto& c_phone = cCols.chars(4); auto& c_acctbal = cCols.floats(5);
    auto oCols = loadColumnsMulti(sf_path + "orders.tbl", {{1, ColType::INT}});
    auto& o_custkey = oCols.ints(1);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double cpuParseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    uint custSize = (uint)c_custkey.size();

    auto c_prefix = extractPhonePrefix(c_phone, 15, custSize);

    // Valid prefixes: 13, 17, 18, 23, 29, 30, 31
    const int valid_prefixes[] = {13, 17, 18, 23, 29, 30, 31};
    uint valid_prefix_mask = 0;
    int prefix_to_bin[32];
    memset(prefix_to_bin, -1, sizeof(prefix_to_bin));
    for (int i = 0; i < 7; i++) {
        valid_prefix_mask |= (1u << valid_prefixes[i]);
        prefix_to_bin[valid_prefixes[i]] = i;
    }

    // CPU pre-compute: avg balance for valid-prefix customers with bal > 0
    double sumBal = 0.0;
    int countBal = 0;
    for (uint i = 0; i < custSize; i++) {
        if (c_acctbal[i] > 0.0f) {
            int prefix = c_prefix[i];
            if (prefix >= 0 && prefix <= 31 && ((valid_prefix_mask >> (uint)prefix) & 1u))
            { sumBal += c_acctbal[i]; countBal++; }
        }
    }
    float avgBal = (countBal > 0) ? (float)(sumBal / countBal) : 0.0f;

    // CPU pre-build: orders custkey bitmap
    int max_custkey = 0;
    for (int k : c_custkey) max_custkey = std::max(max_custkey, k);
    uint cust_bitmap_ints = (max_custkey + 31) / 32 + 1;
    std::vector<uint32_t> custBitmap(cust_bitmap_ints, 0);
    for (int ck : o_custkey)
        if (ck >= 0 && ck <= max_custkey)
            custBitmap[ck / 32] |= (1u << (ck % 32));

    auto pFinalPipe = createPipeline(device, library, "q22_final_aggregate_kernel");
    if (!pFinalPipe) return;

    MTL::Buffer* pPrefixBuf = device->newBuffer(c_prefix.data(), custSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pAcctBalBuf = device->newBuffer(c_acctbal.data(), custSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCustKeyBuf = device->newBuffer(c_custkey.data(), custSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCustBitmapBuf = device->newBuffer(custBitmap.data(), cust_bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* pResultCountBuf = device->newBuffer(7 * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* pResultSumBuf = device->newBuffer(7 * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPrefixToBinBuf = device->newBuffer(prefix_to_bin, 32 * sizeof(int), MTL::ResourceStorageModeShared);

    double gpuSec = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(pResultCountBuf->contents(), 0, 7 * sizeof(uint));
        memset(pResultSumBuf->contents(), 0, 7 * sizeof(float));

        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pFinalPipe);
        enc->setBuffer(pPrefixBuf, 0, 0);
        enc->setBuffer(pAcctBalBuf, 0, 1);
        enc->setBuffer(pCustKeyBuf, 0, 2);
        enc->setBuffer(pCustBitmapBuf, 0, 3);
        enc->setBuffer(pResultCountBuf, 0, 4);
        enc->setBuffer(pResultSumBuf, 0, 5);
        enc->setBytes(&custSize, sizeof(custSize), 6);
        enc->setBytes(&avgBal, sizeof(avgBal), 7);
        enc->setBytes(&valid_prefix_mask, sizeof(valid_prefix_mask), 8);
        enc->setBuffer(pPrefixToBinBuf, 0, 9);
        enc->dispatchThreads(MTL::Size(custSize, 1, 1), MTL::Size(256, 1, 1));
        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
        if (iter == 2)
            gpuSec = cb->GPUEndTime() - cb->GPUStartTime();
    }

    auto postStart = std::chrono::high_resolution_clock::now();
    uint* counts = (uint*)pResultCountBuf->contents();
    float* sums = (float*)pResultSumBuf->contents();
    printf("\nTPC-H Q22 Results:\n");
    printf("+----------+----------+---------------+\n");
    printf("| cntrycode|  numcust |    totacctbal |\n");
    printf("+----------+----------+---------------+\n");
    for (int i = 0; i < 7; i++) {
        if (counts[i] > 0) {
            printf("| %8d | %8u | %13.2f |\n", valid_prefixes[i], counts[i], sums[i]);
        }
    }
    printf("+----------+----------+---------------+\n");
    auto postEnd = std::chrono::high_resolution_clock::now();
    double cpuPostMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nQ22 | %u customers\n", custSize);
    printTimingSummary(cpuParseMs, gpuSec * 1000.0, cpuPostMs);

    releaseAll(pFinalPipe, pPrefixBuf, pAcctBalBuf, pCustKeyBuf,
              pCustBitmapBuf, pResultCountBuf, pResultSumBuf, pPrefixToBinBuf);
}

// --- SF100 Chunked ---
void runQ22BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q22 Benchmark (SF100 Chunked) ===" << std::endl;

    MappedFile custFile, ordFile;
    if (!custFile.open(g_dataset_path + "customer.tbl") ||
        !ordFile.open(g_dataset_path + "orders.tbl")) {
        std::cerr << "Q22 SF100: Cannot open required TBL files" << std::endl;
        return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto custIdx = buildLineIndex(custFile);
    auto ordIdx = buildLineIndex(ordFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    auto bpT0 = std::chrono::high_resolution_clock::now();
    size_t custRows = custIdx.size(), ordRows = ordIdx.size();

    // Load all customer columns (relatively small even at SF100: 15M rows)
    std::vector<int> c_custkey(custRows);
    std::vector<float> c_acctbal(custRows);
    std::vector<char> c_phone(custRows * 15);
    parseIntColumnChunk(custFile, custIdx, 0, custRows, 0, c_custkey.data());
    parseCharColumnChunkFixed(custFile, custIdx, 0, custRows, 4, 15, c_phone.data());
    parseFloatColumnChunk(custFile, custIdx, 0, custRows, 5, c_acctbal.data());

    std::vector<int> c_prefix(custRows);
    for (size_t i = 0; i < custRows; i++)
        c_prefix[i] = (c_phone[i * 15] - '0') * 10 + (c_phone[i * 15 + 1] - '0');

    const int valid_prefixes[] = {13, 17, 18, 23, 29, 30, 31};
    uint valid_prefix_mask = 0;
    int prefix_to_bin[32];
    memset(prefix_to_bin, -1, sizeof(prefix_to_bin));
    for (int i = 0; i < 7; i++) {
        valid_prefix_mask |= (1u << valid_prefixes[i]);
        prefix_to_bin[valid_prefixes[i]] = i;
    }

    int max_custkey = 0;
    for (size_t i = 0; i < custRows; i++) max_custkey = std::max(max_custkey, c_custkey[i]);
    uint cust_bitmap_ints = (max_custkey + 31) / 32 + 1;
    auto bpT1 = std::chrono::high_resolution_clock::now();
    double buildParseMs = indexBuildMs + std::chrono::duration<double, std::milli>(bpT1 - bpT0).count();

    uint custSize = (uint)custRows;

    auto pAvgPipe = createPipeline(device, library, "q22_avg_balance_kernel");
    auto pBitmapPipe = createPipeline(device, library, "q22_build_orders_bitmap_kernel");
    auto pFinalPipe = createPipeline(device, library, "q22_final_aggregate_kernel");
    if (!pAvgPipe || !pBitmapPipe || !pFinalPipe) return;

    MTL::Buffer* pPrefixBuf = device->newBuffer(c_prefix.data(), custSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pAcctBalBuf = device->newBuffer(c_acctbal.data(), custSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCustKeyBuf = device->newBuffer(c_custkey.data(), custSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSumBalBuf = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCountBalBuf = device->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCustBitmapBuf = device->newBuffer(cust_bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* pResultCountBuf = device->newBuffer(7 * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* pResultSumBuf = device->newBuffer(7 * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPrefixToBinBuf = device->newBuffer(prefix_to_bin, 32 * sizeof(int), MTL::ResourceStorageModeShared);

    // Phase 1: avg balance (single pass over customers)
    *(float*)pSumBalBuf->contents() = 0.0f;
    *(uint*)pCountBalBuf->contents() = 0;
    {
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pAvgPipe);
        enc->setBuffer(pPrefixBuf, 0, 0);
        enc->setBuffer(pAcctBalBuf, 0, 1);
        enc->setBuffer(pSumBalBuf, 0, 2);
        enc->setBuffer(pCountBalBuf, 0, 3);
        enc->setBytes(&custSize, sizeof(custSize), 4);
        enc->setBytes(&valid_prefix_mask, sizeof(valid_prefix_mask), 5);
        enc->dispatchThreads(MTL::Size(custSize, 1, 1), MTL::Size(256, 1, 1));
        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
    }
    float sumBal = *(float*)pSumBalBuf->contents();
    uint countBal = *(uint*)pCountBalBuf->contents();
    float avgBal = (countBal > 0) ? sumBal / countBal : 0.0f;

    // Phase 2: stream orders to build bitmap
    memset(pCustBitmapBuf->contents(), 0, cust_bitmap_ints * sizeof(uint));
    size_t ordChunkRows = ChunkConfig::adaptiveChunkSize(device, 4, ordRows);
    struct Q22OrdSlot { MTL::Buffer* custkey; };
    Q22OrdSlot ordSlots[2];
    for (int s = 0; s < 2; s++)
        ordSlots[s].custkey = device->newBuffer(ordChunkRows * sizeof(int), MTL::ResourceStorageModeShared);

    auto bitmapTiming = chunkedStreamLoop(
        commandQueue, ordSlots, 2, ordRows, ordChunkRows,
        [&](Q22OrdSlot& slot, size_t startRow, size_t rowCount) {
            parseIntColumnChunk(ordFile, ordIdx, startRow, rowCount, 1, (int*)slot.custkey->contents());
        },
        [&](Q22OrdSlot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(pBitmapPipe);
            enc->setBuffer(slot.custkey, 0, 0);
            enc->setBuffer(pCustBitmapBuf, 0, 1);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 2);
            enc->dispatchThreads(MTL::Size(chunkSize, 1, 1), MTL::Size(256, 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {}
    );

    // Phase 3: final aggregate (single pass over customers)
    memset(pResultCountBuf->contents(), 0, 7 * sizeof(uint));
    memset(pResultSumBuf->contents(), 0, 7 * sizeof(float));
    double finalGpuMs = 0;
    {
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pFinalPipe);
        enc->setBuffer(pPrefixBuf, 0, 0);
        enc->setBuffer(pAcctBalBuf, 0, 1);
        enc->setBuffer(pCustKeyBuf, 0, 2);
        enc->setBuffer(pCustBitmapBuf, 0, 3);
        enc->setBuffer(pResultCountBuf, 0, 4);
        enc->setBuffer(pResultSumBuf, 0, 5);
        enc->setBytes(&custSize, sizeof(custSize), 6);
        enc->setBytes(&avgBal, sizeof(avgBal), 7);
        enc->setBytes(&valid_prefix_mask, sizeof(valid_prefix_mask), 8);
        enc->setBuffer(pPrefixToBinBuf, 0, 9);
        enc->dispatchThreads(MTL::Size(custSize, 1, 1), MTL::Size(256, 1, 1));
        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
        finalGpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
    }

    uint* counts = (uint*)pResultCountBuf->contents();
    float* sums = (float*)pResultSumBuf->contents();
    printf("\nTPC-H Q22 Results:\n");
    printf("+----------+----------+---------------+\n");
    printf("| cntrycode|  numcust |    totacctbal |\n");
    printf("+----------+----------+---------------+\n");
    for (int i = 0; i < 7; i++) {
        if (counts[i] > 0)
            printf("| %8d | %8u | %13.2f |\n", valid_prefixes[i], counts[i], sums[i]);
    }
    printf("+----------+----------+---------------+\n");

    printf("\nSF100 Q22 | %zu customers | %zu orders | %zu chunks\n", custRows, ordRows, bitmapTiming.chunkCount);
    printTimingSummary(buildParseMs + bitmapTiming.parseMs, bitmapTiming.gpuMs + finalGpuMs, 0.0);

    releaseAll(pAvgPipe, pBitmapPipe, pFinalPipe, pPrefixBuf, pAcctBalBuf, pCustKeyBuf,
              pSumBalBuf, pCountBalBuf, pCustBitmapBuf, pResultCountBuf, pResultSumBuf, pPrefixToBinBuf);
    for (int s = 0; s < 2; s++) releaseAll(ordSlots[s].custkey);
}
