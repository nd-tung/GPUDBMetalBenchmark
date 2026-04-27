#include "infra.h"

// ===================================================================
// TPC-H Q1 — Pricing Summary Report
// ===================================================================

// --- Standard (SF1/SF10) ---
void runQ1Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "--- Running TPC-H Query 1 Benchmark ---" << std::endl;

    auto q1ParseStart = std::chrono::high_resolution_clock::now();
    const std::string filepath = g_dataset_path + "lineitem.tbl";
    auto cols = loadQueryColumns(device, filepath, {
        {4, ColType::FLOAT}, {5, ColType::FLOAT}, {6, ColType::FLOAT}, {7, ColType::FLOAT},
        {8, ColType::CHAR1}, {9, ColType::CHAR1}, {10, ColType::DATE}
    });
    auto q1ParseEnd = std::chrono::high_resolution_clock::now();
    double q1CpuParseMs = std::chrono::duration<double, std::milli>(q1ParseEnd - q1ParseStart).count();
    const uint data_size = (uint)cols.rows();
    if (data_size == 0) { std::cerr << "Q1: no data loaded" << std::endl; return; }

    // Create pipeline for fused single-pass Q1
    auto fusedPSO = createPipeline(device, library, "q1_fused_kernel");
    if (!fusedPSO) return;

    // Inputs from QueryColumns (zero-copy when .colbin v2 present).
    MTL::Buffer* shipdateBuffer = cols.buffer(10);
    MTL::Buffer* flagBuffer     = cols.buffer(8);
    MTL::Buffer* statusBuffer   = cols.buffer(9);
    MTL::Buffer* qtyBuffer      = cols.buffer(4);
    MTL::Buffer* priceBuffer    = cols.buffer(5);
    MTL::Buffer* discBuffer     = cols.buffer(6);
    MTL::Buffer* taxBuffer      = cols.buffer(7);

    // Output buffer: single packed [60] atomic_uint, stride 10 per bin.
    // Layout per bin: qty_lo, qty_hi, base_lo, base_hi, disc_lo, disc_hi,
    //                 charge_lo, charge_hi, disc_bp, count.
    const uint bins = 6;
    const uint kQ1AggsCount = bins * 10;  // 60 uints
    // Threadgroup count: 1024 TGs maximises in-flight occupancy on M1 / latency-hiding
    // for the lineitem scan. Atomic contention is bounded because the kernel does a
    // full TG-local reduction (`tg_reduce_*`) and only emits 6×ops atomic writes per TG.
    // We still cap by row count so tiny SF inputs don't dispatch idle TGs.
    uint num_threadgroups = 1024;
    {
        const uint tg_threads = 1024;
        uint required = (data_size + tg_threads - 1) / tg_threads;
        if (required < 32) required = 32;
        if (num_threadgroups > required) num_threadgroups = required;
    }

    MTL::Buffer* aggsBuffer = device->newBuffer(kQ1AggsCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);

    const int cutoffDate = 19980902; // DATE '1998-12-01' - INTERVAL '90' DAY

    // Dispatch kernel (2 warmup + 1 measured)
    double q1_gpu_ms = 0.0;
    
    for(int iter = 0; iter < 3; ++iter) {
        // Zero packed output buffer
        memset(aggsBuffer->contents(), 0, kQ1AggsCount * sizeof(uint32_t));

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = commandBuffer->computeCommandEncoder();
        
        enc->setComputePipelineState(fusedPSO);
        enc->setBuffer(shipdateBuffer, 0, 0);
        enc->setBuffer(flagBuffer, 0, 1);
        enc->setBuffer(statusBuffer, 0, 2);
        enc->setBuffer(qtyBuffer, 0, 3);
        enc->setBuffer(priceBuffer, 0, 4);
        enc->setBuffer(discBuffer, 0, 5);
        enc->setBuffer(taxBuffer, 0, 6);
        enc->setBuffer(aggsBuffer, 0, 7);
        enc->setBytes(&data_size, sizeof(data_size), 8);
        enc->setBytes(&cutoffDate, sizeof(cutoffDate), 9);
        NS::UInteger tgSize = fusedPSO->maxTotalThreadsPerThreadgroup();
        if (tgSize > 1024) tgSize = 1024;
        enc->dispatchThreadgroups(MTL::Size::Make(num_threadgroups, 1, 1), MTL::Size::Make(tgSize, 1, 1));
        enc->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        
        if (iter == 2) {
            q1_gpu_ms = (commandBuffer->GPUEndTime() - commandBuffer->GPUStartTime()) * 1000.0;
        }
    }

    // CPU post-processing (build final results) timing start
    auto q1_cpu_post_start = std::chrono::high_resolution_clock::now();

    // Read back final results from packed buffer
    const uint32_t* aggs = (const uint32_t*)aggsBuffer->contents();

    auto reconstruct_long = [](uint32_t lo, uint32_t hi) -> long {
        uint64_t v = ((uint64_t)hi << 32) | (uint64_t)lo;
        return (long)v;
    };

    struct Q1Result { double sum_qty, sum_base_price, sum_disc_price, sum_charge, avg_qty, avg_price, avg_disc; uint count; };
    std::map<std::pair<char,char>, Q1Result> final_results;
    auto emit_bin = [&](int rfIdx, int lsIdx, int bin){
        const uint32_t* row = aggs + bin * 10;
        uint count = row[9];
        if (count == 0) return;
        char rf = (rfIdx==0?'A':rfIdx==1?'N':'R');
        char ls = (lsIdx==0?'F':'O');
        Q1Result r;
        r.sum_qty        = (double)reconstruct_long(row[0], row[1]) / 100.0;
        r.sum_base_price = (double)reconstruct_long(row[2], row[3]) / 100.0;
        r.sum_disc_price = (double)reconstruct_long(row[4], row[5]) / 100.0;
        r.sum_charge     = (double)reconstruct_long(row[6], row[7]) / 100.0;
        r.count          = count;
        r.avg_qty   = r.sum_qty        / (double)r.count;
        r.avg_price = r.sum_base_price / (double)r.count;
        r.avg_disc  = ((double)row[8] / 100.0) / (double)r.count;
        final_results[{rf, ls}] = r;
    };
    emit_bin(0,0,0); // A/F
    emit_bin(0,1,1); // A/O
    emit_bin(1,0,2); // N/F
    emit_bin(1,1,3); // N/O
    emit_bin(2,0,4); // R/F
    emit_bin(2,1,5); // R/O

    auto q1_cpu_post_end = std::chrono::high_resolution_clock::now();
    double q1_cpu_ms = std::chrono::duration<double, std::milli>(q1_cpu_post_end - q1_cpu_post_start).count();

    printf("\n+----------+----------+------------+----------------+----------------+----------------+------------+------------+------------+----------+\n");
    printf("| l_return | l_linest |    sum_qty | sum_base_price | sum_disc_price |     sum_charge |    avg_qty |  avg_price |   avg_disc | count    |\n");
    printf("+----------+----------+------------+----------------+----------------+----------------+------------+------------+------------+----------+\n");
    for (auto const& [key, val] : final_results) {
        printf("| %8c | %8c | %10.2f | %14.2f | %14.2f | %14.2f | %10.2f | %10.2f | %10.2f | %8u |\n",
               key.first, key.second, val.sum_qty, val.sum_base_price, val.sum_disc_price, val.sum_charge,
               val.avg_qty, val.avg_price, val.avg_disc, val.count);
    }
    printf("+----------+----------+------------+----------------+----------------+----------------+------------+------------+------------+----------+\n");

    printf("\nQ1 | %u rows\n", data_size);
    printTimingSummary(q1CpuParseMs, q1_gpu_ms, q1_cpu_ms);

    releaseAll(fusedPSO, aggsBuffer);
    // Input buffers (shipdate/flag/status/qty/price/disc/tax) are owned by
    // `cols` (QueryColumns) and released on scope exit.
}


// --- SF100 Chunked ---
void runQ1BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q1 Benchmark (SF100 Chunked) ===" << std::endl;

    // Open lineitem TBL file via mmap
    MappedFile mf;
    if (!mf.open(g_dataset_path + "lineitem.tbl")) {
        std::cerr << "Q1 SF100: Cannot mmap lineitem.tbl" << std::endl;
        return;
    }

    std::cout << "Building line index for lineitem.tbl (" << mf.size / (1024*1024) << " MB)..." << std::endl;
    auto indexStart = std::chrono::high_resolution_clock::now();
    auto lineIndex = buildLineIndex(mf);
    auto indexEnd = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(indexEnd - indexStart).count();
    size_t totalRows = lineIndex.size();
    printf("Indexed %zu rows in %.1f ms\n", totalRows, indexBuildMs);

    // Compute chunk size: Q1 needs 7 columns ~38 bytes/row
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 38, totalRows);
    const uint num_tg = 1024;
    printf("Chunk size: %zu rows, total: %zu rows\n", chunkRows, totalRows);

    // Create pipeline states
    auto s1PSO = createPipeline(device, library, "q1_chunked_stage1");
    auto s2PSO = createPipeline(device, library, "q1_chunked_stage2");
    if (!s1PSO || !s2PSO) return;

    // Allocate double-buffered column buffers (2 slots)
    const int NUM_SLOTS = 2;
    struct ChunkSlot {
        MTL::Buffer* shipdate; MTL::Buffer* returnflag; MTL::Buffer* linestatus;
        MTL::Buffer* quantity; MTL::Buffer* extprice; MTL::Buffer* discount; MTL::Buffer* tax;
    };
    ChunkSlot slots[NUM_SLOTS];
    for (int s = 0; s < NUM_SLOTS; s++) {
        slots[s].shipdate   = device->newBuffer(chunkRows * sizeof(int),   MTL::ResourceStorageModeShared);
        slots[s].returnflag = device->newBuffer(chunkRows * sizeof(char),  MTL::ResourceStorageModeShared);
        slots[s].linestatus = device->newBuffer(chunkRows * sizeof(char),  MTL::ResourceStorageModeShared);
        slots[s].quantity   = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].extprice   = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].discount   = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].tax        = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
    }

    // Partial result buffers (reused per chunk) and global accumulators
    const uint bins = 6;
    MTL::Buffer* p_qtyCents  = device->newBuffer(num_tg * bins * sizeof(long), MTL::ResourceStorageModeShared);
    MTL::Buffer* p_baseCents = device->newBuffer(num_tg * bins * sizeof(long), MTL::ResourceStorageModeShared);
    MTL::Buffer* p_discCents = device->newBuffer(num_tg * bins * sizeof(long), MTL::ResourceStorageModeShared);
    MTL::Buffer* p_chargeCents = device->newBuffer(num_tg * bins * sizeof(long), MTL::ResourceStorageModeShared);
    MTL::Buffer* p_discountBP  = device->newBuffer(num_tg * bins * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* p_counts      = device->newBuffer(num_tg * bins * sizeof(uint32_t), MTL::ResourceStorageModeShared);

    MTL::Buffer* f_qtyCents  = device->newBuffer(bins * sizeof(long), MTL::ResourceStorageModeShared);
    MTL::Buffer* f_baseCents = device->newBuffer(bins * sizeof(long), MTL::ResourceStorageModeShared);
    MTL::Buffer* f_discCents = device->newBuffer(bins * sizeof(long), MTL::ResourceStorageModeShared);
    MTL::Buffer* f_chargeCents = device->newBuffer(bins * sizeof(long), MTL::ResourceStorageModeShared);
    MTL::Buffer* f_discountBP  = device->newBuffer(bins * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* f_counts      = device->newBuffer(bins * sizeof(uint32_t), MTL::ResourceStorageModeShared);

    // Global CPU-side accumulators (accumulate across chunks)
    long g_sum_qty[6]={}, g_sum_base[6]={}, g_sum_disc[6]={}, g_sum_charge[6]={};
    uint32_t g_sum_discbp[6]={}, g_count[6]={};

    const int cutoffDate = 19980902;

    auto timing = chunkedStreamLoop(
        commandQueue, slots, NUM_SLOTS, totalRows, chunkRows,
        // Parse
        [&](ChunkSlot& slot, size_t startRow, size_t rowCount) {
            parseDateColumnChunk(mf, lineIndex, startRow, rowCount, 10, (int*)slot.shipdate->contents());
            parseCharColumnChunk(mf, lineIndex, startRow, rowCount, 8,  (char*)slot.returnflag->contents());
            parseCharColumnChunk(mf, lineIndex, startRow, rowCount, 9,  (char*)slot.linestatus->contents());
            parseFloatColumnChunk(mf, lineIndex, startRow, rowCount, 4, (float*)slot.quantity->contents());
            parseFloatColumnChunk(mf, lineIndex, startRow, rowCount, 5, (float*)slot.extprice->contents());
            parseFloatColumnChunk(mf, lineIndex, startRow, rowCount, 6, (float*)slot.discount->contents());
            parseFloatColumnChunk(mf, lineIndex, startRow, rowCount, 7, (float*)slot.tax->contents());
        },
        // Dispatch
        [&](ChunkSlot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            // Zero partials before dispatch
            memset(p_qtyCents->contents(), 0, num_tg * bins * sizeof(long));
            memset(p_baseCents->contents(), 0, num_tg * bins * sizeof(long));
            memset(p_discCents->contents(), 0, num_tg * bins * sizeof(long));
            memset(p_chargeCents->contents(), 0, num_tg * bins * sizeof(long));
            memset(p_discountBP->contents(), 0, num_tg * bins * sizeof(uint32_t));
            memset(p_counts->contents(), 0, num_tg * bins * sizeof(uint32_t));
            memset(f_qtyCents->contents(), 0, bins * sizeof(long));
            memset(f_baseCents->contents(), 0, bins * sizeof(long));
            memset(f_discCents->contents(), 0, bins * sizeof(long));
            memset(f_chargeCents->contents(), 0, bins * sizeof(long));
            memset(f_discountBP->contents(), 0, bins * sizeof(uint32_t));
            memset(f_counts->contents(), 0, bins * sizeof(uint32_t));

            auto enc = cmdBuf->computeCommandEncoder();
            // Stage 1
            enc->setComputePipelineState(s1PSO);
            enc->setBuffer(slot.shipdate, 0, 0);
            enc->setBuffer(slot.returnflag, 0, 1);
            enc->setBuffer(slot.linestatus, 0, 2);
            enc->setBuffer(slot.quantity, 0, 3);
            enc->setBuffer(slot.extprice, 0, 4);
            enc->setBuffer(slot.discount, 0, 5);
            enc->setBuffer(slot.tax, 0, 6);
            enc->setBuffer(p_qtyCents, 0, 7);
            enc->setBuffer(p_baseCents, 0, 8);
            enc->setBuffer(p_discCents, 0, 9);
            enc->setBuffer(p_chargeCents, 0, 10);
            enc->setBuffer(p_discountBP, 0, 11);
            enc->setBuffer(p_counts, 0, 12);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 13);
            enc->setBytes(&cutoffDate, sizeof(cutoffDate), 14);
            enc->setBytes(&num_tg, sizeof(num_tg), 15);
            NS::UInteger tgSize = s1PSO->maxTotalThreadsPerThreadgroup();
            if (tgSize > 1024) tgSize = 1024;
            enc->dispatchThreadgroups(MTL::Size::Make(num_tg, 1, 1), MTL::Size::Make(tgSize, 1, 1));
            enc->memoryBarrier(MTL::BarrierScopeBuffers);
            // Stage 2
            enc->setComputePipelineState(s2PSO);
            enc->setBuffer(p_qtyCents, 0, 0);
            enc->setBuffer(p_baseCents, 0, 1);
            enc->setBuffer(p_discCents, 0, 2);
            enc->setBuffer(p_chargeCents, 0, 3);
            enc->setBuffer(p_discountBP, 0, 4);
            enc->setBuffer(p_counts, 0, 5);
            enc->setBuffer(f_qtyCents, 0, 6);
            enc->setBuffer(f_baseCents, 0, 7);
            enc->setBuffer(f_discCents, 0, 8);
            enc->setBuffer(f_chargeCents, 0, 9);
            enc->setBuffer(f_discountBP, 0, 10);
            enc->setBuffer(f_counts, 0, 11);
            enc->setBytes(&num_tg, sizeof(num_tg), 12);
            enc->dispatchThreads(MTL::Size::Make(1, 1, 1), MTL::Size::Make(1, 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        // Accumulate
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {
            long* cQty = (long*)f_qtyCents->contents();
            long* cBase = (long*)f_baseCents->contents();
            long* cDisc = (long*)f_discCents->contents();
            long* cCharge = (long*)f_chargeCents->contents();
            uint32_t* cDiscBP = (uint32_t*)f_discountBP->contents();
            uint32_t* cCounts = (uint32_t*)f_counts->contents();
            for (int b = 0; b < 6; b++) {
                g_sum_qty[b] += cQty[b]; g_sum_base[b] += cBase[b];
                g_sum_disc[b] += cDisc[b]; g_sum_charge[b] += cCharge[b];
                g_sum_discbp[b] += cDiscBP[b]; g_count[b] += cCounts[b];
            }
        }
    );

    // CPU post-processing: compute averages from accumulators
    auto cpuPostFinalStart = std::chrono::high_resolution_clock::now();
    struct Q1R { double sum_qty, sum_base, sum_disc, sum_charge, avg_qty, avg_price, avg_disc; uint cnt; };
    auto emit = [&]([[maybe_unused]] int rfi, [[maybe_unused]] int lsi, int bin) -> Q1R {
        Q1R r = {};
        if (g_count[bin] == 0) return r;
        r.sum_qty = (double)g_sum_qty[bin] / 100.0;
        r.sum_base = (double)g_sum_base[bin] / 100.0;
        r.sum_disc = (double)g_sum_disc[bin] / 100.0;
        r.sum_charge = (double)g_sum_charge[bin] / 100.0;
        r.cnt = g_count[bin];
        r.avg_qty = r.sum_qty / r.cnt;
        r.avg_price = r.sum_base / r.cnt;
        r.avg_disc = ((double)g_sum_discbp[bin] / 100.0) / r.cnt;
        return r;
    };
    char rfChars[] = {'A','A','N','N','R','R'};
    char lsChars[] = {'F','O','F','O','F','O'};
    auto cpuPostFinalEnd = std::chrono::high_resolution_clock::now();
    double cpuPostFinalMs = std::chrono::duration<double, std::milli>(cpuPostFinalEnd - cpuPostFinalStart).count();

    printf("\n+----------+----------+------------+----------------+----------------+----------------+------------+------------+------------+----------+\n");
    printf("| l_return | l_linest |    sum_qty | sum_base_price | sum_disc_price |     sum_charge |    avg_qty |  avg_price |   avg_disc | count    |\n");
    printf("+----------+----------+------------+----------------+----------------+----------------+------------+------------+------------+----------+\n");
    for (int b = 0; b < 6; b++) {
        Q1R r = emit(b/2, b%2, b);
        if (r.cnt > 0) {
            printf("| %8c | %8c | %10.2f | %14.2f | %14.2f | %14.2f | %10.2f | %10.2f | %10.2f | %8u |\n",
                   rfChars[b], lsChars[b], r.sum_qty, r.sum_base, r.sum_disc, r.sum_charge,
                   r.avg_qty, r.avg_price, r.avg_disc, r.cnt);
        }
    }
    printf("+----------+----------+------------+----------------+----------------+----------------+------------+------------+------------+----------+\n");

    double totalCpuPostAllMs = timing.postMs + cpuPostFinalMs;
    double allCpuParseMs = indexBuildMs + timing.parseMs;
    printf("\nSF100 Q1 | %zu chunks | %zu rows\n", timing.chunkCount, totalRows);
    printTimingSummary(allCpuParseMs, timing.gpuMs, totalCpuPostAllMs);

    // Cleanup
    releaseAll(s1PSO, s2PSO);
    for (int s = 0; s < NUM_SLOTS; s++)
        releaseAll(slots[s].shipdate, slots[s].returnflag, slots[s].linestatus,
                   slots[s].quantity, slots[s].extprice, slots[s].discount, slots[s].tax);
    releaseAll(p_qtyCents, p_baseCents, p_discCents, p_chargeCents, p_discountBP, p_counts,
              f_qtyCents, f_baseCents, f_discCents, f_chargeCents, f_discountBP, f_counts);
}
