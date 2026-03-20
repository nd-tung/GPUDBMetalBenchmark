#include "infra.h"

// ===================================================================
// TPC-H Q1 — Pricing Summary Report
// ===================================================================

// --- Standard (SF1/SF10) ---
void runQ1Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "--- Running TPC-H Query 1 Benchmark ---" << std::endl;

    auto q1ParseStart = std::chrono::high_resolution_clock::now();
    const std::string filepath = g_dataset_path + "lineitem.tbl";
    auto l_returnflag = loadCharColumn(filepath, 8), l_linestatus = loadCharColumn(filepath, 9);
    auto l_quantity = loadFloatColumn(filepath, 4), l_extendedprice = loadFloatColumn(filepath, 5);
    auto l_discount = loadFloatColumn(filepath, 6), l_tax = loadFloatColumn(filepath, 7);
    auto l_shipdate = loadDateColumn(filepath, 10);
    auto q1ParseEnd = std::chrono::high_resolution_clock::now();
    double q1CpuParseMs = std::chrono::duration<double, std::milli>(q1ParseEnd - q1ParseStart).count();
    const uint data_size = (uint)l_shipdate.size();
    if (data_size == 0) { std::cerr << "Q1: no data loaded" << std::endl; return; }

    // Create pipelines for Integer-cent two-pass Q1
    auto stage1PSO = createPipeline(device, library, "q1_bins_accumulate_int_stage1");
    auto stage2PSO = createPipeline(device, library, "q1_bins_reduce_int_stage2");
    if (!stage1PSO || !stage2PSO) return;

    // Create buffers for columns
    MTL::Buffer* shipdateBuffer = device->newBuffer(l_shipdate.data(), data_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* flagBuffer = device->newBuffer(l_returnflag.data(), data_size * sizeof(char), MTL::ResourceStorageModeShared);
    MTL::Buffer* statusBuffer = device->newBuffer(l_linestatus.data(), data_size * sizeof(char), MTL::ResourceStorageModeShared);
    MTL::Buffer* qtyBuffer = device->newBuffer(l_quantity.data(), data_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* priceBuffer = device->newBuffer(l_extendedprice.data(), data_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* discBuffer = device->newBuffer(l_discount.data(), data_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* taxBuffer = device->newBuffer(l_tax.data(), data_size * sizeof(float), MTL::ResourceStorageModeShared);

    // Buffers for two-pass integer-cent path
    const uint bins = 6;
    const uint num_threadgroups = 1024; // also passed to stage2

    // Stage 1 partials: size = num_threadgroups * bins
    MTL::Buffer* p_sumQtyCents = device->newBuffer(num_threadgroups * bins * sizeof(long), MTL::ResourceStorageModeShared);
    MTL::Buffer* p_sumBaseCents = device->newBuffer(num_threadgroups * bins * sizeof(long), MTL::ResourceStorageModeShared);
    MTL::Buffer* p_sumDiscPriceCents = device->newBuffer(num_threadgroups * bins * sizeof(long), MTL::ResourceStorageModeShared);
    MTL::Buffer* p_sumChargeCents = device->newBuffer(num_threadgroups * bins * sizeof(long), MTL::ResourceStorageModeShared);
    MTL::Buffer* p_sumDiscountBP = device->newBuffer(num_threadgroups * bins * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* p_counts = device->newBuffer(num_threadgroups * bins * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    // Zero initialize partials (defensive)
    memset(p_sumQtyCents->contents(), 0, num_threadgroups * bins * sizeof(long));
    memset(p_sumBaseCents->contents(), 0, num_threadgroups * bins * sizeof(long));
    memset(p_sumDiscPriceCents->contents(), 0, num_threadgroups * bins * sizeof(long));
    memset(p_sumChargeCents->contents(), 0, num_threadgroups * bins * sizeof(long));
    memset(p_sumDiscountBP->contents(), 0, num_threadgroups * bins * sizeof(uint32_t));
    memset(p_counts->contents(), 0, num_threadgroups * bins * sizeof(uint32_t));

    // Stage 2 finals: size = bins
    MTL::Buffer* f_sumQtyCents = device->newBuffer(bins * sizeof(long), MTL::ResourceStorageModeShared);
    MTL::Buffer* f_sumBaseCents = device->newBuffer(bins * sizeof(long), MTL::ResourceStorageModeShared);
    MTL::Buffer* f_sumDiscPriceCents = device->newBuffer(bins * sizeof(long), MTL::ResourceStorageModeShared);
    MTL::Buffer* f_sumChargeCents = device->newBuffer(bins * sizeof(long), MTL::ResourceStorageModeShared);
    MTL::Buffer* f_sumDiscountBP = device->newBuffer(bins * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* f_counts = device->newBuffer(bins * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    memset(f_sumQtyCents->contents(), 0, bins * sizeof(long));
    memset(f_sumBaseCents->contents(), 0, bins * sizeof(long));
    memset(f_sumDiscPriceCents->contents(), 0, bins * sizeof(long));
    memset(f_sumChargeCents->contents(), 0, bins * sizeof(long));
    memset(f_sumDiscountBP->contents(), 0, bins * sizeof(uint32_t));
    memset(f_counts->contents(), 0, bins * sizeof(uint32_t));

    const int cutoffDate = 19980902; // DATE '1998-12-01' - INTERVAL '90' DAY

    // Dispatch kernels (2 warmup + 1 measured)
    double q1_gpu_ms = 0.0;
    
    for(int iter = 0; iter < 3; ++iter) {
        // Reset partials and finals
        memset(p_sumQtyCents->contents(), 0, num_threadgroups * bins * sizeof(long));
        memset(p_sumBaseCents->contents(), 0, num_threadgroups * bins * sizeof(long));
        memset(p_sumDiscPriceCents->contents(), 0, num_threadgroups * bins * sizeof(long));
        memset(p_sumChargeCents->contents(), 0, num_threadgroups * bins * sizeof(long));
        memset(p_sumDiscountBP->contents(), 0, num_threadgroups * bins * sizeof(uint32_t));
        memset(p_counts->contents(), 0, num_threadgroups * bins * sizeof(uint32_t));
        
        memset(f_sumQtyCents->contents(), 0, bins * sizeof(long));
        memset(f_sumBaseCents->contents(), 0, bins * sizeof(long));
        memset(f_sumDiscPriceCents->contents(), 0, bins * sizeof(long));
        memset(f_sumChargeCents->contents(), 0, bins * sizeof(long));
        memset(f_sumDiscountBP->contents(), 0, bins * sizeof(uint32_t));
        memset(f_counts->contents(), 0, bins * sizeof(uint32_t));

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = commandBuffer->computeCommandEncoder();
        
        // Stage 1: accumulate partials
        enc->setComputePipelineState(stage1PSO);
        enc->setBuffer(shipdateBuffer, 0, 0);
        enc->setBuffer(flagBuffer, 0, 1);
        enc->setBuffer(statusBuffer, 0, 2);
        enc->setBuffer(qtyBuffer, 0, 3);
        enc->setBuffer(priceBuffer, 0, 4);
        enc->setBuffer(discBuffer, 0, 5);
        enc->setBuffer(taxBuffer, 0, 6);
        enc->setBuffer(p_sumQtyCents, 0, 7);
        enc->setBuffer(p_sumBaseCents, 0, 8);
        enc->setBuffer(p_sumDiscPriceCents, 0, 9);
        enc->setBuffer(p_sumChargeCents, 0, 10);
        enc->setBuffer(p_sumDiscountBP, 0, 11);
        enc->setBuffer(p_counts, 0, 12);
        enc->setBytes(&data_size, sizeof(data_size), 13);
        enc->setBytes(&cutoffDate, sizeof(cutoffDate), 14);
        enc->setBytes(&num_threadgroups, sizeof(num_threadgroups), 15);
        NS::UInteger tgSize = stage1PSO->maxTotalThreadsPerThreadgroup();
        if (tgSize > 1024) tgSize = 1024; // matches shared arrays in kernel
        enc->dispatchThreadgroups(MTL::Size::Make(num_threadgroups, 1, 1), MTL::Size::Make(tgSize, 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);
        // Stage 2: reduce partials to finals on the same encoder
        enc->setComputePipelineState(stage2PSO);
        enc->setBuffer(p_sumQtyCents, 0, 0);
        enc->setBuffer(p_sumBaseCents, 0, 1);
        enc->setBuffer(p_sumDiscPriceCents, 0, 2);
        enc->setBuffer(p_sumChargeCents, 0, 3);
        enc->setBuffer(p_sumDiscountBP, 0, 4);
        enc->setBuffer(p_counts, 0, 5);
        enc->setBuffer(f_sumQtyCents, 0, 6);
        enc->setBuffer(f_sumBaseCents, 0, 7);
        enc->setBuffer(f_sumDiscPriceCents, 0, 8);
        enc->setBuffer(f_sumChargeCents, 0, 9);
        enc->setBuffer(f_sumDiscountBP, 0, 10);
        enc->setBuffer(f_counts, 0, 11);
        enc->setBytes(&num_threadgroups, sizeof(num_threadgroups), 12);
        enc->dispatchThreads(MTL::Size::Make(1, 1, 1), MTL::Size::Make(1, 1, 1));
        enc->endEncoding();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        
        if (iter == 2) {
            q1_gpu_ms = (commandBuffer->GPUEndTime() - commandBuffer->GPUStartTime()) * 1000.0;
        }
    }

    // CPU post-processing (build final results) timing start
    auto q1_cpu_post_start = std::chrono::high_resolution_clock::now();

    // Read back final results
    long* sum_qty_c = (long*)f_sumQtyCents->contents();
    long* sum_base_c = (long*)f_sumBaseCents->contents();
    long* sum_disc_c = (long*)f_sumDiscPriceCents->contents();
    long* sum_charge_c = (long*)f_sumChargeCents->contents();
    uint32_t* sum_discount_bp = (uint32_t*)f_sumDiscountBP->contents();
    uint32_t* counts = (uint32_t*)f_counts->contents();

    struct Q1Result { double sum_qty, sum_base_price, sum_disc_price, sum_charge, avg_qty, avg_price, avg_disc; uint count; };
    std::map<std::pair<char,char>, Q1Result> final_results;
    auto emit_bin = [&](int rfIdx, int lsIdx, int bin){
        if (counts[bin] == 0) return;
        char rf = (rfIdx==0?'A':rfIdx==1?'N':'R');
        char ls = (lsIdx==0?'F':'O');
        Q1Result r;
        r.sum_qty = (double)sum_qty_c[bin] / 100.0;
        r.sum_base_price = (double)sum_base_c[bin] / 100.0;
        r.sum_disc_price = (double)sum_disc_c[bin] / 100.0;
        r.sum_charge = (double)sum_charge_c[bin] / 100.0;
        r.count = counts[bin];
        r.avg_qty = r.sum_qty / (double)r.count;
        r.avg_price = r.sum_base_price / (double)r.count;
        r.avg_disc = ((double)sum_discount_bp[bin] / 100.0) / (double)r.count; // average discount as fraction
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

    releaseAll(stage1PSO, stage2PSO, shipdateBuffer, flagBuffer, statusBuffer,
              qtyBuffer, priceBuffer, discBuffer, taxBuffer,
              p_sumQtyCents, p_sumBaseCents, p_sumDiscPriceCents, p_sumChargeCents, p_sumDiscountBP, p_counts,
              f_sumQtyCents, f_sumBaseCents, f_sumDiscPriceCents, f_sumChargeCents, f_sumDiscountBP, f_counts);
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
