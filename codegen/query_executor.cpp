#include "query_executor.h"
#include "tpch_schema.h"
#include "../src/infra.h"

#include <chrono>
#include <iostream>
#include <algorithm>
#include <map>
#include <cstring>

namespace codegen {

namespace {

// Helper: find pipeline by kernel name
MTL::ComputePipelineState* findPSO(const RuntimeCompiler::CompiledQuery& cq,
                                    const std::string& name) {
    for (size_t i = 0; i < cq.kernelNames.size(); i++)
        if (cq.kernelNames[i] == name) return cq.pipelines[i];
    return nullptr;
}

// ===================================================================
// Q1 EXECUTOR
// ===================================================================

void executeQ1(MTL::Device* device, MTL::CommandQueue* cmdQueue,
               const RuntimeCompiler::CompiledQuery& cq,
               const std::string& dataDir) {

    auto parseStart = std::chrono::high_resolution_clock::now();
    auto cols = loadColumnsMulti(dataDir + "lineitem.tbl", {
        {4, ColType::FLOAT}, {5, ColType::FLOAT}, {6, ColType::FLOAT}, {7, ColType::FLOAT},
        {8, ColType::CHAR1}, {9, ColType::CHAR1}, {10, ColType::DATE}
    });
    auto& l_quantity = cols.floats(4);
    auto& l_extendedprice = cols.floats(5);
    auto& l_discount = cols.floats(6);
    auto& l_tax = cols.floats(7);
    auto& l_returnflag = cols.chars(8);
    auto& l_linestatus = cols.chars(9);
    auto& l_shipdate = cols.ints(10);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    const uint data_size = (uint)l_shipdate.size();
    if (data_size == 0) { std::cerr << "Q1: no data" << std::endl; return; }

    auto* pso = findPSO(cq, "gen_q1_fused");
    if (!pso) return;

    // Create column buffers
    auto* shipdateBuf = device->newBuffer(l_shipdate.data(), data_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* flagBuf = device->newBuffer(l_returnflag.data(), data_size * sizeof(char), MTL::ResourceStorageModeShared);
    auto* statusBuf = device->newBuffer(l_linestatus.data(), data_size * sizeof(char), MTL::ResourceStorageModeShared);
    auto* qtyBuf = device->newBuffer(l_quantity.data(), data_size * sizeof(float), MTL::ResourceStorageModeShared);
    auto* priceBuf = device->newBuffer(l_extendedprice.data(), data_size * sizeof(float), MTL::ResourceStorageModeShared);
    auto* discBuf = device->newBuffer(l_discount.data(), data_size * sizeof(float), MTL::ResourceStorageModeShared);
    auto* taxBuf = device->newBuffer(l_tax.data(), data_size * sizeof(float), MTL::ResourceStorageModeShared);

    const uint bins = 6;
    const uint num_tg = 1024;

    auto* out_qty_lo = device->newBuffer(bins * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* out_qty_hi = device->newBuffer(bins * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* out_base_lo = device->newBuffer(bins * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* out_base_hi = device->newBuffer(bins * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* out_disc_lo = device->newBuffer(bins * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* out_disc_hi = device->newBuffer(bins * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* out_charge_lo = device->newBuffer(bins * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* out_charge_hi = device->newBuffer(bins * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* out_discount_bp = device->newBuffer(bins * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* out_count = device->newBuffer(bins * sizeof(uint32_t), MTL::ResourceStorageModeShared);

    const int cutoffDate = 19980902;

    double gpuMs = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(out_qty_lo->contents(), 0, bins * sizeof(uint32_t));
        memset(out_qty_hi->contents(), 0, bins * sizeof(uint32_t));
        memset(out_base_lo->contents(), 0, bins * sizeof(uint32_t));
        memset(out_base_hi->contents(), 0, bins * sizeof(uint32_t));
        memset(out_disc_lo->contents(), 0, bins * sizeof(uint32_t));
        memset(out_disc_hi->contents(), 0, bins * sizeof(uint32_t));
        memset(out_charge_lo->contents(), 0, bins * sizeof(uint32_t));
        memset(out_charge_hi->contents(), 0, bins * sizeof(uint32_t));
        memset(out_discount_bp->contents(), 0, bins * sizeof(uint32_t));
        memset(out_count->contents(), 0, bins * sizeof(uint32_t));

        auto* cmdBuf = cmdQueue->commandBuffer();
        auto* enc = cmdBuf->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(shipdateBuf, 0, 0);
        enc->setBuffer(flagBuf, 0, 1);
        enc->setBuffer(statusBuf, 0, 2);
        enc->setBuffer(qtyBuf, 0, 3);
        enc->setBuffer(priceBuf, 0, 4);
        enc->setBuffer(discBuf, 0, 5);
        enc->setBuffer(taxBuf, 0, 6);
        enc->setBuffer(out_qty_lo, 0, 7);
        enc->setBuffer(out_qty_hi, 0, 8);
        enc->setBuffer(out_base_lo, 0, 9);
        enc->setBuffer(out_base_hi, 0, 10);
        enc->setBuffer(out_disc_lo, 0, 11);
        enc->setBuffer(out_disc_hi, 0, 12);
        enc->setBuffer(out_charge_lo, 0, 13);
        enc->setBuffer(out_charge_hi, 0, 14);
        enc->setBuffer(out_discount_bp, 0, 15);
        enc->setBuffer(out_count, 0, 16);
        enc->setBytes(&data_size, sizeof(data_size), 17);
        enc->setBytes(&cutoffDate, sizeof(cutoffDate), 18);

        NS::UInteger tgSize = pso->maxTotalThreadsPerThreadgroup();
        if (tgSize > 1024) tgSize = 1024;
        enc->dispatchThreadgroups(MTL::Size::Make(num_tg, 1, 1), MTL::Size::Make(tgSize, 1, 1));
        enc->endEncoding();
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();

        if (iter == 2)
            gpuMs = (cmdBuf->GPUEndTime() - cmdBuf->GPUStartTime()) * 1000.0;
    }

    // Post-process
    auto postStart = std::chrono::high_resolution_clock::now();
    auto* qty_lo = (uint32_t*)out_qty_lo->contents();
    auto* qty_hi = (uint32_t*)out_qty_hi->contents();
    auto* base_lo = (uint32_t*)out_base_lo->contents();
    auto* base_hi = (uint32_t*)out_base_hi->contents();
    auto* disc_lo = (uint32_t*)out_disc_lo->contents();
    auto* disc_hi = (uint32_t*)out_disc_hi->contents();
    auto* charge_lo = (uint32_t*)out_charge_lo->contents();
    auto* charge_hi = (uint32_t*)out_charge_hi->contents();
    auto* sum_discount_bp = (uint32_t*)out_discount_bp->contents();
    auto* counts = (uint32_t*)out_count->contents();

    auto reconstruct = [](uint32_t lo, uint32_t hi) -> long {
        return (long)(((uint64_t)hi << 32) | (uint64_t)lo);
    };

    struct Q1R { double sum_qty, sum_base, sum_disc, sum_charge, avg_qty, avg_price, avg_disc; uint cnt; };
    char rfChars[] = {'A','A','N','N','R','R'};
    char lsChars[] = {'F','O','F','O','F','O'};

    printf("\n+----------+----------+------------+----------------+----------------+----------------+------------+------------+------------+----------+\n");
    printf("| l_return | l_linest |    sum_qty | sum_base_price | sum_disc_price |     sum_charge |    avg_qty |  avg_price |   avg_disc | count    |\n");
    printf("+----------+----------+------------+----------------+----------------+----------------+------------+------------+------------+----------+\n");
    for (int b = 0; b < 6; b++) {
        if (counts[b] == 0) continue;
        Q1R r;
        r.sum_qty = (double)reconstruct(qty_lo[b], qty_hi[b]) / 100.0;
        r.sum_base = (double)reconstruct(base_lo[b], base_hi[b]) / 100.0;
        r.sum_disc = (double)reconstruct(disc_lo[b], disc_hi[b]) / 100.0;
        r.sum_charge = (double)reconstruct(charge_lo[b], charge_hi[b]) / 100.0;
        r.cnt = counts[b];
        r.avg_qty = r.sum_qty / (double)r.cnt;
        r.avg_price = r.sum_base / (double)r.cnt;
        r.avg_disc = ((double)sum_discount_bp[b] / 100.0) / (double)r.cnt;
        printf("| %8c | %8c | %10.2f | %14.2f | %14.2f | %14.2f | %10.2f | %10.2f | %10.2f | %8u |\n",
               rfChars[b], lsChars[b], r.sum_qty, r.sum_base, r.sum_disc, r.sum_charge,
               r.avg_qty, r.avg_price, r.avg_disc, r.cnt);
    }
    printf("+----------+----------+------------+----------------+----------------+----------------+------------+------------+------------+----------+\n");

    auto postEnd = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nGen-Q1 | %u rows\n", data_size);
    printTimingSummary(parseMs, gpuMs, postMs);

    releaseAll(shipdateBuf, flagBuf, statusBuf, qtyBuf, priceBuf, discBuf, taxBuf,
               out_qty_lo, out_qty_hi, out_base_lo, out_base_hi,
               out_disc_lo, out_disc_hi, out_charge_lo, out_charge_hi,
               out_discount_bp, out_count);
}

// ===================================================================
// Q6 EXECUTOR
// ===================================================================

void executeQ6(MTL::Device* device, MTL::CommandQueue* cmdQueue,
               const RuntimeCompiler::CompiledQuery& cq,
               const std::string& dataDir) {

    auto parseStart = std::chrono::high_resolution_clock::now();
    auto cols = loadColumnsMulti(dataDir + "lineitem.tbl", {
        {4, ColType::FLOAT}, {5, ColType::FLOAT}, {6, ColType::FLOAT}, {10, ColType::DATE}
    });
    auto& l_quantity = cols.floats(4);
    auto& l_extendedprice = cols.floats(5);
    auto& l_discount = cols.floats(6);
    auto& l_shipdate = cols.ints(10);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    uint dataSize = (uint)l_shipdate.size();
    if (dataSize == 0) { std::cerr << "Q6: no data" << std::endl; return; }

    auto* s1pso = findPSO(cq, "gen_q6_stage1");
    auto* s2pso = findPSO(cq, "gen_q6_stage2");
    if (!s1pso || !s2pso) return;

    const uint numTG = 2048;
    int start_date = 19940101, end_date = 19950101;
    float min_discount = 0.05f, max_discount = 0.07f, max_quantity = 24.0f;

    auto* shipdateBuf = device->newBuffer(l_shipdate.data(), dataSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* discBuf = device->newBuffer(l_discount.data(), dataSize * sizeof(float), MTL::ResourceStorageModeShared);
    auto* qtyBuf = device->newBuffer(l_quantity.data(), dataSize * sizeof(float), MTL::ResourceStorageModeShared);
    auto* priceBuf = device->newBuffer(l_extendedprice.data(), dataSize * sizeof(float), MTL::ResourceStorageModeShared);
    auto* partialBuf = device->newBuffer(numTG * sizeof(float), MTL::ResourceStorageModeShared);
    auto* finalBuf = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    double gpuMs = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        auto* cmdBuf = cmdQueue->commandBuffer();
        auto* enc = cmdBuf->computeCommandEncoder();

        enc->setComputePipelineState(s1pso);
        enc->setBuffer(shipdateBuf, 0, 0);
        enc->setBuffer(discBuf, 0, 1);
        enc->setBuffer(qtyBuf, 0, 2);
        enc->setBuffer(priceBuf, 0, 3);
        enc->setBuffer(partialBuf, 0, 4);
        enc->setBytes(&dataSize, sizeof(dataSize), 5);
        enc->setBytes(&start_date, sizeof(start_date), 6);
        enc->setBytes(&end_date, sizeof(end_date), 7);
        enc->setBytes(&min_discount, sizeof(min_discount), 8);
        enc->setBytes(&max_discount, sizeof(max_discount), 9);
        enc->setBytes(&max_quantity, sizeof(max_quantity), 10);

        NS::UInteger tgSize = s1pso->maxTotalThreadsPerThreadgroup();
        if (tgSize > 1024) tgSize = 1024;
        enc->dispatchThreadgroups(MTL::Size::Make(numTG, 1, 1), MTL::Size::Make(tgSize, 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        enc->setComputePipelineState(s2pso);
        enc->setBuffer(partialBuf, 0, 0);
        enc->setBuffer(finalBuf, 0, 1);
        enc->setBytes(&numTG, sizeof(numTG), 2);
        enc->dispatchThreads(MTL::Size::Make(1, 1, 1), MTL::Size::Make(1, 1, 1));

        enc->endEncoding();
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();

        if (iter == 2)
            gpuMs = (cmdBuf->GPUEndTime() - cmdBuf->GPUStartTime()) * 1000.0;
    }

    auto postStart = std::chrono::high_resolution_clock::now();
    float revenue = ((float*)finalBuf->contents())[0];
    auto postEnd = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nTPC-H Gen-Q6 Result:\nTotal Revenue: $%.2f\n", revenue);
    printf("\nGen-Q6 | %u rows\n", dataSize);
    printTimingSummary(parseMs, gpuMs, postMs);

    releaseAll(shipdateBuf, discBuf, qtyBuf, priceBuf, partialBuf, finalBuf);
}

// ===================================================================
// Q3 EXECUTOR
// ===================================================================

void executeQ3(MTL::Device* device, MTL::CommandQueue* cmdQueue,
               const RuntimeCompiler::CompiledQuery& cq,
               const std::string& dataDir) {

    auto parseStart = std::chrono::high_resolution_clock::now();
    auto cCols = loadColumnsMulti(dataDir + "customer.tbl", {{0, ColType::INT}, {6, ColType::CHAR1}});
    auto& c_custkey = cCols.ints(0);
    auto& c_mktsegment = cCols.chars(6);

    auto oCols = loadColumnsMulti(dataDir + "orders.tbl",
        {{0, ColType::INT}, {1, ColType::INT}, {4, ColType::DATE}, {7, ColType::INT}});
    auto& o_orderkey = oCols.ints(0);
    auto& o_custkey = oCols.ints(1);
    auto& o_orderdate = oCols.ints(4);
    auto& o_shippriority = oCols.ints(7);

    auto lCols = loadColumnsMulti(dataDir + "lineitem.tbl",
        {{0, ColType::INT}, {5, ColType::FLOAT}, {6, ColType::FLOAT}, {10, ColType::DATE}});
    auto& l_orderkey = lCols.ints(0);
    auto& l_shipdate = lCols.ints(10);
    auto& l_extendedprice = lCols.floats(5);
    auto& l_discount = lCols.floats(6);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    const uint customer_size = (uint)c_custkey.size();
    const uint orders_size = (uint)o_orderkey.size();
    const uint lineitem_size = (uint)l_orderkey.size();

    auto* bmPSO = findPSO(cq, "gen_q3_build_customer_bitmap");
    auto* mapPSO = findPSO(cq, "gen_q3_build_orders_map");
    auto* probePSO = findPSO(cq, "gen_q3_probe_agg");
    auto* compactPSO = findPSO(cq, "gen_q3_compact");
    if (!bmPSO || !mapPSO || !probePSO || !compactPSO) return;

    // Customer bitmap
    int max_custkey = 0;
    for (int k : c_custkey) max_custkey = std::max(max_custkey, k);
    auto* custBmBuf = createBitmapBuffer(device, max_custkey);

    auto* custKeyBuf = device->newBuffer(c_custkey.data(), customer_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* custMktBuf = device->newBuffer(c_mktsegment.data(), customer_size * sizeof(char), MTL::ResourceStorageModeShared);

    // Orders direct map
    int max_orderkey = 0;
    for (int k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    const uint orders_map_size = max_orderkey + 1;
    auto* ordersMapBuf = device->newBuffer(orders_map_size * sizeof(int), MTL::ResourceStorageModeShared);
    memset(ordersMapBuf->contents(), 0xFF, orders_map_size * sizeof(int)); // -1

    auto* ordKeyBuf = device->newBuffer(o_orderkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* ordCustBuf = device->newBuffer(o_custkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* ordDateBuf = device->newBuffer(o_orderdate.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* ordPrioBuf = device->newBuffer(o_shippriority.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);

    auto* lineOrdKeyBuf = device->newBuffer(l_orderkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* lineShipBuf = device->newBuffer(l_shipdate.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* linePriceBuf = device->newBuffer(l_extendedprice.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);
    auto* lineDiscBuf = device->newBuffer(l_discount.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);

    const uint final_ht_size = nextPow2(orders_size * 2);
    // 16 bytes per entry (int key, float revenue, uint orderdate, uint shippriority)
    auto* finalHTBuf = createFilledBuffer(device, final_ht_size * 16, 0);
    auto* denseBuf = device->newBuffer(final_ht_size * 16, MTL::ResourceStorageModeShared);
    auto* countBuf = createFilledBuffer(device, sizeof(uint), 0);

    const int cutoff_date = 19950315;
    const uint num_tg = 2048;

    double gpuMs = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(custBmBuf->contents(), 0, custBmBuf->length());
        memset(ordersMapBuf->contents(), 0xFF, orders_map_size * sizeof(int));
        memset(finalHTBuf->contents(), 0, final_ht_size * 16);
        memset(countBuf->contents(), 0, sizeof(uint));

        auto* cmdBuf = cmdQueue->commandBuffer();
        auto* enc = cmdBuf->computeCommandEncoder();

        // Kernel 1: Build customer bitmap
        enc->setComputePipelineState(bmPSO);
        enc->setBuffer(custKeyBuf, 0, 0);
        enc->setBuffer(custMktBuf, 0, 1);
        enc->setBuffer(custBmBuf, 0, 2);
        enc->setBytes(&customer_size, sizeof(customer_size), 3);
        enc->dispatchThreads(MTL::Size::Make(customer_size, 1, 1),
                            MTL::Size::Make(std::min((NS::UInteger)customer_size, (NS::UInteger)1024), 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // Kernel 2: Build orders map
        enc->setComputePipelineState(mapPSO);
        enc->setBuffer(ordKeyBuf, 0, 0);
        enc->setBuffer(ordDateBuf, 0, 1);
        enc->setBuffer(ordersMapBuf, 0, 2);
        enc->setBytes(&orders_size, sizeof(orders_size), 3);
        enc->setBytes(&cutoff_date, sizeof(cutoff_date), 4);
        enc->setBuffer(ordCustBuf, 0, 5);
        enc->setBuffer(custBmBuf, 0, 6);
        enc->dispatchThreads(MTL::Size::Make(orders_size, 1, 1),
                            MTL::Size::Make(std::min((NS::UInteger)orders_size, (NS::UInteger)1024), 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // Kernel 3: Probe + aggregate
        enc->setComputePipelineState(probePSO);
        enc->setBuffer(lineOrdKeyBuf, 0, 0);
        enc->setBuffer(lineShipBuf, 0, 1);
        enc->setBuffer(linePriceBuf, 0, 2);
        enc->setBuffer(lineDiscBuf, 0, 3);
        enc->setBuffer(ordersMapBuf, 0, 4);
        enc->setBuffer(ordCustBuf, 0, 5);
        enc->setBuffer(ordDateBuf, 0, 6);
        enc->setBuffer(ordPrioBuf, 0, 7);
        enc->setBuffer(finalHTBuf, 0, 8);
        enc->setBytes(&lineitem_size, sizeof(lineitem_size), 9);
        enc->setBytes(&cutoff_date, sizeof(cutoff_date), 10);
        enc->setBytes(&final_ht_size, sizeof(final_ht_size), 11);
        NS::UInteger tgSize = probePSO->maxTotalThreadsPerThreadgroup();
        if (tgSize > 1024) tgSize = 1024;
        enc->dispatchThreadgroups(MTL::Size::Make(num_tg, 1, 1), MTL::Size::Make(tgSize, 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // Kernel 4: Compact
        enc->setComputePipelineState(compactPSO);
        enc->setBuffer(finalHTBuf, 0, 0);
        enc->setBuffer(denseBuf, 0, 1);
        enc->setBuffer(countBuf, 0, 2);
        enc->setBytes(&final_ht_size, sizeof(final_ht_size), 3);
        enc->dispatchThreads(MTL::Size::Make(final_ht_size, 1, 1),
                            MTL::Size::Make(std::min((NS::UInteger)final_ht_size, (NS::UInteger)1024), 1, 1));

        enc->endEncoding();
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();

        if (iter == 2)
            gpuMs = (cmdBuf->GPUEndTime() - cmdBuf->GPUStartTime()) * 1000.0;
    }

    // Post-process: sort and print
    auto postStart = std::chrono::high_resolution_clock::now();
    uint resultCount = ((uint*)countBuf->contents())[0];
    auto* dense = (Q3Aggregates_CPU*)denseBuf->contents();
    double postMs = sortAndPrintQ3(dense, resultCount);
    auto postEnd = std::chrono::high_resolution_clock::now();
    postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nGen-Q3 | %u customers, %u orders, %u lineitem\n", customer_size, orders_size, lineitem_size);
    printTimingSummary(parseMs, gpuMs, postMs);

    releaseAll(custBmBuf, custKeyBuf, custMktBuf,
               ordersMapBuf, ordKeyBuf, ordCustBuf, ordDateBuf, ordPrioBuf,
               lineOrdKeyBuf, lineShipBuf, linePriceBuf, lineDiscBuf,
               finalHTBuf, denseBuf, countBuf);
}

// ===================================================================
// Q14 EXECUTOR
// ===================================================================

void executeQ14(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                const RuntimeCompiler::CompiledQuery& cq,
                const std::string& dataDir) {

    auto parseStart = std::chrono::high_resolution_clock::now();
    // Part table — build promo bitmap on CPU
    auto pCols = loadColumnsMulti(dataDir + "part.tbl",
        {{0, ColType::INT}, {4, ColType::CHAR_FIXED, 25}});
    auto& p_partkey = pCols.ints(0);
    auto& p_type = pCols.chars(4);

    // Lineitem columns
    auto lCols = loadColumnsMulti(dataDir + "lineitem.tbl",
        {{1, ColType::INT}, {5, ColType::FLOAT}, {6, ColType::FLOAT}, {10, ColType::DATE}});
    auto& l_partkey = lCols.ints(1);
    auto& l_extendedprice = lCols.floats(5);
    auto& l_discount = lCols.floats(6);
    auto& l_shipdate = lCols.ints(10);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    const uint lineitem_size = (uint)l_partkey.size();
    if (lineitem_size == 0) { std::cerr << "Q14: no data" << std::endl; return; }

    // Build promo bitmap on CPU
    auto promoBm = buildCPUBitmap(p_partkey, [&](size_t i) {
        const char* t = p_type.data() + i * 25;
        return t[0]=='P' && t[1]=='R' && t[2]=='O' && t[3]=='M' && t[4]=='O';
    });
    auto* promoBmBuf = uploadBitmap(device, promoBm);

    auto* s1pso = findPSO(cq, "gen_q14_stage1");
    auto* s2pso = findPSO(cq, "gen_q14_stage2");
    if (!s1pso || !s2pso) return;

    const uint numTG = 2048;
    int start_date = 19950901, end_date = 19951001;

    auto* partKeyBuf = device->newBuffer(l_partkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* shipdateBuf = device->newBuffer(l_shipdate.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* priceBuf = device->newBuffer(l_extendedprice.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);
    auto* discBuf = device->newBuffer(l_discount.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);
    auto* partialPromoBuf = device->newBuffer(numTG * sizeof(float), MTL::ResourceStorageModeShared);
    auto* partialTotalBuf = device->newBuffer(numTG * sizeof(float), MTL::ResourceStorageModeShared);
    auto* finalBuf = device->newBuffer(2 * sizeof(float), MTL::ResourceStorageModeShared);

    double gpuMs = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        auto* cmdBuf = cmdQueue->commandBuffer();
        auto* enc = cmdBuf->computeCommandEncoder();

        enc->setComputePipelineState(s1pso);
        enc->setBuffer(partKeyBuf, 0, 0);
        enc->setBuffer(shipdateBuf, 0, 1);
        enc->setBuffer(priceBuf, 0, 2);
        enc->setBuffer(discBuf, 0, 3);
        enc->setBuffer(promoBmBuf, 0, 4);
        enc->setBuffer(partialPromoBuf, 0, 5);
        enc->setBuffer(partialTotalBuf, 0, 6);
        enc->setBytes(&lineitem_size, sizeof(lineitem_size), 7);
        enc->setBytes(&start_date, sizeof(start_date), 8);
        enc->setBytes(&end_date, sizeof(end_date), 9);
        NS::UInteger tgSize = s1pso->maxTotalThreadsPerThreadgroup();
        if (tgSize > 1024) tgSize = 1024;
        enc->dispatchThreadgroups(MTL::Size::Make(numTG, 1, 1), MTL::Size::Make(tgSize, 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        enc->setComputePipelineState(s2pso);
        enc->setBuffer(partialPromoBuf, 0, 0);
        enc->setBuffer(partialTotalBuf, 0, 1);
        enc->setBuffer(finalBuf, 0, 2);
        enc->setBytes(&numTG, sizeof(numTG), 3);
        enc->dispatchThreads(MTL::Size::Make(1, 1, 1), MTL::Size::Make(1, 1, 1));

        enc->endEncoding();
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();

        if (iter == 2)
            gpuMs = (cmdBuf->GPUEndTime() - cmdBuf->GPUStartTime()) * 1000.0;
    }

    auto postStart = std::chrono::high_resolution_clock::now();
    float* res = (float*)finalBuf->contents();
    float promoRevenue = 100.0f * res[0] / res[1];
    auto postEnd = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nTPC-H Gen-Q14 Result:\nPromo Revenue: %.2f%%\n", promoRevenue);
    printf("\nGen-Q14 | %u rows\n", lineitem_size);
    printTimingSummary(parseMs, gpuMs, postMs);

    releaseAll(promoBmBuf, partKeyBuf, shipdateBuf, priceBuf, discBuf,
               partialPromoBuf, partialTotalBuf, finalBuf);
}

// ===================================================================
// Q13 EXECUTOR
// ===================================================================

void executeQ13(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                const RuntimeCompiler::CompiledQuery& cq,
                const std::string& dataDir) {

    auto parseStart = std::chrono::high_resolution_clock::now();
    auto cCols = loadColumnsMulti(dataDir + "customer.tbl", {{0, ColType::INT}});
    auto& c_custkey = cCols.ints(0);
    int max_custkey = 0;
    for (int k : c_custkey) max_custkey = std::max(max_custkey, k);

    auto oCols = loadColumnsMulti(dataDir + "orders.tbl",
        {{1, ColType::INT}, {8, ColType::CHAR_FIXED, 79}});
    auto& o_custkey = oCols.ints(1);
    auto& o_comment = oCols.chars(8);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    const uint orders_size = (uint)o_custkey.size();
    const uint comment_width = 79;

    auto* countPSO = findPSO(cq, "gen_q13_count_orders");
    auto* histPSO = findPSO(cq, "gen_q13_histogram");
    if (!countPSO || !histPSO) return;

    auto* oCustBuf = device->newBuffer(o_custkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* oCommentBuf = device->newBuffer(o_comment.data(), orders_size * comment_width, MTL::ResourceStorageModeShared);
    auto* countsBuf = createFilledBuffer(device, (max_custkey + 1) * sizeof(uint), 0);

    uint maxOrders = 100; // histogram buckets
    auto* histBuf = createFilledBuffer(device, (maxOrders + 1) * sizeof(uint), 0);

    double gpuMs = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(countsBuf->contents(), 0, (max_custkey + 1) * sizeof(uint));
        memset(histBuf->contents(), 0, (maxOrders + 1) * sizeof(uint));

        auto* cmdBuf = cmdQueue->commandBuffer();
        auto* enc = cmdBuf->computeCommandEncoder();

        // Kernel 1: count orders per customer (excluding pattern match)
        enc->setComputePipelineState(countPSO);
        enc->setBuffer(oCustBuf, 0, 0);
        enc->setBuffer(oCommentBuf, 0, 1);
        enc->setBuffer(countsBuf, 0, 2);
        enc->setBytes(&orders_size, sizeof(orders_size), 3);
        enc->setBytes(&comment_width, sizeof(comment_width), 4);
        enc->dispatchThreads(MTL::Size::Make(orders_size, 1, 1),
                            MTL::Size::Make(std::min((NS::UInteger)orders_size, (NS::UInteger)1024), 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // Kernel 2: histogram
        uint maxCK = (uint)max_custkey;
        enc->setComputePipelineState(histPSO);
        enc->setBuffer(countsBuf, 0, 0);
        enc->setBuffer(histBuf, 0, 1);
        enc->setBytes(&maxCK, sizeof(maxCK), 2);
        uint histThreads = max_custkey + 1;
        enc->dispatchThreads(MTL::Size::Make(histThreads, 1, 1),
                            MTL::Size::Make(std::min((NS::UInteger)histThreads, (NS::UInteger)1024), 1, 1));

        enc->endEncoding();
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();

        if (iter == 2)
            gpuMs = (cmdBuf->GPUEndTime() - cmdBuf->GPUStartTime()) * 1000.0;
    }

    // Post-process
    auto postStart = std::chrono::high_resolution_clock::now();
    auto* hist = (uint*)histBuf->contents();

    // Add customers with 0 orders
    uint customersWithOrders = 0;
    for (uint i = 1; i <= maxOrders; i++) customersWithOrders += hist[i];
    hist[0] = (uint)c_custkey.size() - customersWithOrders;

    struct Row { uint c_count; uint custdist; };
    std::vector<Row> rows;
    for (uint i = 0; i <= maxOrders; i++)
        if (hist[i] > 0) rows.push_back({i, hist[i]});

    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        if (a.custdist != b.custdist) return a.custdist > b.custdist;
        return a.c_count > b.c_count;
    });

    printf("\nTPC-H Gen-Q13 Results:\n");
    printf("+----------+----------+\n");
    printf("| c_count  | custdist |\n");
    printf("+----------+----------+\n");
    for (auto& r : rows)
        printf("| %8u | %8u |\n", r.c_count, r.custdist);
    printf("+----------+----------+\n");

    auto postEnd = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nGen-Q13 | %u orders, %u customers\n", orders_size, (uint)c_custkey.size());
    printTimingSummary(parseMs, gpuMs, postMs);

    releaseAll(oCustBuf, oCommentBuf, countsBuf, histBuf);
}

// ===================================================================
// Q4 EXECUTOR
// ===================================================================

void executeQ4(MTL::Device* device, MTL::CommandQueue* cmdQueue,
               const RuntimeCompiler::CompiledQuery& cq,
               const std::string& dataDir) {

    auto parseStart = std::chrono::high_resolution_clock::now();
    auto lCols = loadColumnsMulti(dataDir + "lineitem.tbl",
        {{0, ColType::INT}, {11, ColType::DATE}, {12, ColType::DATE}});
    auto& l_orderkey = lCols.ints(0);
    auto& l_commitdate = lCols.ints(11);
    auto& l_receiptdate = lCols.ints(12);

    auto oCols = loadColumnsMulti(dataDir + "orders.tbl",
        {{0, ColType::INT}, {4, ColType::DATE}, {5, ColType::CHAR1}});
    auto& o_orderkey = oCols.ints(0);
    auto& o_orderdate = oCols.ints(4);
    auto& o_orderpriority = oCols.chars(5);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    uint liSize = (uint)l_orderkey.size();
    uint ordSize = (uint)o_orderkey.size();
    if (liSize == 0 || ordSize == 0) { std::cerr << "Q4: no data" << std::endl; return; }

    auto* bmPSO = findPSO(cq, "gen_q4_build_late_bitmap");
    auto* s1PSO = findPSO(cq, "gen_q4_count_stage1");
    auto* s2PSO = findPSO(cq, "gen_q4_count_stage2");
    if (!bmPSO || !s1PSO || !s2PSO) return;

    int max_orderkey = 0;
    for (int k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    for (int k : l_orderkey) max_orderkey = std::max(max_orderkey, k);
    auto* lateBitmapBuf = createBitmapBuffer(device, max_orderkey);

    const uint numTG = 2048;
    int date_start = 19930701, date_end = 19931001;

    auto* liOrderkeyBuf = device->newBuffer(l_orderkey.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* liCommitBuf   = device->newBuffer(l_commitdate.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* liReceiptBuf  = device->newBuffer(l_receiptdate.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* ordKeyBuf     = device->newBuffer(o_orderkey.data(), ordSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* ordDateBuf    = device->newBuffer(o_orderdate.data(), ordSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* ordPrioBuf    = device->newBuffer(o_orderpriority.data(), ordSize * sizeof(char), MTL::ResourceStorageModeShared);
    auto* partialBuf    = device->newBuffer(numTG * 5 * sizeof(uint), MTL::ResourceStorageModeShared);
    auto* finalBuf      = device->newBuffer(5 * sizeof(uint), MTL::ResourceStorageModeShared);

    double gpuMs = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(lateBitmapBuf->contents(), 0, lateBitmapBuf->length());

        auto* cmdBuf = cmdQueue->commandBuffer();
        auto* enc = cmdBuf->computeCommandEncoder();

        enc->setComputePipelineState(bmPSO);
        enc->setBuffer(liOrderkeyBuf, 0, 0);
        enc->setBuffer(liCommitBuf, 0, 1);
        enc->setBuffer(liReceiptBuf, 0, 2);
        enc->setBuffer(lateBitmapBuf, 0, 3);
        enc->setBytes(&liSize, sizeof(liSize), 4);
        {
            NS::UInteger tgSize = bmPSO->maxTotalThreadsPerThreadgroup();
            if (tgSize > 1024) tgSize = 1024;
            uint numGroups = (liSize + (uint)tgSize - 1) / (uint)tgSize;
            enc->dispatchThreadgroups(MTL::Size::Make(numGroups, 1, 1), MTL::Size::Make(tgSize, 1, 1));
        }

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        enc->setComputePipelineState(s1PSO);
        enc->setBuffer(ordKeyBuf, 0, 0);
        enc->setBuffer(ordDateBuf, 0, 1);
        enc->setBuffer(ordPrioBuf, 0, 2);
        enc->setBuffer(lateBitmapBuf, 0, 3);
        enc->setBuffer(partialBuf, 0, 4);
        enc->setBytes(&ordSize, sizeof(ordSize), 5);
        enc->setBytes(&date_start, sizeof(date_start), 6);
        enc->setBytes(&date_end, sizeof(date_end), 7);
        {
            NS::UInteger tgSize = s1PSO->maxTotalThreadsPerThreadgroup();
            if (tgSize > 1024) tgSize = 1024;
            enc->dispatchThreadgroups(MTL::Size::Make(numTG, 1, 1), MTL::Size::Make(tgSize, 1, 1));
        }

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        const uint numTGs = numTG;
        enc->setComputePipelineState(s2PSO);
        enc->setBuffer(partialBuf, 0, 0);
        enc->setBuffer(finalBuf, 0, 1);
        enc->setBytes(&numTGs, sizeof(numTGs), 2);
        enc->dispatchThreads(MTL::Size::Make(1, 1, 1), MTL::Size::Make(1, 1, 1));

        enc->endEncoding();
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();

        if (iter == 2)
            gpuMs = (cmdBuf->GPUEndTime() - cmdBuf->GPUStartTime()) * 1000.0;
    }

    auto postStart = std::chrono::high_resolution_clock::now();
    uint* res = (uint*)finalBuf->contents();

    const char* prio_names[] = {"1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"};
    printf("\nTPC-H Gen-Q4 Results:\n");
    printf("+------------------+-------------+\n");
    printf("| o_orderpriority  | order_count |\n");
    printf("+------------------+-------------+\n");
    for (int i = 0; i < 5; i++)
        printf("| %-16s | %11u |\n", prio_names[i], res[i]);
    printf("+------------------+-------------+\n");

    auto postEnd = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nGen-Q4 | %u lineitem, %u orders\n", liSize, ordSize);
    printTimingSummary(parseMs, gpuMs, postMs);

    releaseAll(lateBitmapBuf, liOrderkeyBuf, liCommitBuf, liReceiptBuf,
               ordKeyBuf, ordDateBuf, ordPrioBuf, partialBuf, finalBuf);
}

// ===================================================================
// Q12 EXECUTOR
// ===================================================================

void executeQ12(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                const RuntimeCompiler::CompiledQuery& cq,
                const std::string& dataDir) {

    auto parseStart = std::chrono::high_resolution_clock::now();
    auto oCols = loadColumnsMulti(dataDir + "orders.tbl",
        {{0, ColType::INT}, {5, ColType::CHAR1}});
    auto& o_orderkey = oCols.ints(0);
    auto& o_orderpriority = oCols.chars(5);

    auto lCols = loadColumnsMulti(dataDir + "lineitem.tbl",
        {{0, ColType::INT}, {10, ColType::DATE}, {11, ColType::DATE}, {12, ColType::DATE}, {14, ColType::CHAR1}});
    auto& l_orderkey = lCols.ints(0);
    auto& l_shipmode = lCols.chars(14);
    auto& l_shipdate = lCols.ints(10);
    auto& l_commitdate = lCols.ints(11);
    auto& l_receiptdate = lCols.ints(12);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    uint liSize = (uint)l_orderkey.size();
    uint ordSize = (uint)o_orderkey.size();
    if (liSize == 0) { std::cerr << "Q12: no data" << std::endl; return; }

    auto* bmPSO = findPSO(cq, "gen_q12_build_priority_bitmap");
    auto* s1PSO = findPSO(cq, "gen_q12_filter_stage1");
    auto* s2PSO = findPSO(cq, "gen_q12_filter_stage2");
    if (!bmPSO || !s1PSO || !s2PSO) return;

    int max_orderkey = 0;
    for (int k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    auto* bitmapBuf = createBitmapBuffer(device, max_orderkey);

    const uint numTG = 2048;
    int receipt_start = 19940101, receipt_end = 19950101;

    auto* ordKeyBuf     = device->newBuffer(o_orderkey.data(), ordSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* ordPrioBuf    = device->newBuffer(o_orderpriority.data(), ordSize * sizeof(char), MTL::ResourceStorageModeShared);
    auto* orderkeyBuf   = device->newBuffer(l_orderkey.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* shipmodeBuf   = device->newBuffer(l_shipmode.data(), liSize * sizeof(char), MTL::ResourceStorageModeShared);
    auto* shipdateBuf   = device->newBuffer(l_shipdate.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* commitdateBuf = device->newBuffer(l_commitdate.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* receiptBuf    = device->newBuffer(l_receiptdate.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* partialBuf    = device->newBuffer(numTG * 4 * sizeof(uint), MTL::ResourceStorageModeShared);
    auto* finalBuf      = device->newBuffer(4 * sizeof(uint), MTL::ResourceStorageModeShared);

    double gpuMs = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(bitmapBuf->contents(), 0, bitmapBuf->length());

        auto* cmdBuf = cmdQueue->commandBuffer();
        auto* enc = cmdBuf->computeCommandEncoder();

        enc->setComputePipelineState(bmPSO);
        enc->setBuffer(ordKeyBuf, 0, 0);
        enc->setBuffer(ordPrioBuf, 0, 1);
        enc->setBuffer(bitmapBuf, 0, 2);
        enc->setBytes(&ordSize, sizeof(ordSize), 3);
        {
            NS::UInteger tgSize = bmPSO->maxTotalThreadsPerThreadgroup();
            if (tgSize > 1024) tgSize = 1024;
            uint numGroups = (ordSize + (uint)tgSize - 1) / (uint)tgSize;
            enc->dispatchThreadgroups(MTL::Size::Make(numGroups, 1, 1), MTL::Size::Make(tgSize, 1, 1));
        }

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        enc->setComputePipelineState(s1PSO);
        enc->setBuffer(orderkeyBuf, 0, 0);
        enc->setBuffer(shipmodeBuf, 0, 1);
        enc->setBuffer(shipdateBuf, 0, 2);
        enc->setBuffer(commitdateBuf, 0, 3);
        enc->setBuffer(receiptBuf, 0, 4);
        enc->setBuffer(bitmapBuf, 0, 5);
        enc->setBuffer(partialBuf, 0, 6);
        enc->setBytes(&liSize, sizeof(liSize), 7);
        enc->setBytes(&receipt_start, sizeof(receipt_start), 8);
        enc->setBytes(&receipt_end, sizeof(receipt_end), 9);
        {
            NS::UInteger tgSize = s1PSO->maxTotalThreadsPerThreadgroup();
            if (tgSize > 1024) tgSize = 1024;
            enc->dispatchThreadgroups(MTL::Size::Make(numTG, 1, 1), MTL::Size::Make(tgSize, 1, 1));
        }

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        const uint numTGs = numTG;
        enc->setComputePipelineState(s2PSO);
        enc->setBuffer(partialBuf, 0, 0);
        enc->setBuffer(finalBuf, 0, 1);
        enc->setBytes(&numTGs, sizeof(numTGs), 2);
        enc->dispatchThreads(MTL::Size::Make(1, 1, 1), MTL::Size::Make(1, 1, 1));

        enc->endEncoding();
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();

        if (iter == 2)
            gpuMs = (cmdBuf->GPUEndTime() - cmdBuf->GPUStartTime()) * 1000.0;
    }

    auto postStart = std::chrono::high_resolution_clock::now();
    uint* res = (uint*)finalBuf->contents();

    printf("\nTPC-H Gen-Q12 Results:\n");
    printf("+----------+------------------+-----------------+\n");
    printf("| shipmode | high_line_count  | low_line_count  |\n");
    printf("+----------+------------------+-----------------+\n");
    printf("| MAIL     | %16u | %15u |\n", res[0], res[1]);
    printf("| SHIP     | %16u | %15u |\n", res[2], res[3]);
    printf("+----------+------------------+-----------------+\n");

    auto postEnd = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nGen-Q12 | %u lineitem, %u orders\n", liSize, ordSize);
    printTimingSummary(parseMs, gpuMs, postMs);

    releaseAll(bitmapBuf, ordKeyBuf, ordPrioBuf, orderkeyBuf, shipmodeBuf,
               shipdateBuf, commitdateBuf, receiptBuf, partialBuf, finalBuf);
}

// ===================================================================
// Q19 EXECUTOR
// ===================================================================

void executeQ19(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                const RuntimeCompiler::CompiledQuery& cq,
                const std::string& dataDir) {

    auto parseStart = std::chrono::high_resolution_clock::now();
    auto pCols = loadColumnsMulti(dataDir + "part.tbl",
        {{0, ColType::INT}, {3, ColType::CHAR_FIXED, 10}, {5, ColType::INT}, {6, ColType::CHAR_FIXED, 10}});
    auto& p_partkey = pCols.ints(0);
    auto& p_brand = pCols.chars(3);
    auto& p_size = pCols.ints(5);
    auto& p_container = pCols.chars(6);

    auto lCols = loadColumnsMulti(dataDir + "lineitem.tbl",
        {{1, ColType::INT}, {4, ColType::FLOAT}, {5, ColType::FLOAT}, {6, ColType::FLOAT},
         {13, ColType::CHAR_FIXED, 25}, {14, ColType::CHAR_FIXED, 10}});
    auto& l_partkey = lCols.ints(1);
    auto& l_quantity = lCols.floats(4);
    auto& l_extendedprice = lCols.floats(5);
    auto& l_discount = lCols.floats(6);
    auto& l_shipinstruct = lCols.chars(13);
    auto& l_shipmode = lCols.chars(14);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    uint liSize = (uint)l_partkey.size();
    uint partSize = (uint)p_partkey.size();
    if (liSize == 0) { std::cerr << "Q19: no data" << std::endl; return; }

    int max_partkey = 0;
    for (int k : p_partkey) max_partkey = std::max(max_partkey, k);
    uint mapSize = (uint)(max_partkey + 1);
    const uint brand_stride = 10;
    const uint container_stride = 10;
    const uint shipmode_stride = 10;
    const uint shipinstruct_stride = 25;

    auto* mapPSO    = findPSO(cq, "gen_q19_build_part_group_map");
    auto* filterPSO = findPSO(cq, "gen_q19_shipmode_filter");
    auto* s1PSO     = findPSO(cq, "gen_q19_sum_stage1");
    auto* s2PSO     = findPSO(cq, "gen_q19_sum_stage2");
    if (!mapPSO || !filterPSO || !s1PSO || !s2PSO) return;

    const uint numTG = 2048;

    auto* pPartKeyBuf   = device->newBuffer(p_partkey.data(), partSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pBrandBuf     = device->newBuffer(p_brand.data(), (size_t)partSize * brand_stride, MTL::ResourceStorageModeShared);
    auto* pContainerBuf = device->newBuffer(p_container.data(), (size_t)partSize * container_stride, MTL::ResourceStorageModeShared);
    auto* pSizeBuf      = device->newBuffer(p_size.data(), partSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* mapBuf        = device->newBuffer(mapSize * sizeof(uint8_t), MTL::ResourceStorageModeShared);

    auto* smBuf        = device->newBuffer(l_shipmode.data(), (size_t)liSize * shipmode_stride, MTL::ResourceStorageModeShared);
    auto* siBuf        = device->newBuffer(l_shipinstruct.data(), (size_t)liSize * shipinstruct_stride, MTL::ResourceStorageModeShared);
    auto* qualifiesBuf = device->newBuffer(liSize * sizeof(uint8_t), MTL::ResourceStorageModeShared);

    auto* partkeyBuf = device->newBuffer(l_partkey.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* qtyBuf     = device->newBuffer(l_quantity.data(), liSize * sizeof(float), MTL::ResourceStorageModeShared);
    auto* priceBuf   = device->newBuffer(l_extendedprice.data(), liSize * sizeof(float), MTL::ResourceStorageModeShared);
    auto* discBuf    = device->newBuffer(l_discount.data(), liSize * sizeof(float), MTL::ResourceStorageModeShared);
    auto* partialBuf = device->newBuffer(numTG * sizeof(float), MTL::ResourceStorageModeShared);
    auto* finalBuf   = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    double gpuMs = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(mapBuf->contents(), 0xFF, mapSize * sizeof(uint8_t));

        auto* cmdBuf = cmdQueue->commandBuffer();
        auto* enc = cmdBuf->computeCommandEncoder();

        // Phase 1: Build part group map
        enc->setComputePipelineState(mapPSO);
        enc->setBuffer(pPartKeyBuf, 0, 0);
        enc->setBuffer(pBrandBuf, 0, 1);
        enc->setBuffer(pContainerBuf, 0, 2);
        enc->setBuffer(pSizeBuf, 0, 3);
        enc->setBuffer(mapBuf, 0, 4);
        enc->setBytes(&partSize, sizeof(partSize), 5);
        enc->setBytes(&brand_stride, sizeof(brand_stride), 6);
        enc->setBytes(&container_stride, sizeof(container_stride), 7);
        {
            NS::UInteger tgSize = mapPSO->maxTotalThreadsPerThreadgroup();
            if (tgSize > 1024) tgSize = 1024;
            uint numGroups = (partSize + (uint)tgSize - 1) / (uint)tgSize;
            enc->dispatchThreadgroups(MTL::Size::Make(numGroups, 1, 1), MTL::Size::Make(tgSize, 1, 1));
        }

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // Phase 2: Compute shipmode qualifies flag
        enc->setComputePipelineState(filterPSO);
        enc->setBuffer(smBuf, 0, 0);
        enc->setBuffer(siBuf, 0, 1);
        enc->setBuffer(qualifiesBuf, 0, 2);
        enc->setBytes(&liSize, sizeof(liSize), 3);
        enc->setBytes(&shipmode_stride, sizeof(shipmode_stride), 4);
        enc->setBytes(&shipinstruct_stride, sizeof(shipinstruct_stride), 5);
        {
            NS::UInteger tgSize = filterPSO->maxTotalThreadsPerThreadgroup();
            if (tgSize > 1024) tgSize = 1024;
            uint numGroups = (liSize + (uint)tgSize - 1) / (uint)tgSize;
            enc->dispatchThreadgroups(MTL::Size::Make(numGroups, 1, 1), MTL::Size::Make(tgSize, 1, 1));
        }

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // Phase 3: Main filter+sum
        enc->setComputePipelineState(s1PSO);
        enc->setBuffer(partkeyBuf, 0, 0);
        enc->setBuffer(qtyBuf, 0, 1);
        enc->setBuffer(priceBuf, 0, 2);
        enc->setBuffer(discBuf, 0, 3);
        enc->setBuffer(qualifiesBuf, 0, 4);
        enc->setBuffer(mapBuf, 0, 5);
        enc->setBuffer(partialBuf, 0, 6);
        enc->setBytes(&liSize, sizeof(liSize), 7);
        enc->setBytes(&mapSize, sizeof(mapSize), 8);
        {
            NS::UInteger tgSize = s1PSO->maxTotalThreadsPerThreadgroup();
            if (tgSize > 1024) tgSize = 1024;
            enc->dispatchThreadgroups(MTL::Size::Make(numTG, 1, 1), MTL::Size::Make(tgSize, 1, 1));
        }

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        const uint numTGs = numTG;
        enc->setComputePipelineState(s2PSO);
        enc->setBuffer(partialBuf, 0, 0);
        enc->setBuffer(finalBuf, 0, 1);
        enc->setBytes(&numTGs, sizeof(numTGs), 2);
        enc->dispatchThreads(MTL::Size::Make(1, 1, 1), MTL::Size::Make(1, 1, 1));

        enc->endEncoding();
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();

        if (iter == 2)
            gpuMs = (cmdBuf->GPUEndTime() - cmdBuf->GPUStartTime()) * 1000.0;
    }

    auto postStart = std::chrono::high_resolution_clock::now();
    float revenue = *(float*)finalBuf->contents();
    auto postEnd = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nTPC-H Gen-Q19 Result: Revenue = $%.2f\n", revenue);
    printf("\nGen-Q19 | %u lineitem, %u part\n", liSize, partSize);
    printTimingSummary(parseMs, gpuMs, postMs);

    releaseAll(pPartKeyBuf, pBrandBuf, pContainerBuf, pSizeBuf,
               smBuf, siBuf, qualifiesBuf,
               partkeyBuf, qtyBuf, priceBuf, discBuf,
               mapBuf, partialBuf, finalBuf);
}

} // anonymous namespace

// ===================================================================
// PUBLIC: executeQuery
// ===================================================================

void executeQuery(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                  const QueryPlan& plan,
                  const RuntimeCompiler::CompiledQuery& compiled,
                  const GeneratedKernels& /*gen*/,
                  const std::string& dataDir) {
    if (plan.name == "Q1")  executeQ1(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q6")  executeQ6(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q3")  executeQ3(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q14") executeQ14(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q13") executeQ13(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q4")  executeQ4(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q12") executeQ12(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q19") executeQ19(device, cmdQueue, compiled, dataDir);
    else throw std::runtime_error("No executor for plan: " + plan.name);
}

} // namespace codegen
