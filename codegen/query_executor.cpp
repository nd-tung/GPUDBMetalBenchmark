#include "query_executor.h"
#include "tpch_schema.h"
#include "../src/infra.h"

#include <chrono>
#include <iostream>
#include <algorithm>
#include <map>
#include <set>
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

    auto* pso = findPSO(cq, "Q1_reduce");
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

    auto* s1pso = findPSO(cq, "Q6_reduce");
    auto* s2pso = findPSO(cq, "Q6_reduce_final");
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

    auto* bmPSO = findPSO(cq, "Q3_bitmap_build");
    auto* mapPSO = findPSO(cq, "Q3_map_build");
    auto* probePSO = findPSO(cq, "Q3_probe_agg");
    auto* compactPSO = findPSO(cq, "Q3_compact");
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

    auto* s1pso = findPSO(cq, "Q14_reduce");
    auto* s2pso = findPSO(cq, "Q14_reduce_final");
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

    auto* countPSO = findPSO(cq, "Q13_str_match_count");
    auto* histPSO = findPSO(cq, "Q13_histogram");
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

    auto* bmPSO = findPSO(cq, "Q4_bitmap_build");
    auto* s1PSO = findPSO(cq, "Q4_reduce");
    auto* s2PSO = findPSO(cq, "Q4_reduce_final");
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

    auto* bmPSO = findPSO(cq, "Q12_bitmap_build");
    auto* s1PSO = findPSO(cq, "Q12_reduce");
    auto* s2PSO = findPSO(cq, "Q12_reduce_final");
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

    auto* mapPSO    = findPSO(cq, "Q19_map_classify");
    auto* filterPSO = findPSO(cq, "Q19_str_filter");
    auto* s1PSO     = findPSO(cq, "Q19_reduce");
    auto* s2PSO     = findPSO(cq, "Q19_reduce_final");
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

// ===================================================================
// Q15 EXECUTOR
// ===================================================================

void executeQ15(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                const RuntimeCompiler::CompiledQuery& cq,
                const std::string& dataDir) {

    auto parseStart = std::chrono::high_resolution_clock::now();
    auto lCols = loadColumnsMulti(dataDir + "lineitem.tbl", {
        {2, ColType::INT}, {5, ColType::FLOAT}, {6, ColType::FLOAT}, {10, ColType::DATE}
    });
    auto& l_suppkey = lCols.ints(2);
    auto& l_shipdate = lCols.ints(10);
    auto& l_extendedprice = lCols.floats(5);
    auto& l_discount = lCols.floats(6);

    auto sCols = loadColumnsMulti(dataDir + "supplier.tbl", {
        {0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}, {2, ColType::CHAR_FIXED, 40}, {4, ColType::CHAR_FIXED, 15}
    });
    auto& s_suppkey = sCols.ints(0);
    auto& s_name = sCols.chars(1);
    auto& s_address = sCols.chars(2);
    auto& s_phone = sCols.chars(4);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    uint liSize = (uint)l_suppkey.size();
    int max_suppkey = 0;
    for (int k : s_suppkey) max_suppkey = std::max(max_suppkey, k);
    uint map_size = max_suppkey + 1;

    auto* pso = findPSO(cq, "Q15_atomic_agg");
    if (!pso) return;

    auto* suppkeyBuf = device->newBuffer(l_suppkey.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* shipdateBuf = device->newBuffer(l_shipdate.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* priceBuf = device->newBuffer(l_extendedprice.data(), liSize * sizeof(float), MTL::ResourceStorageModeShared);
    auto* discBuf = device->newBuffer(l_discount.data(), liSize * sizeof(float), MTL::ResourceStorageModeShared);
    auto* revenueMapBuf = device->newBuffer(map_size * sizeof(float), MTL::ResourceStorageModeShared);

    int date_start = 19960101, date_end = 19960401;

    double gpuMs = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(revenueMapBuf->contents(), 0, map_size * sizeof(float));

        auto* cb = cmdQueue->commandBuffer();
        auto* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(suppkeyBuf, 0, 0);
        enc->setBuffer(shipdateBuf, 0, 1);
        enc->setBuffer(priceBuf, 0, 2);
        enc->setBuffer(discBuf, 0, 3);
        enc->setBuffer(revenueMapBuf, 0, 4);
        enc->setBytes(&liSize, sizeof(liSize), 5);
        enc->setBytes(&date_start, sizeof(date_start), 6);
        enc->setBytes(&date_end, sizeof(date_end), 7);
        enc->dispatchThreads(MTL::Size(liSize, 1, 1), MTL::Size(256, 1, 1));
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        if (iter == 2) gpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
    }

    // CPU post: find max revenue, print matching supplier(s)
    auto postStart = std::chrono::high_resolution_clock::now();
    float* revenue_map = (float*)revenueMapBuf->contents();
    float max_revenue = 0.0f;
    for (uint i = 0; i < map_size; i++)
        if (revenue_map[i] > max_revenue) max_revenue = revenue_map[i];

    std::vector<size_t> supp_index(map_size, SIZE_MAX);
    for (size_t i = 0; i < s_suppkey.size(); i++) supp_index[s_suppkey[i]] = i;

    printf("\nTPC-H Gen-Q15 Results:\n");
    printf("+---------+------------------+------------------+------------------+------------------+\n");
    printf("| suppkey |           s_name |        s_address |          s_phone |    total_revenue |\n");
    printf("+---------+------------------+------------------+------------------+------------------+\n");
    for (uint i = 0; i < map_size; i++) {
        if (revenue_map[i] == max_revenue && max_revenue > 0.0f) {
            size_t si = supp_index[i];
            if (si == SIZE_MAX) continue;
            printf("| %7d | %-16s | %-16s | %-16s | %16.2f |\n",
                   (int)i,
                   trimFixed(s_name.data(), si, 25).c_str(),
                   trimFixed(s_address.data(), si, 40).c_str(),
                   trimFixed(s_phone.data(), si, 15).c_str(),
                   revenue_map[i]);
        }
    }
    printf("+---------+------------------+------------------+------------------+------------------+\n");
    auto postEnd = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nGen-Q15 | %u rows (lineitem)\n", liSize);
    printTimingSummary(parseMs, gpuMs, postMs);

    releaseAll(suppkeyBuf, shipdateBuf, priceBuf, discBuf, revenueMapBuf);
}

// ===================================================================
// Q11 EXECUTOR
// ===================================================================

void executeQ11(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                const RuntimeCompiler::CompiledQuery& cq,
                const std::string& dataDir) {

    auto parseStart = std::chrono::high_resolution_clock::now();
    auto nat = loadNation(dataDir);
    auto s = loadSupplierBasic(dataDir);
    auto& s_suppkey = s.suppkey;
    auto& s_nationkey = s.nationkey;

    auto psCols = loadColumnsMulti(dataDir + "partsupp.tbl", {
        {0, ColType::INT}, {1, ColType::INT}, {2, ColType::INT}, {3, ColType::FLOAT}
    });
    auto& ps_partkey = psCols.ints(0);
    auto& ps_suppkey = psCols.ints(1);
    auto& ps_availqty = psCols.ints(2);
    auto& ps_supplycost = psCols.floats(3);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    uint partsupp_size = (uint)ps_partkey.size();
    uint supplier_size = (uint)s_suppkey.size();

    // CPU: Find GERMANY nationkey, build supplier bitmap
    int germany_nk = findNationKey(nat, "GERMANY");
    if (germany_nk == -1) { std::cerr << "GERMANY not found" << std::endl; return; }

    std::vector<int> germany_keys = {germany_nk};
    auto suppBm = buildSuppBitmapAndIndex(s_suppkey.data(), s_nationkey.data(),
                                           supplier_size, germany_keys);

    auto* pso = findPSO(cq, "Q11_atomic_agg");
    if (!pso) return;

    int max_partkey = 0;
    for (int k : ps_partkey) max_partkey = std::max(max_partkey, k);
    uint value_map_size = (uint)(max_partkey + 1);
    uint tg_size = 256;
    uint num_tg = (partsupp_size + tg_size - 1) / tg_size;

    auto* psPartKeyBuf = device->newBuffer(ps_partkey.data(), partsupp_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* psSuppKeyBuf = device->newBuffer(ps_suppkey.data(), partsupp_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* psSupplyCostBuf = device->newBuffer(ps_supplycost.data(), partsupp_size * sizeof(float), MTL::ResourceStorageModeShared);
    auto* psAvailQtyBuf = device->newBuffer(ps_availqty.data(), partsupp_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* suppBitmapBuf = device->newBuffer(suppBm.bitmap.data(), suppBm.bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);
    auto* valueMapBuf = device->newBuffer(value_map_size * sizeof(float), MTL::ResourceStorageModeShared);
    auto* partialSumsBuf = device->newBuffer(num_tg * sizeof(float), MTL::ResourceStorageModeShared);

    double gpuMs = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(valueMapBuf->contents(), 0, value_map_size * sizeof(float));
        memset(partialSumsBuf->contents(), 0, num_tg * sizeof(float));

        auto* cb = cmdQueue->commandBuffer();
        auto* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(psPartKeyBuf, 0, 0);
        enc->setBuffer(psSuppKeyBuf, 0, 1);
        enc->setBuffer(psSupplyCostBuf, 0, 2);
        enc->setBuffer(psAvailQtyBuf, 0, 3);
        enc->setBuffer(suppBitmapBuf, 0, 4);
        enc->setBuffer(valueMapBuf, 0, 5);
        enc->setBuffer(partialSumsBuf, 0, 6);
        enc->setBytes(&partsupp_size, sizeof(partsupp_size), 7);
        enc->dispatchThreads(MTL::Size(partsupp_size, 1, 1), MTL::Size(tg_size, 1, 1));
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        if (iter == 2) gpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
    }

    // CPU post: compute global sum → threshold, filter+sort
    auto postStart = std::chrono::high_resolution_clock::now();
    float* partial_sums = (float*)partialSumsBuf->contents();
    double global_sum = 0.0;
    for (uint i = 0; i < num_tg; i++) global_sum += partial_sums[i];
    double threshold = global_sum * 0.0001;

    float* value_map = (float*)valueMapBuf->contents();
    struct Q11Result { int partkey; double value; };
    std::vector<Q11Result> results;
    for (uint i = 0; i < value_map_size; i++) {
        if (value_map[i] > threshold)
            results.push_back({(int)i, (double)value_map[i]});
    }
    std::sort(results.begin(), results.end(),
              [](const Q11Result& a, const Q11Result& b) { return a.value > b.value; });

    printf("\nTPC-H Gen-Q11 Results (Top 20 of %zu):\n", results.size());
    printf("+-----------+------------------+\n");
    printf("| ps_partkey|            value |\n");
    printf("+-----------+------------------+\n");
    size_t limit = std::min(results.size(), (size_t)20);
    for (size_t i = 0; i < limit; i++)
        printf("| %9d | %16.2f |\n", results[i].partkey, results[i].value);
    printf("+-----------+------------------+\n");
    printf("Total qualifying rows: %zu, Global sum: %.2f, Threshold: %.2f\n",
           results.size(), global_sum, threshold);
    auto postEnd = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nGen-Q11 | %u rows (partsupp)\n", partsupp_size);
    printTimingSummary(parseMs, gpuMs, postMs);

    releaseAll(psPartKeyBuf, psSuppKeyBuf, psSupplyCostBuf, psAvailQtyBuf,
               suppBitmapBuf, valueMapBuf, partialSumsBuf);
}

// ===================================================================
// Q10 EXECUTOR
// ===================================================================

void executeQ10(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                const RuntimeCompiler::CompiledQuery& cq,
                const std::string& dataDir) {

    auto parseStart = std::chrono::high_resolution_clock::now();
    auto cCols = loadColumnsMulti(dataDir + "customer.tbl", {
        {0, ColType::INT}, {1, ColType::CHAR_FIXED, 18}, {2, ColType::CHAR_FIXED, 40},
        {3, ColType::INT}, {4, ColType::CHAR_FIXED, 15}, {5, ColType::FLOAT}, {7, ColType::CHAR_FIXED, 117}
    });
    auto& c_custkey = cCols.ints(0);
    auto& c_name = cCols.chars(1);
    auto& c_nationkey = cCols.ints(3);
    auto& c_acctbal = cCols.floats(5);

    auto oCols = loadColumnsMulti(dataDir + "orders.tbl", {
        {0, ColType::INT}, {1, ColType::INT}, {4, ColType::DATE}
    });
    auto& o_orderkey = oCols.ints(0);
    auto& o_custkey = oCols.ints(1);
    auto& o_orderdate = oCols.ints(4);

    auto lCols = loadColumnsMulti(dataDir + "lineitem.tbl", {
        {0, ColType::INT}, {5, ColType::FLOAT}, {6, ColType::FLOAT}, {8, ColType::CHAR1}
    });
    auto& l_orderkey = lCols.ints(0);
    auto& l_extendedprice = lCols.floats(5);
    auto& l_discount = lCols.floats(6);
    auto& l_returnflag = lCols.chars(8);

    auto nat = loadNation(dataDir);
    auto nation_names = buildNationNames(nat.nationkey, nat.name.data(), NationData::NAME_WIDTH);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    uint ordSize = (uint)o_orderkey.size();
    uint liSize = (uint)l_orderkey.size();

    // Build custkey → row index
    int max_custkey = 0;
    for (int k : c_custkey) max_custkey = std::max(max_custkey, k);
    std::vector<size_t> cust_index(max_custkey + 1, SIZE_MAX);
    for (size_t i = 0; i < c_custkey.size(); i++) cust_index[c_custkey[i]] = i;

    auto* buildPSO = findPSO(cq, "Q10_map_build");
    auto* probePSO = findPSO(cq, "Q10_probe_agg");
    if (!buildPSO || !probePSO) return;

    int max_orderkey = 0;
    for (int k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    uint map_size = max_orderkey + 1;

    auto* ordKeyBuf = device->newBuffer(o_orderkey.data(), ordSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* ordCustBuf = device->newBuffer(o_custkey.data(), ordSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* ordDateBuf = device->newBuffer(o_orderdate.data(), ordSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* ordersMapBuf = createFilledBuffer(device, (size_t)map_size * sizeof(int), -1);

    auto* liOrdKeyBuf = device->newBuffer(l_orderkey.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* liRetFlagBuf = device->newBuffer(l_returnflag.data(), liSize * sizeof(char), MTL::ResourceStorageModeShared);
    auto* liPriceBuf = device->newBuffer(l_extendedprice.data(), liSize * sizeof(float), MTL::ResourceStorageModeShared);
    auto* liDiscBuf = device->newBuffer(l_discount.data(), liSize * sizeof(float), MTL::ResourceStorageModeShared);

    uint cust_rev_size = max_custkey + 1;
    auto* custRevBuf = device->newBuffer(cust_rev_size * sizeof(float), MTL::ResourceStorageModeShared);

    int date_start = 19931001, date_end = 19940101;

    double gpuMs = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(ordersMapBuf->contents(), -1, (size_t)map_size * sizeof(int));
        memset(custRevBuf->contents(), 0, cust_rev_size * sizeof(float));

        auto* cb = cmdQueue->commandBuffer();
        auto* enc = cb->computeCommandEncoder();

        // Build orders map
        enc->setComputePipelineState(buildPSO);
        enc->setBuffer(ordKeyBuf, 0, 0);
        enc->setBuffer(ordCustBuf, 0, 1);
        enc->setBuffer(ordDateBuf, 0, 2);
        enc->setBuffer(ordersMapBuf, 0, 3);
        enc->setBytes(&ordSize, sizeof(ordSize), 4);
        enc->setBytes(&date_start, sizeof(date_start), 5);
        enc->setBytes(&date_end, sizeof(date_end), 6);
        enc->dispatchThreads(MTL::Size(ordSize, 1, 1), MTL::Size(256, 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // Probe lineitem
        enc->setComputePipelineState(probePSO);
        enc->setBuffer(liOrdKeyBuf, 0, 0);
        enc->setBuffer(liRetFlagBuf, 0, 1);
        enc->setBuffer(liPriceBuf, 0, 2);
        enc->setBuffer(liDiscBuf, 0, 3);
        enc->setBuffer(ordersMapBuf, 0, 4);
        enc->setBuffer(custRevBuf, 0, 5);
        enc->setBytes(&liSize, sizeof(liSize), 6);
        enc->setBytes(&map_size, sizeof(map_size), 7);
        enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));

        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        if (iter == 2) gpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
    }

    // CPU post: Top 20 by revenue DESC using bounded min-heap
    auto postStart = std::chrono::high_resolution_clock::now();
    float* cust_revenue = (float*)custRevBuf->contents();

    struct Q10Result { int custkey; float revenue; };
    auto cmp = [](const Q10Result& a, const Q10Result& b) { return a.revenue > b.revenue; };
    std::vector<Q10Result> topHeap;
    topHeap.reserve(21);
    for (uint i = 0; i < cust_rev_size; i++) {
        if (cust_revenue[i] > 0.0f) {
            if (topHeap.size() < 20) {
                topHeap.push_back({(int)i, cust_revenue[i]});
                std::push_heap(topHeap.begin(), topHeap.end(), cmp);
            } else if (cust_revenue[i] > topHeap.front().revenue) {
                std::pop_heap(topHeap.begin(), topHeap.end(), cmp);
                topHeap.back() = {(int)i, cust_revenue[i]};
                std::push_heap(topHeap.begin(), topHeap.end(), cmp);
            }
        }
    }
    std::sort_heap(topHeap.begin(), topHeap.end(), cmp);

    printf("\nTPC-H Gen-Q10 Results (Top 20):\n");
    printf("+---------+------------------+------------+----------+------------------+\n");
    printf("| custkey |           c_name |    revenue | c_acctbal|           n_name |\n");
    printf("+---------+------------------+------------+----------+------------------+\n");
    for (size_t i = 0; i < topHeap.size(); i++) {
        int ck = topHeap[i].custkey;
        size_t ci = cust_index[ck];
        if (ci == SIZE_MAX) continue;
        printf("| %7d | %-16s | $%10.2f| %8.2f | %-16s |\n",
               ck, trimFixed(c_name.data(), ci, 18).c_str(),
               topHeap[i].revenue, c_acctbal[ci],
               nation_names[c_nationkey[ci]].c_str());
    }
    printf("+---------+------------------+------------+----------+------------------+\n");
    auto postEnd = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nGen-Q10 | %u orders, %u lineitem\n", ordSize, liSize);
    printTimingSummary(parseMs, gpuMs, postMs);

    releaseAll(ordKeyBuf, ordCustBuf, ordDateBuf, ordersMapBuf,
               liOrdKeyBuf, liRetFlagBuf, liPriceBuf, liDiscBuf, custRevBuf);
}

// ===================================================================
// Q5 EXECUTOR
// ===================================================================

void executeQ5(MTL::Device* device, MTL::CommandQueue* cmdQueue,
               const RuntimeCompiler::CompiledQuery& cq,
               const std::string& dataDir) {

    auto parseStart = std::chrono::high_resolution_clock::now();
    auto cCols = loadColumnsMulti(dataDir + "customer.tbl", {{0, ColType::INT}, {3, ColType::INT}});
    auto& c_custkey = cCols.ints(0); auto& c_nationkey = cCols.ints(3);

    auto s = loadSupplierBasic(dataDir);
    auto& s_suppkey = s.suppkey; auto& s_nationkey = s.nationkey;

    auto oCols = loadColumnsMulti(dataDir + "orders.tbl", {{0, ColType::INT}, {1, ColType::INT}, {4, ColType::DATE}});
    auto& o_orderkey = oCols.ints(0); auto& o_custkey = oCols.ints(1); auto& o_orderdate = oCols.ints(4);

    auto lCols = loadColumnsMulti(dataDir + "lineitem.tbl", {{0, ColType::INT}, {2, ColType::INT}, {5, ColType::FLOAT}, {6, ColType::FLOAT}});
    auto& l_orderkey = lCols.ints(0); auto& l_suppkey = lCols.ints(2);
    auto& l_extendedprice = lCols.floats(5); auto& l_discount = lCols.floats(6);

    auto nat = loadNation(dataDir, true);
    auto reg = loadRegion(dataDir);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    const uint customer_size = (uint)c_custkey.size();
    const uint supplier_size = (uint)s_suppkey.size();
    const uint orders_size = (uint)o_orderkey.size();
    const uint lineitem_size = (uint)l_orderkey.size();

    int asia_regionkey = findRegionKey(reg.regionkey, reg.name.data(), RegionData::NAME_WIDTH, "ASIA");
    if (asia_regionkey == -1) { std::cerr << "Error: ASIA region not found" << std::endl; return; }
    auto nation_names = buildNationNames(nat.nationkey, nat.name.data(), NationData::NAME_WIDTH);
    uint cpu_nation_bitmap = buildNationBitmap(nat.nationkey, nat.regionkey, asia_regionkey);

    auto* custMapPSO = findPSO(cq, "Q5_map_build_cust");
    auto* suppMapPSO = findPSO(cq, "Q5_map_build_supp");
    auto* ordMapPSO = findPSO(cq, "Q5_map_build_orders");
    auto* probePSO = findPSO(cq, "Q5_probe_agg");
    if (!custMapPSO || !suppMapPSO || !ordMapPSO || !probePSO) return;

    int max_custkey = 0, max_suppkey = 0, max_orderkey = 0;
    for (int k : c_custkey) max_custkey = std::max(max_custkey, k);
    for (int k : s_suppkey) max_suppkey = std::max(max_suppkey, k);
    for (int k : o_orderkey) max_orderkey = std::max(max_orderkey, k);

    uint cust_map_size = max_custkey + 1;
    uint supp_map_size = max_suppkey + 1;
    uint map_size = max_orderkey + 1;

    auto* pCustKeyBuf = device->newBuffer(c_custkey.data(), customer_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pCustNationBuf = device->newBuffer(c_nationkey.data(), customer_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pCustNationMapBuf = createFilledBuffer(device, cust_map_size * sizeof(int), -1);

    auto* pSuppKeyBuf = device->newBuffer(s_suppkey.data(), supplier_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pSuppNationBuf = device->newBuffer(s_nationkey.data(), supplier_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pSuppNationMapBuf = createFilledBuffer(device, supp_map_size * sizeof(int), -1);

    auto* pNationBitmapBuf = device->newBuffer(&cpu_nation_bitmap, sizeof(uint), MTL::ResourceStorageModeShared);

    auto* pOrdKeyBuf = device->newBuffer(o_orderkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pOrdCustBuf = device->newBuffer(o_custkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pOrdDateBuf = device->newBuffer(o_orderdate.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pOrdersMapBuf = createFilledBuffer(device, (size_t)map_size * sizeof(int), -1);

    auto* pLineOrdKeyBuf = device->newBuffer(l_orderkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pLineSuppKeyBuf = device->newBuffer(l_suppkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pLinePriceBuf = device->newBuffer(l_extendedprice.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);
    auto* pLineDiscBuf = device->newBuffer(l_discount.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);
    auto* pNationRevenueBuf = device->newBuffer(25 * sizeof(float), MTL::ResourceStorageModeShared);

    const int date_start = 19940101;
    const int date_end = 19950101;

    double gpuMs = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(pCustNationMapBuf->contents(), -1, cust_map_size * sizeof(int));
        memset(pSuppNationMapBuf->contents(), -1, supp_map_size * sizeof(int));
        memset(pOrdersMapBuf->contents(), -1, (size_t)map_size * sizeof(int));
        memset(pNationRevenueBuf->contents(), 0, 25 * sizeof(float));

        // Build phase
        auto* cb = cmdQueue->commandBuffer();
        auto* enc = cb->computeCommandEncoder();

        enc->setComputePipelineState(custMapPSO);
        enc->setBuffer(pCustKeyBuf, 0, 0);
        enc->setBuffer(pCustNationBuf, 0, 1);
        enc->setBuffer(pCustNationMapBuf, 0, 2);
        enc->setBuffer(pNationBitmapBuf, 0, 3);
        enc->setBytes(&customer_size, sizeof(customer_size), 4);
        enc->dispatchThreads(MTL::Size(customer_size, 1, 1), MTL::Size(256, 1, 1));

        enc->setComputePipelineState(suppMapPSO);
        enc->setBuffer(pSuppKeyBuf, 0, 0);
        enc->setBuffer(pSuppNationBuf, 0, 1);
        enc->setBuffer(pSuppNationMapBuf, 0, 2);
        enc->setBuffer(pNationBitmapBuf, 0, 3);
        enc->setBytes(&supplier_size, sizeof(supplier_size), 4);
        enc->dispatchThreads(MTL::Size(supplier_size, 1, 1), MTL::Size(256, 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        enc->setComputePipelineState(ordMapPSO);
        enc->setBuffer(pOrdKeyBuf, 0, 0);
        enc->setBuffer(pOrdCustBuf, 0, 1);
        enc->setBuffer(pOrdDateBuf, 0, 2);
        enc->setBuffer(pOrdersMapBuf, 0, 3);
        enc->setBytes(&orders_size, sizeof(orders_size), 4);
        enc->setBytes(&date_start, sizeof(date_start), 5);
        enc->setBytes(&date_end, sizeof(date_end), 6);
        enc->setBytes(&map_size, sizeof(map_size), 7);
        enc->setBuffer(pCustNationMapBuf, 0, 8);
        enc->dispatchThreads(MTL::Size(orders_size, 1, 1), MTL::Size(256, 1, 1));

        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        double buildTime = 0;
        if (iter == 2) buildTime = cb->GPUEndTime() - cb->GPUStartTime();

        // Probe phase
        cb = cmdQueue->commandBuffer();
        enc = cb->computeCommandEncoder();

        enc->setComputePipelineState(probePSO);
        enc->setBuffer(pLineOrdKeyBuf, 0, 0);
        enc->setBuffer(pLineSuppKeyBuf, 0, 1);
        enc->setBuffer(pLinePriceBuf, 0, 2);
        enc->setBuffer(pLineDiscBuf, 0, 3);
        enc->setBuffer(pOrdersMapBuf, 0, 4);
        enc->setBuffer(pSuppNationMapBuf, 0, 5);
        enc->setBuffer(pNationRevenueBuf, 0, 6);
        enc->setBytes(&lineitem_size, sizeof(lineitem_size), 7);
        enc->setBytes(&map_size, sizeof(map_size), 8);
        enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));

        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();

        if (iter == 2) {
            double probeTime = cb->GPUEndTime() - cb->GPUStartTime();
            gpuMs = (buildTime + probeTime) * 1000.0;
        }
    }

    auto postStart = std::chrono::high_resolution_clock::now();
    postProcessQ5((float*)pNationRevenueBuf->contents(), nation_names);
    auto postEnd = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nGen-Q5 | %u rows (lineitem)\n", lineitem_size);
    printTimingSummary(parseMs, gpuMs, postMs);

    releaseAll(pCustKeyBuf, pCustNationBuf, pCustNationMapBuf,
               pSuppKeyBuf, pSuppNationBuf, pSuppNationMapBuf,
               pNationBitmapBuf,
               pOrdKeyBuf, pOrdCustBuf, pOrdDateBuf, pOrdersMapBuf,
               pLineOrdKeyBuf, pLineSuppKeyBuf, pLinePriceBuf, pLineDiscBuf,
               pNationRevenueBuf);
}

// ===================================================================
// Q7 EXECUTOR
// ===================================================================

void executeQ7(MTL::Device* device, MTL::CommandQueue* cmdQueue,
               const RuntimeCompiler::CompiledQuery& cq,
               const std::string& dataDir) {

    auto parseStart = std::chrono::high_resolution_clock::now();
    auto s = loadSupplierBasic(dataDir);
    auto& s_suppkey = s.suppkey; auto& s_nationkey = s.nationkey;

    auto cCols = loadColumnsMulti(dataDir + "customer.tbl", {{0, ColType::INT}, {3, ColType::INT}});
    auto& c_custkey = cCols.ints(0); auto& c_nationkey = cCols.ints(3);

    auto oCols = loadColumnsMulti(dataDir + "orders.tbl", {{0, ColType::INT}, {1, ColType::INT}});
    auto& o_orderkey = oCols.ints(0); auto& o_custkey = oCols.ints(1);

    auto lCols = loadColumnsMulti(dataDir + "lineitem.tbl",
        {{0, ColType::INT}, {2, ColType::INT}, {5, ColType::FLOAT}, {6, ColType::FLOAT}, {10, ColType::DATE}});
    auto& l_orderkey = lCols.ints(0); auto& l_suppkey = lCols.ints(2);
    auto& l_shipdate = lCols.ints(10);
    auto& l_extendedprice = lCols.floats(5); auto& l_discount = lCols.floats(6);

    auto nat = loadNation(dataDir);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    int france_nk = findNationKey(nat, "FRANCE");
    int germany_nk = findNationKey(nat, "GERMANY");
    if (france_nk == -1 || germany_nk == -1) {
        std::cerr << "Error: FRANCE/GERMANY not found" << std::endl;
        return;
    }

    uint suppSize = (uint)s_suppkey.size();
    uint custSize = (uint)c_custkey.size();
    uint ordSize = (uint)o_orderkey.size();
    uint liSize = (uint)l_orderkey.size();

    int max_suppkey = 0, max_custkey = 0, max_orderkey = 0;
    for (int k : s_suppkey) max_suppkey = std::max(max_suppkey, k);
    for (int k : c_custkey) max_custkey = std::max(max_custkey, k);
    for (int k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    uint supp_map_size = max_suppkey + 1;
    uint cust_map_size = max_custkey + 1;
    uint ord_map_size = max_orderkey + 1;

    auto* suppMapPSO = findPSO(cq, "Q7_map_build_supp");
    auto* custMapPSO = findPSO(cq, "Q7_map_build_cust");
    auto* ordMapPSO = findPSO(cq, "Q7_map_build_orders");
    auto* probePSO = findPSO(cq, "Q7_probe_agg");
    if (!suppMapPSO || !custMapPSO || !ordMapPSO || !probePSO) return;

    auto* pSuppKeyBuf = device->newBuffer(s_suppkey.data(), suppSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pSuppNationBuf = device->newBuffer(s_nationkey.data(), suppSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pSuppNationMapBuf = createFilledBuffer(device, supp_map_size * sizeof(int), -1);

    auto* pCustKeyBuf = device->newBuffer(c_custkey.data(), custSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pCustNationBuf = device->newBuffer(c_nationkey.data(), custSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pCustNationMapBuf = createFilledBuffer(device, cust_map_size * sizeof(int), -1);

    auto* pOrdKeyBuf = device->newBuffer(o_orderkey.data(), ordSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pOrdCustBuf = device->newBuffer(o_custkey.data(), ordSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pOrdMapBuf = createFilledBuffer(device, (size_t)ord_map_size * sizeof(int), -1);

    auto* pLineOrdKeyBuf = device->newBuffer(l_orderkey.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pLineSuppKeyBuf = device->newBuffer(l_suppkey.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pLineShipDateBuf = device->newBuffer(l_shipdate.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pLinePriceBuf = device->newBuffer(l_extendedprice.data(), liSize * sizeof(float), MTL::ResourceStorageModeShared);
    auto* pLineDiscBuf = device->newBuffer(l_discount.data(), liSize * sizeof(float), MTL::ResourceStorageModeShared);

    auto* pRevenueBinsBuf = device->newBuffer(4 * sizeof(float), MTL::ResourceStorageModeShared);

    int date_start = 19950101, date_end = 19961231;

    double gpuMs = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(pSuppNationMapBuf->contents(), -1, supp_map_size * sizeof(int));
        memset(pCustNationMapBuf->contents(), -1, cust_map_size * sizeof(int));
        memset(pOrdMapBuf->contents(), -1, (size_t)ord_map_size * sizeof(int));
        memset(pRevenueBinsBuf->contents(), 0, 4 * sizeof(float));

        auto* cb = cmdQueue->commandBuffer();
        auto* enc = cb->computeCommandEncoder();

        // Build supplier map
        enc->setComputePipelineState(suppMapPSO);
        enc->setBuffer(pSuppKeyBuf, 0, 0); enc->setBuffer(pSuppNationBuf, 0, 1);
        enc->setBuffer(pSuppNationMapBuf, 0, 2);
        enc->setBytes(&france_nk, sizeof(france_nk), 3);
        enc->setBytes(&germany_nk, sizeof(germany_nk), 4);
        enc->setBytes(&suppSize, sizeof(suppSize), 5);
        enc->dispatchThreads(MTL::Size(suppSize, 1, 1), MTL::Size(256, 1, 1));

        // Build customer map
        enc->setComputePipelineState(custMapPSO);
        enc->setBuffer(pCustKeyBuf, 0, 0); enc->setBuffer(pCustNationBuf, 0, 1);
        enc->setBuffer(pCustNationMapBuf, 0, 2);
        enc->setBytes(&france_nk, sizeof(france_nk), 3);
        enc->setBytes(&germany_nk, sizeof(germany_nk), 4);
        enc->setBytes(&custSize, sizeof(custSize), 5);
        enc->dispatchThreads(MTL::Size(custSize, 1, 1), MTL::Size(256, 1, 1));

        // Build orders map
        enc->setComputePipelineState(ordMapPSO);
        enc->setBuffer(pOrdKeyBuf, 0, 0); enc->setBuffer(pOrdCustBuf, 0, 1);
        enc->setBuffer(pOrdMapBuf, 0, 2);
        enc->setBytes(&ordSize, sizeof(ordSize), 3);
        enc->dispatchThreads(MTL::Size(ordSize, 1, 1), MTL::Size(256, 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // Probe lineitem
        enc->setComputePipelineState(probePSO);
        enc->setBuffer(pLineOrdKeyBuf, 0, 0); enc->setBuffer(pLineSuppKeyBuf, 0, 1);
        enc->setBuffer(pLineShipDateBuf, 0, 2); enc->setBuffer(pLinePriceBuf, 0, 3);
        enc->setBuffer(pLineDiscBuf, 0, 4);
        enc->setBuffer(pOrdMapBuf, 0, 5); enc->setBuffer(pCustNationMapBuf, 0, 6);
        enc->setBuffer(pSuppNationMapBuf, 0, 7); enc->setBuffer(pRevenueBinsBuf, 0, 8);
        enc->setBytes(&liSize, sizeof(liSize), 9);
        enc->setBytes(&ord_map_size, sizeof(ord_map_size), 10);
        enc->setBytes(&france_nk, sizeof(france_nk), 11);
        enc->setBytes(&germany_nk, sizeof(germany_nk), 12);
        enc->setBytes(&date_start, sizeof(date_start), 13);
        enc->setBytes(&date_end, sizeof(date_end), 14);
        enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));

        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        if (iter == 2) gpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
    }

    auto postStart = std::chrono::high_resolution_clock::now();
    float* bins = (float*)pRevenueBinsBuf->contents();
    printf("\nTPC-H Gen-Q7 Results:\n");
    printf("+----------+----------+--------+-----------------+\n");
    printf("| supp_nat | cust_nat | l_year |         revenue |\n");
    printf("+----------+----------+--------+-----------------+\n");
    const char* pair_supp[] = {"FRANCE", "GERMANY"};
    const char* pair_cust[] = {"GERMANY", "FRANCE"};
    for (int p = 0; p < 2; p++)
        for (int y = 0; y < 2; y++)
            printf("| %-8s | %-8s | %6d | $%14.2f |\n",
                   pair_supp[p], pair_cust[p], 1995 + y, bins[p * 2 + y]);
    printf("+----------+----------+--------+-----------------+\n");
    auto postEnd = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nGen-Q7 | %u lineitem\n", liSize);
    printTimingSummary(parseMs, gpuMs, postMs);

    releaseAll(pSuppKeyBuf, pSuppNationBuf, pSuppNationMapBuf,
               pCustKeyBuf, pCustNationBuf, pCustNationMapBuf,
               pOrdKeyBuf, pOrdCustBuf, pOrdMapBuf,
               pLineOrdKeyBuf, pLineSuppKeyBuf, pLineShipDateBuf, pLinePriceBuf, pLineDiscBuf,
               pRevenueBinsBuf);
}

// ===================================================================
// Q8 EXECUTOR
// ===================================================================

void executeQ8(MTL::Device* device, MTL::CommandQueue* cmdQueue,
               const RuntimeCompiler::CompiledQuery& cq,
               const std::string& dataDir) {

    auto parseStart = std::chrono::high_resolution_clock::now();
    auto pCols = loadColumnsMulti(dataDir + "part.tbl", {{0, ColType::INT}, {4, ColType::CHAR_FIXED, 25}});
    auto& p_partkey = pCols.ints(0); auto& p_type = pCols.chars(4);

    auto s = loadSupplierBasic(dataDir);
    auto& s_suppkey = s.suppkey; auto& s_nationkey = s.nationkey;

    auto cCols = loadColumnsMulti(dataDir + "customer.tbl", {{0, ColType::INT}, {3, ColType::INT}});
    auto& c_custkey = cCols.ints(0); auto& c_nationkey = cCols.ints(3);

    auto oCols = loadColumnsMulti(dataDir + "orders.tbl", {{0, ColType::INT}, {1, ColType::INT}, {4, ColType::DATE}});
    auto& o_orderkey = oCols.ints(0); auto& o_custkey = oCols.ints(1); auto& o_orderdate = oCols.ints(4);

    auto lCols = loadColumnsMulti(dataDir + "lineitem.tbl",
        {{0, ColType::INT}, {1, ColType::INT}, {2, ColType::INT}, {5, ColType::FLOAT}, {6, ColType::FLOAT}});
    auto& l_orderkey = lCols.ints(0); auto& l_partkey = lCols.ints(1); auto& l_suppkey = lCols.ints(2);
    auto& l_extendedprice = lCols.floats(5); auto& l_discount = lCols.floats(6);

    auto nat = loadNation(dataDir, true);
    auto reg = loadRegion(dataDir);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    int america_rk = findRegionKey(reg.regionkey, reg.name.data(), RegionData::NAME_WIDTH, "AMERICA");
    int brazil_nk = findNationKey(nat, "BRAZIL");

    // CPU: Build part bitmap
    auto part_bm = buildCPUBitmap(p_partkey, [&](size_t i) {
        return trimFixed(p_type.data(), i, 25) == "ECONOMY ANODIZED STEEL";
    });

    // CPU: Build customer→nationkey map (AMERICA only)
    int max_custkey = 0;
    for (int k : c_custkey) max_custkey = std::max(max_custkey, k);
    uint cust_map_size = max_custkey + 1;
    std::vector<int> cust_nation_map(cust_map_size, -1);
    uint america_bitmap = buildNationBitmap(nat.nationkey, nat.regionkey, america_rk);
    for (size_t i = 0; i < c_custkey.size(); i++) {
        int nk = c_nationkey[i];
        if ((america_bitmap >> nk) & 1)
            cust_nation_map[c_custkey[i]] = nk;
    }

    // CPU: Build supplier→nationkey map
    int max_suppkey = 0;
    for (int k : s_suppkey) max_suppkey = std::max(max_suppkey, k);
    uint supp_map_size = max_suppkey + 1;
    std::vector<int> supp_nation_map(supp_map_size, -1);
    for (size_t i = 0; i < s_suppkey.size(); i++) supp_nation_map[s_suppkey[i]] = s_nationkey[i];

    uint ordSize = (uint)o_orderkey.size();
    uint liSize = (uint)l_orderkey.size();

    int max_orderkey = 0;
    for (int k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    uint ord_map_size = max_orderkey + 1;

    auto* buildPSO = findPSO(cq, "Q8_map_build");
    auto* probePSO = findPSO(cq, "Q8_probe_agg");
    if (!buildPSO || !probePSO) return;

    auto* pPartBitmapBuf = uploadBitmap(device, part_bm);
    auto* pCustNationMapBuf = device->newBuffer(cust_nation_map.data(), cust_map_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pSuppNationMapBuf = device->newBuffer(supp_nation_map.data(), supp_map_size * sizeof(int), MTL::ResourceStorageModeShared);

    auto* pOrdKeyBuf = device->newBuffer(o_orderkey.data(), ordSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pOrdCustBuf = device->newBuffer(o_custkey.data(), ordSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pOrdDateBuf = device->newBuffer(o_orderdate.data(), ordSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pOrdCustMapBuf = createFilledBuffer(device, (size_t)ord_map_size * sizeof(int), -1);
    auto* pOrdYearMapBuf = device->newBuffer((size_t)ord_map_size * sizeof(int), MTL::ResourceStorageModeShared);

    auto* pLineOrdKeyBuf = device->newBuffer(l_orderkey.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pLinePartKeyBuf = device->newBuffer(l_partkey.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pLineSuppKeyBuf = device->newBuffer(l_suppkey.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pLinePriceBuf = device->newBuffer(l_extendedprice.data(), liSize * sizeof(float), MTL::ResourceStorageModeShared);
    auto* pLineDiscBuf = device->newBuffer(l_discount.data(), liSize * sizeof(float), MTL::ResourceStorageModeShared);

    auto* pResultBinsBuf = device->newBuffer(4 * sizeof(float), MTL::ResourceStorageModeShared);

    int date_start = 19950101, date_end = 19961231;

    double gpuMs = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(pOrdCustMapBuf->contents(), -1, (size_t)ord_map_size * sizeof(int));
        memset(pOrdYearMapBuf->contents(), 0, (size_t)ord_map_size * sizeof(int));
        memset(pResultBinsBuf->contents(), 0, 4 * sizeof(float));

        auto* cb = cmdQueue->commandBuffer();
        auto* enc = cb->computeCommandEncoder();

        // Build orders map
        enc->setComputePipelineState(buildPSO);
        enc->setBuffer(pOrdKeyBuf, 0, 0); enc->setBuffer(pOrdCustBuf, 0, 1);
        enc->setBuffer(pOrdDateBuf, 0, 2);
        enc->setBuffer(pOrdCustMapBuf, 0, 3); enc->setBuffer(pOrdYearMapBuf, 0, 4);
        enc->setBuffer(pCustNationMapBuf, 0, 5);
        enc->setBytes(&ordSize, sizeof(ordSize), 6);
        enc->setBytes(&date_start, sizeof(date_start), 7);
        enc->setBytes(&date_end, sizeof(date_end), 8);
        enc->dispatchThreads(MTL::Size(ordSize, 1, 1), MTL::Size(256, 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // Probe lineitem
        enc->setComputePipelineState(probePSO);
        enc->setBuffer(pLineOrdKeyBuf, 0, 0); enc->setBuffer(pLinePartKeyBuf, 0, 1);
        enc->setBuffer(pLineSuppKeyBuf, 0, 2);
        enc->setBuffer(pLinePriceBuf, 0, 3); enc->setBuffer(pLineDiscBuf, 0, 4);
        enc->setBuffer(pPartBitmapBuf, 0, 5);
        enc->setBuffer(pOrdCustMapBuf, 0, 6); enc->setBuffer(pOrdYearMapBuf, 0, 7);
        enc->setBuffer(pSuppNationMapBuf, 0, 8);
        enc->setBuffer(pResultBinsBuf, 0, 9);
        enc->setBytes(&liSize, sizeof(liSize), 10);
        enc->setBytes(&ord_map_size, sizeof(ord_map_size), 11);
        enc->setBytes(&brazil_nk, sizeof(brazil_nk), 12);
        enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));

        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        if (iter == 2) gpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
    }

    auto postStart = std::chrono::high_resolution_clock::now();
    float* bins = (float*)pResultBinsBuf->contents();
    printf("\nTPC-H Gen-Q8 Results:\n");
    printf("+--------+------------+\n");
    printf("| o_year |  mkt_share |\n");
    printf("+--------+------------+\n");
    for (int y = 0; y < 2; y++) {
        float total = bins[2 + y];
        float mkt_share = (total > 0.0f) ? bins[y] / total : 0.0f;
        printf("| %6d | %10.6f |\n", 1995 + y, mkt_share);
    }
    printf("+--------+------------+\n");
    auto postEnd = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nGen-Q8 | %u lineitem\n", liSize);
    printTimingSummary(parseMs, gpuMs, postMs);

    releaseAll(pPartBitmapBuf, pCustNationMapBuf, pSuppNationMapBuf,
               pOrdKeyBuf, pOrdCustBuf, pOrdDateBuf, pOrdCustMapBuf, pOrdYearMapBuf,
               pLineOrdKeyBuf, pLinePartKeyBuf, pLineSuppKeyBuf, pLinePriceBuf, pLineDiscBuf,
               pResultBinsBuf);
}

// ===================================================================
// Q17: Small-Quantity-Order Revenue
// ===================================================================
static void executeQ17(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                       const RuntimeCompiler::CompiledQuery& compiled,
                       const std::string& dataDir) {
    auto parseStart = std::chrono::high_resolution_clock::now();
    auto pCols = loadColumnsMulti(dataDir + "part.tbl",
        {{0, ColType::INT}, {3, ColType::CHAR_FIXED, 10}, {6, ColType::CHAR_FIXED, 10}});
    auto& p_partkey = pCols.ints(0); auto& p_brand = pCols.chars(3); auto& p_container = pCols.chars(6);

    auto lCols = loadColumnsMulti(dataDir + "lineitem.tbl",
        {{1, ColType::INT}, {4, ColType::FLOAT}, {5, ColType::FLOAT}});
    auto& l_partkey = lCols.ints(1); auto& l_quantity = lCols.floats(4); auto& l_extendedprice = lCols.floats(5);

    // Build part bitmap: Brand#23, MED BOX
    auto part_bm = buildCPUBitmap(p_partkey, [&](size_t i) {
        return trimFixed(p_brand.data(), i, 10) == "Brand#23" &&
               trimFixed(p_container.data(), i, 10) == "MED BOX";
    });
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    uint liSize = (uint)l_partkey.size();
    uint mapSize = part_bm.max_key + 1;

    MTL::Buffer* pPartBitmapBuf = uploadBitmap(device, part_bm);
    MTL::Buffer* pLinePartKeyBuf = device->newBuffer(l_partkey.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineQtyBuf = device->newBuffer(l_quantity.data(), liSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLinePriceBuf = device->newBuffer(l_extendedprice.data(), liSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSumQtyMapBuf = device->newBuffer((size_t)mapSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCountMapBuf = device->newBuffer((size_t)mapSize * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* pThresholdMapBuf = device->newBuffer((size_t)mapSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pTotalRevenueBuf = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    memset(pSumQtyMapBuf->contents(), 0, (size_t)mapSize * sizeof(float));
    memset(pCountMapBuf->contents(), 0, (size_t)mapSize * sizeof(uint));
    *(float*)pTotalRevenueBuf->contents() = 0.0f;

    // Pass 1: aggregate qty stats
    MTL::CommandBuffer* cb1 = cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* enc1 = cb1->computeCommandEncoder();
    enc1->setComputePipelineState(findPSO(compiled, "Q17_atomic_agg"));
    enc1->setBuffer(pLinePartKeyBuf, 0, 0);
    enc1->setBuffer(pLineQtyBuf, 0, 1);
    enc1->setBuffer(pPartBitmapBuf, 0, 2);
    enc1->setBuffer(pSumQtyMapBuf, 0, 3);
    enc1->setBuffer(pCountMapBuf, 0, 4);
    enc1->setBytes(&liSize, sizeof(liSize), 5);
    enc1->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));
    enc1->endEncoding();
    cb1->commit(); cb1->waitUntilCompleted();

    // CPU: compute threshold = 0.2 * avg per partkey
    float* sumQty = (float*)pSumQtyMapBuf->contents();
    uint* countQty = (uint*)pCountMapBuf->contents();
    float* threshold = (float*)pThresholdMapBuf->contents();
    for (uint pk = 0; pk < mapSize; pk++) {
        threshold[pk] = (countQty[pk] > 0) ? 0.2f * (sumQty[pk] / (float)countQty[pk]) : 0.0f;
    }

    // Pass 2: sum revenue
    MTL::CommandBuffer* cb2 = cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* enc2 = cb2->computeCommandEncoder();
    enc2->setComputePipelineState(findPSO(compiled, "Q17_probe_reduce"));
    enc2->setBuffer(pLinePartKeyBuf, 0, 0);
    enc2->setBuffer(pLineQtyBuf, 0, 1);
    enc2->setBuffer(pLinePriceBuf, 0, 2);
    enc2->setBuffer(pPartBitmapBuf, 0, 3);
    enc2->setBuffer(pThresholdMapBuf, 0, 4);
    enc2->setBuffer(pTotalRevenueBuf, 0, 5);
    enc2->setBytes(&liSize, sizeof(liSize), 6);
    enc2->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));
    enc2->endEncoding();
    cb2->commit(); cb2->waitUntilCompleted();

    double gpuMs = ((cb1->GPUEndTime() - cb1->GPUStartTime()) +
                    (cb2->GPUEndTime() - cb2->GPUStartTime())) * 1000.0;

    float totalRevenue = *(float*)pTotalRevenueBuf->contents();
    float avgYearly = totalRevenue / 7.0f;
    printf("\nTPC-H Gen-Q17 Result:\n");
    printf("+------------------+\n");
    printf("|      avg_yearly  |\n");
    printf("+------------------+\n");
    printf("| %16.2f |\n", avgYearly);
    printf("+------------------+\n");

    printf("\nGen-Q17 | %u lineitem\n", liSize);
    printTimingSummary(parseMs, gpuMs, 0.0);

    releaseAll(pPartBitmapBuf, pLinePartKeyBuf, pLineQtyBuf, pLinePriceBuf,
               pSumQtyMapBuf, pCountMapBuf, pThresholdMapBuf, pTotalRevenueBuf);
}

// ===================================================================
// Q22: Global Sales Opportunity
// ===================================================================
static std::vector<int> extractPhonePrefixCodegen(const std::vector<char>& phone_chars, int width, size_t count) {
    std::vector<int> prefixes(count);
    for (size_t i = 0; i < count; i++) {
        const char* p = phone_chars.data() + i * width;
        prefixes[i] = (p[0] - '0') * 10 + (p[1] - '0');
    }
    return prefixes;
}

static void executeQ22(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                       const RuntimeCompiler::CompiledQuery& compiled,
                       const std::string& dataDir) {
    auto parseStart = std::chrono::high_resolution_clock::now();
    auto cCols = loadColumnsMulti(dataDir + "customer.tbl",
        {{0, ColType::INT}, {4, ColType::CHAR_FIXED, 15}, {5, ColType::FLOAT}});
    auto& c_custkey = cCols.ints(0); auto& c_phone = cCols.chars(4); auto& c_acctbal = cCols.floats(5);
    auto oCols = loadColumnsMulti(dataDir + "orders.tbl", {{1, ColType::INT}});
    auto& o_custkey = oCols.ints(1);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    uint custSize = (uint)c_custkey.size();
    uint ordSize = (uint)o_custkey.size();

    auto c_prefix = extractPhonePrefixCodegen(c_phone, 15, custSize);

    const int valid_prefixes[] = {13, 17, 18, 23, 29, 30, 31};
    uint valid_prefix_mask = 0;
    int prefix_to_bin[32];
    memset(prefix_to_bin, -1, sizeof(prefix_to_bin));
    for (int i = 0; i < 7; i++) {
        valid_prefix_mask |= (1u << valid_prefixes[i]);
        prefix_to_bin[valid_prefixes[i]] = i;
    }

    int max_custkey = 0;
    for (int k : c_custkey) max_custkey = std::max(max_custkey, k);
    uint cust_bitmap_ints = (max_custkey + 31) / 32 + 1;

    MTL::Buffer* pPrefixBuf = device->newBuffer(c_prefix.data(), custSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pAcctBalBuf = device->newBuffer(c_acctbal.data(), custSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCustKeyBuf = device->newBuffer(c_custkey.data(), custSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdCustKeyBuf = device->newBuffer(o_custkey.data(), ordSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSumBalBuf = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCountBalBuf = device->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCustBitmapBuf = device->newBuffer(cust_bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* pResultCountBuf = device->newBuffer(7 * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* pResultSumBuf = device->newBuffer(7 * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPrefixToBinBuf = device->newBuffer(prefix_to_bin, 32 * sizeof(int), MTL::ResourceStorageModeShared);

    *(float*)pSumBalBuf->contents() = 0.0f;
    *(uint*)pCountBalBuf->contents() = 0;
    memset(pCustBitmapBuf->contents(), 0, cust_bitmap_ints * sizeof(uint));
    memset(pResultCountBuf->contents(), 0, 7 * sizeof(uint));
    memset(pResultSumBuf->contents(), 0, 7 * sizeof(float));

    // Phase 1+2: avg balance + orders bitmap (single command buffer)
    MTL::CommandBuffer* cb1 = cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* enc1 = cb1->computeCommandEncoder();
    enc1->setComputePipelineState(findPSO(compiled, "Q22_scalar_agg"));
    enc1->setBuffer(pPrefixBuf, 0, 0);
    enc1->setBuffer(pAcctBalBuf, 0, 1);
    enc1->setBuffer(pSumBalBuf, 0, 2);
    enc1->setBuffer(pCountBalBuf, 0, 3);
    enc1->setBytes(&custSize, sizeof(custSize), 4);
    enc1->setBytes(&valid_prefix_mask, sizeof(valid_prefix_mask), 5);
    enc1->dispatchThreads(MTL::Size(custSize, 1, 1), MTL::Size(256, 1, 1));

    enc1->memoryBarrier(MTL::BarrierScopeBuffers);
    enc1->setComputePipelineState(findPSO(compiled, "Q22_bitmap_build"));
    enc1->setBuffer(pOrdCustKeyBuf, 0, 0);
    enc1->setBuffer(pCustBitmapBuf, 0, 1);
    enc1->setBytes(&ordSize, sizeof(ordSize), 2);
    enc1->dispatchThreads(MTL::Size(ordSize, 1, 1), MTL::Size(256, 1, 1));
    enc1->endEncoding();
    cb1->commit(); cb1->waitUntilCompleted();

    // CPU: compute avg balance
    float sumBal = *(float*)pSumBalBuf->contents();
    uint countBal = *(uint*)pCountBalBuf->contents();
    float avgBal = (countBal > 0) ? sumBal / countBal : 0.0f;

    // Phase 3: final aggregate
    MTL::CommandBuffer* cb2 = cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* enc2 = cb2->computeCommandEncoder();
    enc2->setComputePipelineState(findPSO(compiled, "Q22_bin_agg"));
    enc2->setBuffer(pPrefixBuf, 0, 0);
    enc2->setBuffer(pAcctBalBuf, 0, 1);
    enc2->setBuffer(pCustKeyBuf, 0, 2);
    enc2->setBuffer(pCustBitmapBuf, 0, 3);
    enc2->setBuffer(pResultCountBuf, 0, 4);
    enc2->setBuffer(pResultSumBuf, 0, 5);
    enc2->setBytes(&custSize, sizeof(custSize), 6);
    enc2->setBytes(&avgBal, sizeof(avgBal), 7);
    enc2->setBytes(&valid_prefix_mask, sizeof(valid_prefix_mask), 8);
    enc2->setBuffer(pPrefixToBinBuf, 0, 9);
    enc2->dispatchThreads(MTL::Size(custSize, 1, 1), MTL::Size(256, 1, 1));
    enc2->endEncoding();
    cb2->commit(); cb2->waitUntilCompleted();

    double gpuMs = ((cb1->GPUEndTime() - cb1->GPUStartTime()) +
                    (cb2->GPUEndTime() - cb2->GPUStartTime())) * 1000.0;

    uint* counts = (uint*)pResultCountBuf->contents();
    float* sums = (float*)pResultSumBuf->contents();
    printf("\nTPC-H Gen-Q22 Results:\n");
    printf("+----------+----------+---------------+\n");
    printf("| cntrycode|  numcust |    totacctbal |\n");
    printf("+----------+----------+---------------+\n");
    for (int i = 0; i < 7; i++) {
        if (counts[i] > 0)
            printf("| %8d | %8u | %13.2f |\n", valid_prefixes[i], counts[i], sums[i]);
    }
    printf("+----------+----------+---------------+\n");

    printf("\nGen-Q22 | %u customers | %u orders\n", custSize, ordSize);
    printTimingSummary(parseMs, gpuMs, 0.0);

    releaseAll(pPrefixBuf, pAcctBalBuf, pCustKeyBuf, pOrdCustKeyBuf,
               pSumBalBuf, pCountBalBuf, pCustBitmapBuf,
               pResultCountBuf, pResultSumBuf, pPrefixToBinBuf);
}

// ===================================================================
// Q2: Minimum Cost Supplier
// ===================================================================
static void executeQ2(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                      const RuntimeCompiler::CompiledQuery& compiled,
                      const std::string& dataDir) {
    auto parseStart = std::chrono::high_resolution_clock::now();
    auto pCols = loadColumnsMulti(dataDir + "part.tbl",
        {{0, ColType::INT}, {2, ColType::CHAR_FIXED, 25}, {4, ColType::CHAR_FIXED, 25}, {5, ColType::INT}});
    auto& p_partkey = pCols.ints(0); auto& p_mfgr = pCols.chars(2);
    auto& p_type = pCols.chars(4); auto& p_size = pCols.ints(5);

    auto sCols = loadColumnsMulti(dataDir + "supplier.tbl", {
        {0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}, {2, ColType::CHAR_FIXED, 40},
        {3, ColType::INT}, {4, ColType::CHAR_FIXED, 15}, {5, ColType::FLOAT}, {6, ColType::CHAR_FIXED, 101}
    });
    auto& s_suppkey = sCols.ints(0); auto& s_name = sCols.chars(1); auto& s_address = sCols.chars(2);
    auto& s_nationkey = sCols.ints(3); auto& s_phone = sCols.chars(4); auto& s_acctbal = sCols.floats(5);
    auto& s_comment = sCols.chars(6);

    auto psCols = loadColumnsMulti(dataDir + "partsupp.tbl",
        {{0, ColType::INT}, {1, ColType::INT}, {3, ColType::FLOAT}});
    auto& ps_partkey = psCols.ints(0); auto& ps_suppkey = psCols.ints(1); auto& ps_supplycost = psCols.floats(3);

    auto nCols = loadColumnsMulti(dataDir + "nation.tbl",
        {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}, {2, ColType::INT}});
    auto& n_nationkey = nCols.ints(0); auto& n_name = nCols.chars(1); auto& n_regionkey = nCols.ints(2);

    auto rCols = loadColumnsMulti(dataDir + "region.tbl",
        {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}});
    auto& r_regionkey = rCols.ints(0); auto& r_name = rCols.chars(1);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    uint part_size = (uint)p_partkey.size();
    uint partsupp_size = (uint)ps_partkey.size();

    // CPU: EUROPE nations and supplier bitmap
    int europe_regionkey = findRegionKey(r_regionkey, r_name.data(), 25, "EUROPE");
    if (europe_regionkey == -1) { std::cerr << "EUROPE not found" << std::endl; return; }
    auto nation_names = buildNationNames(n_nationkey, n_name.data(), 25);
    auto europe_nation_keys = filterNationsByRegion(n_nationkey, n_regionkey, europe_regionkey);
    auto suppBitmap = buildSuppBitmapAndIndex(s_suppkey.data(), s_nationkey.data(),
                                               (size_t)s_suppkey.size(), europe_nation_keys);

    int max_partkey = 0;
    for (int k : p_partkey) max_partkey = std::max(max_partkey, k);
    uint part_bitmap_ints = (max_partkey + 31) / 32 + 1;

    MTL::Buffer* pPartKeyBuf = device->newBuffer(p_partkey.data(), part_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartSizeBuf = device->newBuffer(p_size.data(), part_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartTypeBuf = device->newBuffer(p_type.data(), p_type.size() * sizeof(char), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartBitmapBuf = device->newBuffer(part_bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSuppBitmapBuf = device->newBuffer(suppBitmap.bitmap.data(), suppBitmap.bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsPartKeyBuf = device->newBuffer(ps_partkey.data(), partsupp_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsSuppKeyBuf = device->newBuffer(ps_suppkey.data(), partsupp_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsSupplyCostBuf = device->newBuffer(ps_supplycost.data(), partsupp_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pMinCostBuf = device->newBuffer((max_partkey + 1) * sizeof(uint), MTL::ResourceStorageModeShared);
    const uint max_results = 10000;
    // GenQ2MatchResult = {int, int, uint} = 12 bytes
    MTL::Buffer* pResultsBuf = device->newBuffer(max_results * 12, MTL::ResourceStorageModeShared);
    MTL::Buffer* pResultCountBuf = device->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared);

    memset(pPartBitmapBuf->contents(), 0, part_bitmap_ints * sizeof(uint));
    memset(pMinCostBuf->contents(), 0xFF, (max_partkey + 1) * sizeof(uint));
    *(uint*)pResultCountBuf->contents() = 0;
    int target_size = 15;

    // GPU: 3 stages in single command buffer
    MTL::CommandBuffer* cb = cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

    // Stage 1: filter parts → bitmap
    enc->setComputePipelineState(findPSO(compiled, "Q2_bitmap_build"));
    enc->setBuffer(pPartKeyBuf, 0, 0);
    enc->setBuffer(pPartSizeBuf, 0, 1);
    enc->setBuffer(pPartTypeBuf, 0, 2);
    enc->setBuffer(pPartBitmapBuf, 0, 3);
    enc->setBytes(&part_size, sizeof(part_size), 4);
    enc->setBytes(&target_size, sizeof(target_size), 5);
    enc->dispatchThreads(MTL::Size(part_size, 1, 1), MTL::Size(256, 1, 1));

    // Stage 2: find min cost per partkey
    enc->memoryBarrier(MTL::BarrierScopeBuffers);
    enc->setComputePipelineState(findPSO(compiled, "Q2_atomic_min"));
    enc->setBuffer(pPsPartKeyBuf, 0, 0);
    enc->setBuffer(pPsSuppKeyBuf, 0, 1);
    enc->setBuffer(pPsSupplyCostBuf, 0, 2);
    enc->setBuffer(pPartBitmapBuf, 0, 3);
    enc->setBuffer(pSuppBitmapBuf, 0, 4);
    enc->setBuffer(pMinCostBuf, 0, 5);
    enc->setBytes(&partsupp_size, sizeof(partsupp_size), 6);
    enc->dispatchThreads(MTL::Size(partsupp_size, 1, 1), MTL::Size(256, 1, 1));

    // Stage 3: match suppliers → compact output
    enc->memoryBarrier(MTL::BarrierScopeBuffers);
    enc->setComputePipelineState(findPSO(compiled, "Q2_compact"));
    enc->setBuffer(pPsPartKeyBuf, 0, 0);
    enc->setBuffer(pPsSuppKeyBuf, 0, 1);
    enc->setBuffer(pPsSupplyCostBuf, 0, 2);
    enc->setBuffer(pPartBitmapBuf, 0, 3);
    enc->setBuffer(pSuppBitmapBuf, 0, 4);
    enc->setBuffer(pMinCostBuf, 0, 5);
    enc->setBuffer(pResultsBuf, 0, 6);
    enc->setBuffer(pResultCountBuf, 0, 7);
    enc->setBytes(&partsupp_size, sizeof(partsupp_size), 8);
    enc->setBytes(&max_results, sizeof(max_results), 9);
    enc->dispatchThreads(MTL::Size(partsupp_size, 1, 1), MTL::Size(256, 1, 1));

    enc->endEncoding();
    cb->commit(); cb->waitUntilCompleted();
    double gpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;

    // CPU post-processing
    auto postStart = std::chrono::high_resolution_clock::now();
    uint result_count = std::min(*(uint*)pResultCountBuf->contents(), max_results);
    postProcessQ2((Q2MatchResult_CPU*)pResultsBuf->contents(), result_count,
                  suppBitmap.index, s_acctbal.data(), s_nationkey.data(),
                  s_name.data(), s_address.data(), s_phone.data(), s_comment.data(),
                  nation_names, p_partkey.data(), part_size, p_mfgr.data());
    auto postEnd = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nGen-Q2 | %u partsupp\n", partsupp_size);
    printTimingSummary(parseMs, gpuMs, postMs);

    releaseAll(pPartKeyBuf, pPartSizeBuf, pPartTypeBuf, pPartBitmapBuf, pSuppBitmapBuf,
               pPsPartKeyBuf, pPsSuppKeyBuf, pPsSupplyCostBuf,
               pMinCostBuf, pResultsBuf, pResultCountBuf);
}

// ===================================================================
// Q18: Large Volume Customer
// ===================================================================
static void executeQ18(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                       const RuntimeCompiler::CompiledQuery& compiled,
                       const std::string& dataDir) {
    auto parseStart = std::chrono::high_resolution_clock::now();
    auto cCols = loadColumnsMulti(dataDir + "customer.tbl",
        {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}});
    auto& c_custkey = cCols.ints(0); auto& c_name = cCols.chars(1);

    auto oCols = loadColumnsMulti(dataDir + "orders.tbl",
        {{0, ColType::INT}, {1, ColType::INT}, {3, ColType::FLOAT}, {4, ColType::DATE}});
    auto& o_orderkey = oCols.ints(0); auto& o_custkey = oCols.ints(1);
    auto& o_totalprice = oCols.floats(3); auto& o_orderdate = oCols.ints(4);

    auto lCols = loadColumnsMulti(dataDir + "lineitem.tbl",
        {{0, ColType::INT}, {4, ColType::FLOAT}});
    auto& l_orderkey = lCols.ints(0); auto& l_quantity = lCols.floats(4);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    uint liSize = (uint)l_orderkey.size();
    uint ordSize = (uint)o_orderkey.size();

    int max_orderkey = 0;
    for (int k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    uint qty_map_size = max_orderkey + 1;

    MTL::Buffer* pLineOrdKeyBuf = device->newBuffer(l_orderkey.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineQtyBuf = device->newBuffer(l_quantity.data(), liSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pQtyMapBuf = device->newBuffer((size_t)qty_map_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdKeyBuf = device->newBuffer(o_orderkey.data(), ordSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdCustKeyBuf = device->newBuffer(o_custkey.data(), ordSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdDateBuf = device->newBuffer(o_orderdate.data(), ordSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdPriceBuf = device->newBuffer(o_totalprice.data(), ordSize * sizeof(float), MTL::ResourceStorageModeShared);
    // GenQ18OutputRow = {int, int, int, float, float} = 20 bytes
    MTL::Buffer* pOutputBuf = device->newBuffer((size_t)ordSize * 20, MTL::ResourceStorageModeShared);
    MTL::Buffer* pOutputCountBuf = device->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared);

    memset(pQtyMapBuf->contents(), 0, (size_t)qty_map_size * sizeof(float));
    *(uint*)pOutputCountBuf->contents() = 0;
    float threshold = 300.0f;

    // GPU: 2 kernels in single command buffer
    MTL::CommandBuffer* cb = cmdQueue->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

    enc->setComputePipelineState(findPSO(compiled, "Q18_atomic_agg"));
    enc->setBuffer(pLineOrdKeyBuf, 0, 0);
    enc->setBuffer(pLineQtyBuf, 0, 1);
    enc->setBuffer(pQtyMapBuf, 0, 2);
    enc->setBytes(&liSize, sizeof(liSize), 3);
    enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));

    enc->memoryBarrier(MTL::BarrierScopeBuffers);

    enc->setComputePipelineState(findPSO(compiled, "Q18_compact"));
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
    double gpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;

    // CPU post: join customer name, sort top-100
    auto postStart = std::chrono::high_resolution_clock::now();
    uint outputCount = *(uint*)pOutputCountBuf->contents();
    struct Q18Row { int o_orderkey; int o_custkey; int o_orderdate; float o_totalprice; float sum_qty; };
    auto* gpuRows = (Q18Row*)pOutputBuf->contents();

    int max_custkey = 0;
    for (int k : c_custkey) max_custkey = std::max(max_custkey, k);
    std::vector<int> cust_index(max_custkey + 1, -1);
    for (size_t i = 0; i < c_custkey.size(); i++) cust_index[c_custkey[i]] = (int)i;

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
            name = trimFixed(c_name.data(), cust_index[ck], 25);
        results.push_back({name, ck, gpuRows[i].o_orderkey, gpuRows[i].o_orderdate,
                           gpuRows[i].o_totalprice, gpuRows[i].sum_qty});
    }

    size_t topK = std::min((size_t)100, results.size());
    std::partial_sort(results.begin(), results.begin() + topK, results.end(),
        [](const Q18Result& a, const Q18Result& b) {
            if (a.o_totalprice != b.o_totalprice) return a.o_totalprice > b.o_totalprice;
            return a.o_orderdate < b.o_orderdate;
        });

    printf("\nTPC-H Gen-Q18 Results (Top 10 of LIMIT 100):\n");
    printf("+------------------+----------+----------+------------+-------------+---------+\n");
    printf("| c_name           | c_custkey| o_orderkey| o_orderdate| o_totalprice| sum_qty |\n");
    printf("+------------------+----------+----------+------------+-------------+---------+\n");
    size_t show = std::min((size_t)10, topK);
    for (size_t i = 0; i < show; i++) {
        printf("| %-16s | %8d | %9d | %10d | %11.2f | %7.2f |\n",
               results[i].c_name.c_str(), results[i].c_custkey, results[i].o_orderkey,
               results[i].o_orderdate, results[i].o_totalprice, results[i].sum_qty);
    }
    printf("+------------------+----------+----------+------------+-------------+---------+\n");
    printf("Total qualifying orders: %zu\n", results.size());
    auto postEnd = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nGen-Q18 | %u lineitem\n", liSize);
    printTimingSummary(parseMs, gpuMs, postMs);

    releaseAll(pLineOrdKeyBuf, pLineQtyBuf, pQtyMapBuf,
               pOrdKeyBuf, pOrdCustKeyBuf, pOrdDateBuf, pOrdPriceBuf,
               pOutputBuf, pOutputCountBuf);
}

// ===================================================================
// Q9: Product Type Profit Measure
// ===================================================================
static void executeQ9(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                      const RuntimeCompiler::CompiledQuery& compiled,
                      const std::string& dataDir) {
    auto parseStart = std::chrono::high_resolution_clock::now();
    auto pCols = loadColumnsMulti(dataDir + "part.tbl", {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 55}});
    auto& p_partkey = pCols.ints(0); auto& p_name = pCols.chars(1);

    auto s = loadSupplierBasic(dataDir);
    auto& s_suppkey = s.suppkey; auto& s_nationkey = s.nationkey;

    auto lCols = loadColumnsMulti(dataDir + "lineitem.tbl",
        {{0, ColType::INT}, {1, ColType::INT}, {2, ColType::INT}, {4, ColType::FLOAT}, {5, ColType::FLOAT}, {6, ColType::FLOAT}});
    auto& l_orderkey = lCols.ints(0); auto& l_partkey = lCols.ints(1); auto& l_suppkey = lCols.ints(2);
    auto& l_quantity = lCols.floats(4); auto& l_extendedprice = lCols.floats(5); auto& l_discount = lCols.floats(6);

    auto psCols = loadColumnsMulti(dataDir + "partsupp.tbl", {{0, ColType::INT}, {1, ColType::INT}, {3, ColType::FLOAT}});
    auto& ps_partkey = psCols.ints(0); auto& ps_suppkey = psCols.ints(1); auto& ps_supplycost = psCols.floats(3);

    auto oCols = loadColumnsMulti(dataDir + "orders.tbl", {{0, ColType::INT}, {4, ColType::DATE}});
    auto& o_orderkey = oCols.ints(0); auto& o_orderdate = oCols.ints(4);

    auto nat = loadNation(dataDir);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    auto nation_names = buildNationNames(nat.nationkey, nat.name.data(), NationData::NAME_WIDTH);

    const uint part_size = (uint)p_partkey.size();
    const uint supplier_size = (uint)s_suppkey.size();
    const uint lineitem_size = (uint)l_partkey.size();
    const uint partsupp_size = (uint)ps_partkey.size();
    const uint orders_size = (uint)o_orderkey.size();

    int max_partkey = 0, max_suppkey = 0;
    for (int k : p_partkey) max_partkey = std::max(max_partkey, k);
    for (int k : s_suppkey) max_suppkey = std::max(max_suppkey, k);
    uint supp_map_size = max_suppkey + 1;
    uint partsupp_ht_size = nextPow2(partsupp_size);
    uint orders_ht_size = nextPow2(orders_size * 2);
    uint agg_size = nextPow2(25 * 10); // 25 nations × ~10 years

    auto* pPartKeyBuf = device->newBuffer(p_partkey.data(), part_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pPartNameBuf = device->newBuffer(p_name.data(), p_name.size() * sizeof(char), MTL::ResourceStorageModeShared);
    auto* pPartBitmapBuf = createBitmapBuffer(device, max_partkey);
    auto* pSuppKeyBuf = device->newBuffer(s_suppkey.data(), supplier_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pSuppNationBuf = device->newBuffer(s_nationkey.data(), supplier_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pSuppMapBuf = createFilledBuffer(device, supp_map_size * sizeof(int), -1);
    auto* pPsPartKeyBuf = device->newBuffer(ps_partkey.data(), partsupp_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pPsSuppKeyBuf = device->newBuffer(ps_suppkey.data(), partsupp_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pPsSupplyCostBuf = device->newBuffer(ps_supplycost.data(), partsupp_size * sizeof(float), MTL::ResourceStorageModeShared);
    // PartSuppEntry: 4 ints per entry (partkey, suppkey, idx, pad)
    auto* pPartSuppHTBuf = device->newBuffer((size_t)partsupp_ht_size * 4 * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pOrdKeyBuf = device->newBuffer(o_orderkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pOrdDateBuf = device->newBuffer(o_orderdate.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    // OrdersHTEntry: 2 ints per entry (key, value)
    auto* pOrdersHTBuf = device->newBuffer((size_t)orders_ht_size * 2 * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pLineSuppKeyBuf = device->newBuffer(l_suppkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pLinePartKeyBuf = device->newBuffer(l_partkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pLineOrdKeyBuf = device->newBuffer(l_orderkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pLinePriceBuf = device->newBuffer(l_extendedprice.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);
    auto* pLineDiscBuf = device->newBuffer(l_discount.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);
    auto* pLineQtyBuf = device->newBuffer(l_quantity.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);
    // Q9Aggregates: 2 uints per entry (key, profit-as-float)
    auto* pAggBuf = device->newBuffer((size_t)agg_size * 2 * sizeof(uint), MTL::ResourceStorageModeShared);

    double gpuMs = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        memset(pPartBitmapBuf->contents(), 0, pPartBitmapBuf->length());
        memset(pSuppMapBuf->contents(), -1, supp_map_size * sizeof(int));
        memset(pPartSuppHTBuf->contents(), 0xFF, (size_t)partsupp_ht_size * 4 * sizeof(int));
        memset(pOrdersHTBuf->contents(), 0xFF, (size_t)orders_ht_size * 2 * sizeof(int));
        memset(pAggBuf->contents(), 0, (size_t)agg_size * 2 * sizeof(uint));

        // Build phase: 4 kernels in single encoder
        auto* cb = cmdQueue->commandBuffer();
        auto* enc = cb->computeCommandEncoder();

        // K1: build part bitmap
        enc->setComputePipelineState(findPSO(compiled, "Q9_bitmap_build"));
        enc->setBuffer(pPartKeyBuf, 0, 0);
        enc->setBuffer(pPartNameBuf, 0, 1);
        enc->setBuffer(pPartBitmapBuf, 0, 2);
        enc->setBytes(&part_size, sizeof(part_size), 3);
        enc->dispatchThreads(MTL::Size(part_size, 1, 1), MTL::Size(256, 1, 1));

        // K2: build supplier map
        enc->setComputePipelineState(findPSO(compiled, "Q9_map_build"));
        enc->setBuffer(pSuppKeyBuf, 0, 0);
        enc->setBuffer(pSuppNationBuf, 0, 1);
        enc->setBuffer(pSuppMapBuf, 0, 2);
        enc->setBytes(&supplier_size, sizeof(supplier_size), 3);
        enc->dispatchThreads(MTL::Size(supplier_size, 1, 1), MTL::Size(256, 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // K3: build partsupp HT (depends on part bitmap)
        enc->setComputePipelineState(findPSO(compiled, "Q9_ht_build_partsupp"));
        enc->setBuffer(pPsPartKeyBuf, 0, 0);
        enc->setBuffer(pPsSuppKeyBuf, 0, 1);
        enc->setBuffer(pPartSuppHTBuf, 0, 2);
        enc->setBytes(&partsupp_size, sizeof(partsupp_size), 3);
        enc->setBytes(&partsupp_ht_size, sizeof(partsupp_ht_size), 4);
        enc->setBuffer(pPartBitmapBuf, 0, 5);
        enc->dispatchThreads(MTL::Size(partsupp_size, 1, 1), MTL::Size(256, 1, 1));

        // K4: build orders HT (independent)
        enc->setComputePipelineState(findPSO(compiled, "Q9_ht_build_orders"));
        enc->setBuffer(pOrdKeyBuf, 0, 0);
        enc->setBuffer(pOrdDateBuf, 0, 1);
        enc->setBuffer(pOrdersHTBuf, 0, 2);
        enc->setBytes(&orders_size, sizeof(orders_size), 3);
        enc->setBytes(&orders_ht_size, sizeof(orders_ht_size), 4);
        enc->dispatchThreads(MTL::Size(orders_size, 1, 1), MTL::Size(256, 1, 1));

        enc->endEncoding();

        // Probe phase: separate encoder for memory ordering
        auto* enc2 = cb->computeCommandEncoder();
        enc2->setComputePipelineState(findPSO(compiled, "Q9_probe_agg"));
        enc2->setBuffer(pLineSuppKeyBuf, 0, 0);
        enc2->setBuffer(pLinePartKeyBuf, 0, 1);
        enc2->setBuffer(pLineOrdKeyBuf, 0, 2);
        enc2->setBuffer(pLinePriceBuf, 0, 3);
        enc2->setBuffer(pLineDiscBuf, 0, 4);
        enc2->setBuffer(pLineQtyBuf, 0, 5);
        enc2->setBuffer(pPsSupplyCostBuf, 0, 6);
        enc2->setBuffer(pPartBitmapBuf, 0, 7);
        enc2->setBuffer(pSuppMapBuf, 0, 8);
        enc2->setBuffer(pPartSuppHTBuf, 0, 9);
        enc2->setBuffer(pOrdersHTBuf, 0, 10);
        enc2->setBuffer(pAggBuf, 0, 11);
        enc2->setBytes(&lineitem_size, sizeof(lineitem_size), 12);
        enc2->setBytes(&partsupp_ht_size, sizeof(partsupp_ht_size), 13);
        enc2->setBytes(&orders_ht_size, sizeof(orders_ht_size), 14);
        enc2->setBytes(&agg_size, sizeof(agg_size), 15);
        enc2->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));
        enc2->endEncoding();

        cb->commit(); cb->waitUntilCompleted();
        if (iter == 2) gpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
    }

    auto postStart = std::chrono::high_resolution_clock::now();
    postProcessQ9(pAggBuf->contents(), agg_size, nation_names);
    auto postEnd = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nGen-Q9 | %u lineitem\n", lineitem_size);
    printTimingSummary(parseMs, gpuMs, postMs);

    releaseAll(pPartKeyBuf, pPartNameBuf, pPartBitmapBuf,
               pSuppKeyBuf, pSuppNationBuf, pSuppMapBuf,
               pPsPartKeyBuf, pPsSuppKeyBuf, pPsSupplyCostBuf, pPartSuppHTBuf,
               pOrdKeyBuf, pOrdDateBuf, pOrdersHTBuf,
               pLineSuppKeyBuf, pLinePartKeyBuf, pLineOrdKeyBuf,
               pLinePriceBuf, pLineDiscBuf, pLineQtyBuf,
               pAggBuf);
}

// ===================================================================
// Q16: Parts/Supplier Relationship
// ===================================================================
static void executeQ16(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                       const RuntimeCompiler::CompiledQuery& compiled,
                       const std::string& dataDir) {
    auto parseStart = std::chrono::high_resolution_clock::now();
    auto pCols = loadColumnsMulti(dataDir + "part.tbl",
        {{0, ColType::INT}, {3, ColType::CHAR_FIXED, 10}, {4, ColType::CHAR_FIXED, 25}, {5, ColType::INT}});
    auto& p_partkey = pCols.ints(0); auto& p_brand = pCols.chars(3);
    auto& p_type = pCols.chars(4); auto& p_size = pCols.ints(5);

    auto psCols = loadColumnsMulti(dataDir + "partsupp.tbl", {{0, ColType::INT}, {1, ColType::INT}});
    auto& ps_partkey = psCols.ints(0); auto& ps_suppkey = psCols.ints(1);

    auto sCols = loadColumnsMulti(dataDir + "supplier.tbl", {{0, ColType::INT}, {6, ColType::CHAR_FIXED, 101}});
    auto& s_suppkey = sCols.ints(0); auto& s_comment = sCols.chars(6);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    // CPU: build complaint bitmap
    auto complaint_bm = buildCPUBitmap(s_suppkey, [&](size_t i) {
        std::string comment = trimFixed(s_comment.data(), i, 101);
        auto pos1 = comment.find("Customer");
        return pos1 != std::string::npos && comment.find("Complaints", pos1) != std::string::npos;
    });

    // CPU: build part group map
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
        } else gid = it->second;
        part_group_map[p_partkey[i]] = gid;
    }

    uint psSize = (uint)ps_partkey.size();
    uint partMapSize = (uint)(max_partkey + 1);
    uint numGroups = (uint)groups.size();

    int max_sk = 0;
    for (int k : ps_suppkey) max_sk = std::max(max_sk, k);
    uint bv_ints = (max_sk + 32) / 32;

    auto* pPsPartKeyBuf = device->newBuffer(ps_partkey.data(), psSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pPsSuppKeyBuf = device->newBuffer(ps_suppkey.data(), psSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pPartGroupMapBuf = device->newBuffer(part_group_map.data(), partMapSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pComplaintBitmapBuf = uploadBitmap(device, complaint_bm);

    size_t bitmapBytes = (size_t)numGroups * bv_ints * sizeof(uint);
    auto* pGroupBitmapsBuf = device->newBuffer(bitmapBytes, MTL::ResourceStorageModeShared);
    auto* pGroupCountsBuf = device->newBuffer(numGroups * sizeof(uint), MTL::ResourceStorageModeShared);

    memset(pGroupBitmapsBuf->contents(), 0, bitmapBytes);
    memset(pGroupCountsBuf->contents(), 0, numGroups * sizeof(uint));

    // GPU: scan + popcount
    auto* cb = cmdQueue->commandBuffer();
    auto* enc = cb->computeCommandEncoder();

    enc->setComputePipelineState(findPSO(compiled, "Q16_group_bitmap"));
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

    enc->setComputePipelineState(findPSO(compiled, "Q16_popcount"));
    enc->setBuffer(pGroupBitmapsBuf, 0, 0);
    enc->setBuffer(pGroupCountsBuf, 0, 1);
    enc->setBytes(&numGroups, sizeof(numGroups), 2);
    enc->setBytes(&bv_ints, sizeof(bv_ints), 3);
    uint tgSizePop = std::min((uint)256, bv_ints);
    if (tgSizePop < 1) tgSizePop = 1;
    enc->dispatchThreadgroups(MTL::Size(numGroups, 1, 1), MTL::Size(tgSizePop, 1, 1));

    enc->endEncoding();
    cb->commit(); cb->waitUntilCompleted();
    double gpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;

    // CPU post: read counts and format
    auto postStart = std::chrono::high_resolution_clock::now();
    uint* gpuGroupCounts = (uint*)pGroupCountsBuf->contents();

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

    printf("\nTPC-H Gen-Q16 Results (Top 10):\n");
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
    double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nGen-Q16 | %u partsupp\n", psSize);
    printTimingSummary(parseMs, gpuMs, postMs);

    releaseAll(pPsPartKeyBuf, pPsSuppKeyBuf, pPartGroupMapBuf, pComplaintBitmapBuf,
               pGroupBitmapsBuf, pGroupCountsBuf);
}

// ===================================================================
// Q20: Potential Part Promotion
// ===================================================================
static void executeQ20(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                       const RuntimeCompiler::CompiledQuery& compiled,
                       const std::string& dataDir) {
    auto parseStart = std::chrono::high_resolution_clock::now();
    auto pCols = loadColumnsMulti(dataDir + "part.tbl", {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 55}});
    auto& p_partkey = pCols.ints(0); auto& p_name = pCols.chars(1);

    auto sCols = loadColumnsMulti(dataDir + "supplier.tbl",
        {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}, {2, ColType::CHAR_FIXED, 40}, {3, ColType::INT}});
    auto& s_suppkey = sCols.ints(0); auto& s_name = sCols.chars(1);
    auto& s_address = sCols.chars(2); auto& s_nationkey = sCols.ints(3);

    auto psCols = loadColumnsMulti(dataDir + "partsupp.tbl", {{0, ColType::INT}, {1, ColType::INT}, {2, ColType::INT}});
    auto& ps_partkey = psCols.ints(0); auto& ps_suppkey = psCols.ints(1); auto& ps_availqty = psCols.ints(2);

    auto lCols = loadColumnsMulti(dataDir + "lineitem.tbl",
        {{1, ColType::INT}, {2, ColType::INT}, {4, ColType::FLOAT}, {10, ColType::DATE}});
    auto& l_partkey = lCols.ints(1); auto& l_suppkey = lCols.ints(2);
    auto& l_quantity = lCols.floats(4); auto& l_shipdate = lCols.ints(10);

    auto nat = loadNation(dataDir);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

    int canada_nk = findNationKey(nat, "CANADA");

    // Build part bitmap: p_name LIKE 'forest%'
    auto part_bm = buildCPUBitmap(p_partkey, [&](size_t i) {
        return trimFixed(p_name.data(), i, 55).substr(0, 6) == "forest";
    });

    // Build CANADA supplier bitmap
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

    uint liSize = (uint)l_partkey.size();
    uint htCapacity = nextPow2(std::max(liSize / 4, 1024u));
    uint htMask = htCapacity - 1;
    int date_start = 19940101, date_end = 19950101;
    uint psSize = (uint)ps_partkey.size();

    // Q20HTEntry: 3 fields (int, int, float) = 12 bytes
    struct Q20HTEntry_CPU { int key_hi; int key_lo; float value; };

    auto* pLinePartKeyBuf = device->newBuffer(l_partkey.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pLineSuppKeyBuf = device->newBuffer(l_suppkey.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pLineQtyBuf = device->newBuffer(l_quantity.data(), liSize * sizeof(float), MTL::ResourceStorageModeShared);
    auto* pLineDateBuf = device->newBuffer(l_shipdate.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pPartBitmapBuf = uploadBitmap(device, part_bm);
    auto* pHTBuf = device->newBuffer((size_t)htCapacity * sizeof(Q20HTEntry_CPU), MTL::ResourceStorageModeShared);
    auto* pPSPartKeyBuf = device->newBuffer(ps_partkey.data(), psSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pPSSuppKeyBuf = device->newBuffer(ps_suppkey.data(), psSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pPSAvailQtyBuf = device->newBuffer(ps_availqty.data(), psSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pCanadaBmBuf = device->newBuffer(canada_bm.data(), canada_bv_ints * sizeof(uint), MTL::ResourceStorageModeShared);
    auto* pQualBmBuf = device->newBuffer(canada_bv_ints * sizeof(uint), MTL::ResourceStorageModeShared);

    // Init HT: key_hi = -1
    auto* ht = (Q20HTEntry_CPU*)pHTBuf->contents();
    for (uint j = 0; j < htCapacity; j++) { ht[j].key_hi = -1; ht[j].key_lo = 0; ht[j].value = 0.0f; }
    memset(pQualBmBuf->contents(), 0, canada_bv_ints * sizeof(uint));

    // GPU: 2 kernels in separate encoders
    auto* cb = cmdQueue->commandBuffer();

    auto* enc = cb->computeCommandEncoder();
    enc->setComputePipelineState(findPSO(compiled, "Q20_ht_agg"));
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

    auto* enc2 = cb->computeCommandEncoder();
    enc2->setComputePipelineState(findPSO(compiled, "Q20_probe_check"));
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
    double gpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;

    // CPU post: read qualifying bitmap, collect supplier names
    auto postStart = std::chrono::high_resolution_clock::now();
    auto* qual_bm = (uint*)pQualBmBuf->contents();
    struct Q20Result { std::string s_name; std::string s_address; };
    std::vector<Q20Result> results;
    for (size_t i = 0; i < s_suppkey.size(); i++) {
        int sk = s_suppkey[i];
        if (sk >= 0 && (uint)sk <= (uint)max_sk && ((qual_bm[sk / 32] >> (sk % 32)) & 1))
            results.push_back({trimFixed(s_name.data(), i, 25), trimFixed(s_address.data(), i, 40)});
    }
    std::sort(results.begin(), results.end(), [](const Q20Result& a, const Q20Result& b) {
        return a.s_name < b.s_name;
    });

    printf("\nTPC-H Gen-Q20 Results (Top 10):\n");
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
    double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nGen-Q20 | %u lineitem\n", liSize);
    printTimingSummary(parseMs, gpuMs, postMs);

    releaseAll(pLinePartKeyBuf, pLineSuppKeyBuf, pLineQtyBuf, pLineDateBuf,
               pPartBitmapBuf, pHTBuf, pPSPartKeyBuf, pPSSuppKeyBuf, pPSAvailQtyBuf,
               pCanadaBmBuf, pQualBmBuf);
}

// ===================================================================
// Q21: Suppliers Who Kept Orders Waiting
// ===================================================================
static void executeQ21(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                       const RuntimeCompiler::CompiledQuery& compiled,
                       const std::string& dataDir) {
    auto parseStart = std::chrono::high_resolution_clock::now();
    auto sCols = loadColumnsMulti(dataDir + "supplier.tbl",
        {{0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}, {3, ColType::INT}});
    auto& s_suppkey = sCols.ints(0); auto& s_name = sCols.chars(1); auto& s_nationkey = sCols.ints(3);

    auto oCols = loadColumnsMulti(dataDir + "orders.tbl", {{0, ColType::INT}, {2, ColType::CHAR1}});
    auto& o_orderkey = oCols.ints(0); auto& o_orderstatus = oCols.chars(2);

    auto lCols = loadColumnsMulti(dataDir + "lineitem.tbl",
        {{0, ColType::INT}, {2, ColType::INT}, {11, ColType::DATE}, {12, ColType::DATE}});
    auto& l_orderkey = lCols.ints(0); auto& l_suppkey = lCols.ints(2);
    auto& l_commitdate = lCols.ints(11); auto& l_receiptdate = lCols.ints(12);

    auto nat = loadNation(dataDir);
    auto parseEnd = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();

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

    auto* pLineOrdKeyBuf = device->newBuffer(l_orderkey.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pLineSuppKeyBuf = device->newBuffer(l_suppkey.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pLineReceiptBuf = device->newBuffer(l_receiptdate.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pLineCommitBuf = device->newBuffer(l_commitdate.data(), liSize * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pOrderStatusMapBuf = device->newBuffer(orders_status_map.data(), map_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pFirstSuppBuf = device->newBuffer((size_t)map_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pMultiSuppBmBuf = device->newBuffer((size_t)bitmapInts * sizeof(uint), MTL::ResourceStorageModeShared);
    auto* pLateSuppBuf = device->newBuffer((size_t)map_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* pMultiLateBmBuf = device->newBuffer((size_t)bitmapInts * sizeof(uint), MTL::ResourceStorageModeShared);
    auto* pSaBitmapBuf = uploadBitmap(device, sa_bm);
    auto* pSuppCountBuf = device->newBuffer((size_t)(max_suppkey + 1) * sizeof(uint), MTL::ResourceStorageModeShared);

    memset(pFirstSuppBuf->contents(), -1, (size_t)map_size * sizeof(int));
    memset(pMultiSuppBmBuf->contents(), 0, (size_t)bitmapInts * sizeof(uint));
    memset(pLateSuppBuf->contents(), -1, (size_t)map_size * sizeof(int));
    memset(pMultiLateBmBuf->contents(), 0, (size_t)bitmapInts * sizeof(uint));
    memset(pSuppCountBuf->contents(), 0, (size_t)(max_suppkey + 1) * sizeof(uint));

    // GPU: 2 passes in single encoder with barrier
    auto* cb = cmdQueue->commandBuffer();
    auto* enc = cb->computeCommandEncoder();

    enc->setComputePipelineState(findPSO(compiled, "Q21_track_build"));
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

    enc->setComputePipelineState(findPSO(compiled, "Q21_count_qualify"));
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
    double gpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;

    // CPU post: collect results, sort top-100
    auto postStart = std::chrono::high_resolution_clock::now();
    uint* suppCounts = (uint*)pSuppCountBuf->contents();

    std::vector<int> supp_idx(max_suppkey + 1, -1);
    for (size_t i = 0; i < s_suppkey.size(); i++) supp_idx[s_suppkey[i]] = (int)i;

    struct Q21Result { std::string s_name; uint numwait; };
    std::vector<Q21Result> results;
    for (int sk = 0; sk <= max_suppkey; sk++) {
        if (suppCounts[sk] > 0 && supp_idx[sk] >= 0)
            results.push_back({trimFixed(s_name.data(), supp_idx[sk], 25), suppCounts[sk]});
    }

    size_t topK = std::min((size_t)100, results.size());
    std::partial_sort(results.begin(), results.begin() + topK, results.end(),
        [](const Q21Result& a, const Q21Result& b) {
            if (a.numwait != b.numwait) return a.numwait > b.numwait;
            return a.s_name < b.s_name;
        });

    printf("\nTPC-H Gen-Q21 Results (Top 10 of LIMIT 100):\n");
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
    double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nGen-Q21 | %u lineitem\n", liSize);
    printTimingSummary(parseMs, gpuMs, postMs);

    releaseAll(pLineOrdKeyBuf, pLineSuppKeyBuf, pLineReceiptBuf, pLineCommitBuf,
               pOrderStatusMapBuf, pFirstSuppBuf, pMultiSuppBmBuf,
               pLateSuppBuf, pMultiLateBmBuf, pSaBitmapBuf, pSuppCountBuf);
}

// ===================================================================
// SF-100 CHUNKED EXECUTORS
// ===================================================================

void executeQ6SF100(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                    const RuntimeCompiler::CompiledQuery& cq,
                    const std::string& dataDir) {
    MappedFile mf;
    if (!mf.open(dataDir + "lineitem.tbl")) {
        std::cerr << "Gen-Q6 SF100: Cannot mmap lineitem.tbl" << std::endl; return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto lineIndex = buildLineIndex(mf);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    size_t totalRows = lineIndex.size();
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 16, totalRows);
    const uint numTG = 2048;
    printf("Gen-Q6 SF100: %zu rows, chunk size: %zu (index %.1f ms)\n", totalRows, chunkRows, indexBuildMs);

    auto* s1pso = findPSO(cq, "Q6_reduce");
    auto* s2pso = findPSO(cq, "Q6_reduce_final");
    if (!s1pso || !s2pso) return;

    struct Q6Slot { MTL::Buffer* shipdate; MTL::Buffer* discount; MTL::Buffer* quantity; MTL::Buffer* extprice; };
    Q6Slot slots[2];
    for (int s = 0; s < 2; s++) {
        slots[s].shipdate = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].discount = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].quantity = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].extprice = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
    }

    auto* partialBuf = device->newBuffer(numTG * sizeof(float), MTL::ResourceStorageModeShared);
    auto* finalBuf = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    int start_date = 19940101, end_date = 19950101;
    float min_discount = 0.05f, max_discount = 0.07f, max_quantity = 24.0f;
    double globalRevenue = 0.0;

    auto timing = chunkedStreamLoop(
        cmdQueue, slots, 2, totalRows, chunkRows,
        [&](Q6Slot& slot, size_t startRow, size_t rowCount) {
            parseDateColumnChunk(mf, lineIndex, startRow, rowCount, 10, (int*)slot.shipdate->contents());
            parseFloatColumnChunk(mf, lineIndex, startRow, rowCount, 6, (float*)slot.discount->contents());
            parseFloatColumnChunk(mf, lineIndex, startRow, rowCount, 4, (float*)slot.quantity->contents());
            parseFloatColumnChunk(mf, lineIndex, startRow, rowCount, 5, (float*)slot.extprice->contents());
        },
        [&](Q6Slot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto* enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(s1pso);
            enc->setBuffer(slot.shipdate, 0, 0);
            enc->setBuffer(slot.discount, 0, 1);
            enc->setBuffer(slot.quantity, 0, 2);
            enc->setBuffer(slot.extprice, 0, 3);
            enc->setBuffer(partialBuf, 0, 4);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 5);
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
        },
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {
            globalRevenue += *(float*)finalBuf->contents();
        }
    );

    printf("\nTPC-H Gen-Q6 Result:\nTotal Revenue: $%.2f\n", globalRevenue);
    printf("\nGen-Q6 SF100 | %zu chunks | %zu rows\n", timing.chunkCount, totalRows);
    printTimingSummary(indexBuildMs + timing.parseMs, timing.gpuMs, timing.postMs);

    for (int s = 0; s < 2; s++)
        releaseAll(slots[s].shipdate, slots[s].discount, slots[s].quantity, slots[s].extprice);
    releaseAll(partialBuf, finalBuf);
}

void executeQ1SF100(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                    const RuntimeCompiler::CompiledQuery& cq,
                    const std::string& dataDir) {
    MappedFile mf;
    if (!mf.open(dataDir + "lineitem.tbl")) {
        std::cerr << "Gen-Q1 SF100: Cannot mmap lineitem.tbl" << std::endl; return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto lineIndex = buildLineIndex(mf);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    size_t totalRows = lineIndex.size();
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 38, totalRows);
    const uint num_tg = 1024;
    const uint bins = 6;
    printf("Gen-Q1 SF100: %zu rows, chunk size: %zu (index %.1f ms)\n", totalRows, chunkRows, indexBuildMs);

    auto* pso = findPSO(cq, "Q1_reduce");
    if (!pso) return;

    struct Q1Slot {
        MTL::Buffer* shipdate; MTL::Buffer* returnflag; MTL::Buffer* linestatus;
        MTL::Buffer* quantity; MTL::Buffer* extprice; MTL::Buffer* discount; MTL::Buffer* tax;
    };
    Q1Slot slots[2];
    for (int s = 0; s < 2; s++) {
        slots[s].shipdate   = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].returnflag = device->newBuffer(chunkRows * sizeof(char), MTL::ResourceStorageModeShared);
        slots[s].linestatus = device->newBuffer(chunkRows * sizeof(char), MTL::ResourceStorageModeShared);
        slots[s].quantity   = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].extprice   = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].discount   = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].tax        = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
    }

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

    long g_qty[6]={}, g_base[6]={}, g_disc[6]={}, g_charge[6]={};
    uint32_t g_discbp[6]={}, g_count[6]={};

    auto reconstruct = [](uint32_t lo, uint32_t hi) -> long {
        return (long)(((uint64_t)hi << 32) | (uint64_t)lo);
    };

    auto timing = chunkedStreamLoop(
        cmdQueue, slots, 2, totalRows, chunkRows,
        [&](Q1Slot& slot, size_t startRow, size_t rowCount) {
            parseDateColumnChunk(mf, lineIndex, startRow, rowCount, 10, (int*)slot.shipdate->contents());
            parseCharColumnChunk(mf, lineIndex, startRow, rowCount, 8, (char*)slot.returnflag->contents());
            parseCharColumnChunk(mf, lineIndex, startRow, rowCount, 9, (char*)slot.linestatus->contents());
            parseFloatColumnChunk(mf, lineIndex, startRow, rowCount, 4, (float*)slot.quantity->contents());
            parseFloatColumnChunk(mf, lineIndex, startRow, rowCount, 5, (float*)slot.extprice->contents());
            parseFloatColumnChunk(mf, lineIndex, startRow, rowCount, 6, (float*)slot.discount->contents());
            parseFloatColumnChunk(mf, lineIndex, startRow, rowCount, 7, (float*)slot.tax->contents());
        },
        [&](Q1Slot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
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

            auto* enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(pso);
            enc->setBuffer(slot.shipdate, 0, 0);
            enc->setBuffer(slot.returnflag, 0, 1);
            enc->setBuffer(slot.linestatus, 0, 2);
            enc->setBuffer(slot.quantity, 0, 3);
            enc->setBuffer(slot.extprice, 0, 4);
            enc->setBuffer(slot.discount, 0, 5);
            enc->setBuffer(slot.tax, 0, 6);
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
            enc->setBytes(&chunkSize, sizeof(chunkSize), 17);
            enc->setBytes(&cutoffDate, sizeof(cutoffDate), 18);
            NS::UInteger tgSize = pso->maxTotalThreadsPerThreadgroup();
            if (tgSize > 1024) tgSize = 1024;
            enc->dispatchThreadgroups(MTL::Size::Make(num_tg, 1, 1), MTL::Size::Make(tgSize, 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {
            auto* qty_lo = (uint32_t*)out_qty_lo->contents();
            auto* qty_hi = (uint32_t*)out_qty_hi->contents();
            auto* base_lo = (uint32_t*)out_base_lo->contents();
            auto* base_hi = (uint32_t*)out_base_hi->contents();
            auto* disc_lo = (uint32_t*)out_disc_lo->contents();
            auto* disc_hi = (uint32_t*)out_disc_hi->contents();
            auto* charge_lo = (uint32_t*)out_charge_lo->contents();
            auto* charge_hi = (uint32_t*)out_charge_hi->contents();
            auto* discbp = (uint32_t*)out_discount_bp->contents();
            auto* counts = (uint32_t*)out_count->contents();
            for (int b = 0; b < 6; b++) {
                g_qty[b] += reconstruct(qty_lo[b], qty_hi[b]);
                g_base[b] += reconstruct(base_lo[b], base_hi[b]);
                g_disc[b] += reconstruct(disc_lo[b], disc_hi[b]);
                g_charge[b] += reconstruct(charge_lo[b], charge_hi[b]);
                g_discbp[b] += discbp[b];
                g_count[b] += counts[b];
            }
        }
    );

    // Post-process
    auto postStart = std::chrono::high_resolution_clock::now();
    struct Q1R { double sum_qty, sum_base, sum_disc, sum_charge, avg_qty, avg_price, avg_disc; uint cnt; };
    char rfChars[] = {'A','A','N','N','R','R'};
    char lsChars[] = {'F','O','F','O','F','O'};

    printf("\n+----------+----------+------------+----------------+----------------+----------------+------------+------------+------------+----------+\n");
    printf("| l_return | l_linest |    sum_qty | sum_base_price | sum_disc_price |     sum_charge |    avg_qty |  avg_price |   avg_disc | count    |\n");
    printf("+----------+----------+------------+----------------+----------------+----------------+------------+------------+------------+----------+\n");
    for (int b = 0; b < 6; b++) {
        if (g_count[b] == 0) continue;
        Q1R r;
        r.sum_qty = (double)g_qty[b] / 100.0;
        r.sum_base = (double)g_base[b] / 100.0;
        r.sum_disc = (double)g_disc[b] / 100.0;
        r.sum_charge = (double)g_charge[b] / 100.0;
        r.cnt = g_count[b];
        r.avg_qty = r.sum_qty / (double)r.cnt;
        r.avg_price = r.sum_base / (double)r.cnt;
        r.avg_disc = ((double)g_discbp[b] / 100.0) / (double)r.cnt;
        printf("| %8c | %8c | %10.2f | %14.2f | %14.2f | %14.2f | %10.2f | %10.2f | %10.2f | %8u |\n",
               rfChars[b], lsChars[b], r.sum_qty, r.sum_base, r.sum_disc, r.sum_charge,
               r.avg_qty, r.avg_price, r.avg_disc, r.cnt);
    }
    printf("+----------+----------+------------+----------------+----------------+----------------+------------+------------+------------+----------+\n");
    auto postEnd = std::chrono::high_resolution_clock::now();
    double cpuPostMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nGen-Q1 SF100 | %zu chunks | %zu rows\n", timing.chunkCount, totalRows);
    printTimingSummary(indexBuildMs + timing.parseMs, timing.gpuMs, timing.postMs + cpuPostMs);

    for (int s = 0; s < 2; s++)
        releaseAll(slots[s].shipdate, slots[s].returnflag, slots[s].linestatus,
                   slots[s].quantity, slots[s].extprice, slots[s].discount, slots[s].tax);
    releaseAll(out_qty_lo, out_qty_hi, out_base_lo, out_base_hi,
               out_disc_lo, out_disc_hi, out_charge_lo, out_charge_hi,
               out_discount_bp, out_count);
}

void executeQ3SF100(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                    const RuntimeCompiler::CompiledQuery& cq,
                    const std::string& dataDir) {
    MappedFile custFile, ordFile, liFile;
    if (!custFile.open(dataDir + "customer.tbl") ||
        !ordFile.open(dataDir + "orders.tbl") ||
        !liFile.open(dataDir + "lineitem.tbl")) {
        std::cerr << "Gen-Q3 SF100: Cannot open TBL files" << std::endl; return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto custIdx = buildLineIndex(custFile);
    auto ordIdx = buildLineIndex(ordFile);
    auto liIdx = buildLineIndex(liFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    size_t custRows = custIdx.size(), ordRows = ordIdx.size(), liRows = liIdx.size();
    printf("Gen-Q3 SF100: customer=%zu, orders=%zu, lineitem=%zu (index %.1f ms)\n",
           custRows, ordRows, liRows, indexBuildMs);

    // Load dimension tables fully
    auto bpT0 = std::chrono::high_resolution_clock::now();
    std::vector<int> c_custkey(custRows);
    std::vector<char> c_mktsegment(custRows);
    parseIntColumnChunk(custFile, custIdx, 0, custRows, 0, c_custkey.data());
    parseCharColumnChunk(custFile, custIdx, 0, custRows, 6, c_mktsegment.data());

    std::vector<int> o_orderkey(ordRows), o_custkey(ordRows), o_orderdate(ordRows), o_shippriority(ordRows);
    parseIntColumnChunk(ordFile, ordIdx, 0, ordRows, 0, o_orderkey.data());
    parseIntColumnChunk(ordFile, ordIdx, 0, ordRows, 1, o_custkey.data());
    parseDateColumnChunk(ordFile, ordIdx, 0, ordRows, 4, o_orderdate.data());
    parseIntColumnChunk(ordFile, ordIdx, 0, ordRows, 7, o_shippriority.data());
    auto bpT1 = std::chrono::high_resolution_clock::now();
    double buildParseMs = std::chrono::duration<double, std::milli>(bpT1 - bpT0).count();

    auto* bmPSO = findPSO(cq, "Q3_bitmap_build");
    auto* mapPSO = findPSO(cq, "Q3_map_build");
    auto* probePSO = findPSO(cq, "Q3_probe_agg");
    auto* compactPSO = findPSO(cq, "Q3_compact");
    if (!bmPSO || !mapPSO || !probePSO || !compactPSO) return;

    const uint customer_size = (uint)custRows;
    const uint orders_size = (uint)ordRows;
    const int cutoff_date = 19950315;
    const uint num_tg = 2048;

    int max_custkey = 0;
    for (int k : c_custkey) max_custkey = std::max(max_custkey, k);
    auto* custBmBuf = createBitmapBuffer(device, max_custkey);
    auto* custKeyBuf = device->newBuffer(c_custkey.data(), customer_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* custMktBuf = device->newBuffer(c_mktsegment.data(), customer_size * sizeof(char), MTL::ResourceStorageModeShared);

    int max_orderkey = 0;
    for (int k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    const uint orders_map_size = max_orderkey + 1;

    size_t mapBytes = (size_t)orders_map_size * sizeof(int);
    size_t maxMem = device->recommendedMaxWorkingSetSize();
    if (mapBytes > maxMem / 3) {
        std::cerr << "Gen-Q3 SF100: orders direct map too large ("
                  << mapBytes / (1024*1024) << " MB), need HT fallback" << std::endl;
        releaseAll(custBmBuf, custKeyBuf, custMktBuf);
        return;
    }

    auto* ordersMapBuf = device->newBuffer(orders_map_size * sizeof(int), MTL::ResourceStorageModeShared);
    memset(ordersMapBuf->contents(), 0xFF, orders_map_size * sizeof(int));

    auto* ordKeyBuf = device->newBuffer(o_orderkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* ordCustBuf = device->newBuffer(o_custkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* ordDateBuf = device->newBuffer(o_orderdate.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    auto* ordPrioBuf = device->newBuffer(o_shippriority.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);

    // Build phase: customer bitmap + orders map
    double buildGpuMs = 0;
    {
        auto* cmdBuf = cmdQueue->commandBuffer();
        auto* enc = cmdBuf->computeCommandEncoder();

        enc->setComputePipelineState(bmPSO);
        enc->setBuffer(custKeyBuf, 0, 0);
        enc->setBuffer(custMktBuf, 0, 1);
        enc->setBuffer(custBmBuf, 0, 2);
        enc->setBytes(&customer_size, sizeof(customer_size), 3);
        enc->dispatchThreads(MTL::Size::Make(customer_size, 1, 1),
                            MTL::Size::Make(std::min((NS::UInteger)customer_size, (NS::UInteger)1024), 1, 1));

        enc->memoryBarrier(MTL::BarrierScopeBuffers);

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

        enc->endEncoding();
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();
        buildGpuMs = (cmdBuf->GPUEndTime() - cmdBuf->GPUStartTime()) * 1000.0;
        printf("Build phase done (GPU: %.2f ms)\n", buildGpuMs);
    }

    releaseAll(custKeyBuf, custMktBuf, ordKeyBuf);

    // Final HT: smaller sizing for SF-100
    const uint final_ht_size = nextPow2(std::max((uint)(ordRows / 64), (uint)(1 << 20)));
    auto* finalHTBuf = createFilledBuffer(device, (size_t)final_ht_size * 16, 0);
    auto* denseBuf = device->newBuffer((size_t)final_ht_size * 16, MTL::ResourceStorageModeShared);
    auto* countBuf = createFilledBuffer(device, sizeof(uint), 0);

    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 20, liRows);
    printf("Lineitem chunk size: %zu rows\n", chunkRows);

    struct Q3Slot { MTL::Buffer* orderkey; MTL::Buffer* shipdate; MTL::Buffer* extprice; MTL::Buffer* discount; };
    Q3Slot slots[2];
    for (int s = 0; s < 2; s++) {
        slots[s].orderkey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].shipdate = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].extprice = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].discount = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
    }

    auto timing = chunkedStreamLoop(
        cmdQueue, slots, 2, liRows, chunkRows,
        [&](Q3Slot& slot, size_t startRow, size_t rowCount) {
            parseIntColumnChunk(liFile, liIdx, startRow, rowCount, 0, (int*)slot.orderkey->contents());
            parseDateColumnChunk(liFile, liIdx, startRow, rowCount, 10, (int*)slot.shipdate->contents());
            parseFloatColumnChunk(liFile, liIdx, startRow, rowCount, 5, (float*)slot.extprice->contents());
            parseFloatColumnChunk(liFile, liIdx, startRow, rowCount, 6, (float*)slot.discount->contents());
        },
        [&](Q3Slot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto* enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(probePSO);
            enc->setBuffer(slot.orderkey, 0, 0);
            enc->setBuffer(slot.shipdate, 0, 1);
            enc->setBuffer(slot.extprice, 0, 2);
            enc->setBuffer(slot.discount, 0, 3);
            enc->setBuffer(ordersMapBuf, 0, 4);
            enc->setBuffer(ordCustBuf, 0, 5);
            enc->setBuffer(ordDateBuf, 0, 6);
            enc->setBuffer(ordPrioBuf, 0, 7);
            enc->setBuffer(finalHTBuf, 0, 8);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 9);
            enc->setBytes(&cutoff_date, sizeof(cutoff_date), 10);
            enc->setBytes(&final_ht_size, sizeof(final_ht_size), 11);
            NS::UInteger tgSize = probePSO->maxTotalThreadsPerThreadgroup();
            if (tgSize > 1024) tgSize = 1024;
            enc->dispatchThreadgroups(MTL::Size::Make(num_tg, 1, 1), MTL::Size::Make(tgSize, 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {}
    );

    // Compact final HT
    *(uint*)countBuf->contents() = 0;
    {
        auto* cb = cmdQueue->commandBuffer();
        auto* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(compactPSO);
        enc->setBuffer(finalHTBuf, 0, 0);
        enc->setBuffer(denseBuf, 0, 1);
        enc->setBuffer(countBuf, 0, 2);
        enc->setBytes(&final_ht_size, sizeof(final_ht_size), 3);
        enc->dispatchThreads(MTL::Size::Make(final_ht_size, 1, 1),
                            MTL::Size::Make(std::min((NS::UInteger)final_ht_size, (NS::UInteger)1024), 1, 1));
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
    }

    auto postStart = std::chrono::high_resolution_clock::now();
    uint resultCount = *(uint*)countBuf->contents();
    auto* dense = (Q3Aggregates_CPU*)denseBuf->contents();
    sortAndPrintQ3(dense, resultCount);
    auto postEnd = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nGen-Q3 SF100 | %zu chunks | %zu lineitem\n", timing.chunkCount, liRows);
    printTimingSummary(indexBuildMs + buildParseMs + timing.parseMs, buildGpuMs + timing.gpuMs, postMs);

    releaseAll(custBmBuf, ordersMapBuf, ordCustBuf, ordDateBuf, ordPrioBuf,
               finalHTBuf, denseBuf, countBuf);
    for (int s = 0; s < 2; s++)
        releaseAll(slots[s].orderkey, slots[s].shipdate, slots[s].extprice, slots[s].discount);
}

void executeQ13SF100(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                     const RuntimeCompiler::CompiledQuery& cq,
                     const std::string& dataDir) {
    MappedFile custFile, ordFile;
    if (!custFile.open(dataDir + "customer.tbl") || !ordFile.open(dataDir + "orders.tbl")) {
        std::cerr << "Gen-Q13 SF100: Cannot open TBL files" << std::endl; return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto custIdx = buildLineIndex(custFile);
    auto ordIdx = buildLineIndex(ordFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    size_t custRows = custIdx.size(), ordRows = ordIdx.size();
    printf("Gen-Q13 SF100: customer=%zu, orders=%zu (index %.1f ms)\n", custRows, ordRows, indexBuildMs);

    auto buildT0 = std::chrono::high_resolution_clock::now();
    std::vector<int> c_custkey(custRows);
    parseIntColumnChunk(custFile, custIdx, 0, custRows, 0, c_custkey.data());
    int max_custkey = 0;
    for (int k : c_custkey) max_custkey = std::max(max_custkey, k);
    auto buildT1 = std::chrono::high_resolution_clock::now();
    double buildMs = std::chrono::duration<double, std::milli>(buildT1 - buildT0).count();

    auto* countPSO = findPSO(cq, "Q13_str_match_count");
    auto* histPSO = findPSO(cq, "Q13_histogram");
    if (!countPSO || !histPSO) return;

    auto* countsBuf = createFilledBuffer(device, (max_custkey + 1) * sizeof(uint), 0);

    const uint comment_width = 79;
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, sizeof(int) + comment_width, ordRows);
    printf("Chunk size: %zu rows\n", chunkRows);

    struct Q13Slot { MTL::Buffer* custKey; MTL::Buffer* comment; };
    Q13Slot slots[2];
    for (int s = 0; s < 2; s++) {
        slots[s].custKey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].comment = device->newBuffer(chunkRows * comment_width, MTL::ResourceStorageModeShared);
    }

    auto timing = chunkedStreamLoop(
        cmdQueue, slots, 2, ordRows, chunkRows,
        [&](Q13Slot& slot, size_t startRow, size_t rowCount) {
            parseIntColumnChunk(ordFile, ordIdx, startRow, rowCount, 1, (int*)slot.custKey->contents());
            parseCharColumnChunkFixed(ordFile, ordIdx, startRow, rowCount, 8, comment_width, (char*)slot.comment->contents());
        },
        [&](Q13Slot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto* enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(countPSO);
            enc->setBuffer(slot.custKey, 0, 0);
            enc->setBuffer(slot.comment, 0, 1);
            enc->setBuffer(countsBuf, 0, 2);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 3);
            enc->setBytes(&comment_width, sizeof(comment_width), 4);
            enc->dispatchThreads(MTL::Size::Make(chunkSize, 1, 1),
                                MTL::Size::Make(std::min((NS::UInteger)chunkSize, (NS::UInteger)1024), 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {}
    );

    // Dispatch histogram once
    uint maxOrders = 100;
    auto* histBuf = createFilledBuffer(device, (maxOrders + 1) * sizeof(uint), 0);
    uint maxCK = (uint)max_custkey;
    {
        auto* cb = cmdQueue->commandBuffer();
        auto* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(histPSO);
        enc->setBuffer(countsBuf, 0, 0);
        enc->setBuffer(histBuf, 0, 1);
        enc->setBytes(&maxCK, sizeof(maxCK), 2);
        uint histThreads = max_custkey + 1;
        enc->dispatchThreads(MTL::Size::Make(histThreads, 1, 1),
                            MTL::Size::Make(std::min((NS::UInteger)histThreads, (NS::UInteger)1024), 1, 1));
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
    }

    auto postStart = std::chrono::high_resolution_clock::now();
    auto* hist = (uint*)histBuf->contents();
    uint customersWithOrders = 0;
    for (uint i = 1; i <= maxOrders; i++) customersWithOrders += hist[i];
    hist[0] = (uint)custRows - customersWithOrders;

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

    printf("\nGen-Q13 SF100 | %zu chunks | %zu orders, %zu customers\n", timing.chunkCount, ordRows, custRows);
    printTimingSummary(indexBuildMs + buildMs + timing.parseMs, timing.gpuMs, timing.postMs + postMs);

    for (int s = 0; s < 2; s++)
        releaseAll(slots[s].custKey, slots[s].comment);
    releaseAll(countsBuf, histBuf);
}

void executeQ14SF100(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                     const RuntimeCompiler::CompiledQuery& cq,
                     const std::string& dataDir) {
    MappedFile partFile, liFile;
    if (!partFile.open(dataDir + "part.tbl") || !liFile.open(dataDir + "lineitem.tbl")) {
        std::cerr << "Gen-Q14 SF100: Cannot open TBL files" << std::endl; return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto partIndex = buildLineIndex(partFile);
    auto liIndex = buildLineIndex(liFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    size_t partRows = partIndex.size(), liRows = liIndex.size();
    printf("Gen-Q14 SF100: part=%zu, lineitem=%zu (index %.1f ms)\n", partRows, liRows, indexBuildMs);

    auto buildT0 = std::chrono::high_resolution_clock::now();
    std::vector<int> p_partkey(partRows);
    parseIntColumnChunk(partFile, partIndex, 0, partRows, 0, p_partkey.data());
    const uint type_stride = 25;
    std::vector<char> p_type(partRows * type_stride);
    parseCharColumnChunkFixed(partFile, partIndex, 0, partRows, 4, type_stride, p_type.data());

    auto promoBm = buildCPUBitmap(p_partkey, [&](size_t i) {
        const char* t = p_type.data() + i * type_stride;
        return t[0]=='P' && t[1]=='R' && t[2]=='O' && t[3]=='M' && t[4]=='O';
    });
    auto* promoBmBuf = uploadBitmap(device, promoBm);
    auto buildT1 = std::chrono::high_resolution_clock::now();
    double buildMs = std::chrono::duration<double, std::milli>(buildT1 - buildT0).count();

    auto* s1pso = findPSO(cq, "Q14_reduce");
    auto* s2pso = findPSO(cq, "Q14_reduce_final");
    if (!s1pso || !s2pso) return;

    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 16, liRows);
    const uint numTG = 2048;
    printf("Chunk size: %zu rows\n", chunkRows);

    struct Q14Slot { MTL::Buffer* partkey; MTL::Buffer* shipdate; MTL::Buffer* extprice; MTL::Buffer* discount; };
    Q14Slot slots[2];
    for (int s = 0; s < 2; s++) {
        slots[s].partkey  = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].shipdate = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].extprice = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].discount = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
    }

    auto* partialPromoBuf = device->newBuffer(numTG * sizeof(float), MTL::ResourceStorageModeShared);
    auto* partialTotalBuf = device->newBuffer(numTG * sizeof(float), MTL::ResourceStorageModeShared);
    auto* finalBuf = device->newBuffer(2 * sizeof(float), MTL::ResourceStorageModeShared);

    int start_date = 19950901, end_date = 19951001;
    double globalPromo = 0.0, globalTotal = 0.0;

    auto timing = chunkedStreamLoop(
        cmdQueue, slots, 2, liRows, chunkRows,
        [&](Q14Slot& slot, size_t startRow, size_t rowCount) {
            parseIntColumnChunk(liFile, liIndex, startRow, rowCount, 1, (int*)slot.partkey->contents());
            parseDateColumnChunk(liFile, liIndex, startRow, rowCount, 10, (int*)slot.shipdate->contents());
            parseFloatColumnChunk(liFile, liIndex, startRow, rowCount, 5, (float*)slot.extprice->contents());
            parseFloatColumnChunk(liFile, liIndex, startRow, rowCount, 6, (float*)slot.discount->contents());
        },
        [&](Q14Slot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto* enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(s1pso);
            enc->setBuffer(slot.partkey, 0, 0);
            enc->setBuffer(slot.shipdate, 0, 1);
            enc->setBuffer(slot.extprice, 0, 2);
            enc->setBuffer(slot.discount, 0, 3);
            enc->setBuffer(promoBmBuf, 0, 4);
            enc->setBuffer(partialPromoBuf, 0, 5);
            enc->setBuffer(partialTotalBuf, 0, 6);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 7);
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
        },
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {
            float* res = (float*)finalBuf->contents();
            globalPromo += res[0];
            globalTotal += res[1];
        }
    );

    double promo_pct = (globalTotal > 0.0) ? 100.0 * globalPromo / globalTotal : 0.0;
    printf("\nTPC-H Gen-Q14 Result:\nPromo Revenue: %.2f%%\n", promo_pct);
    printf("\nGen-Q14 SF100 | %zu chunks | %zu rows\n", timing.chunkCount, liRows);
    printTimingSummary(indexBuildMs + buildMs + timing.parseMs, timing.gpuMs, timing.postMs);

    releaseAll(promoBmBuf, partialPromoBuf, partialTotalBuf, finalBuf);
    for (int s = 0; s < 2; s++)
        releaseAll(slots[s].partkey, slots[s].shipdate, slots[s].extprice, slots[s].discount);
}

void executeQ18SF100(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                     const RuntimeCompiler::CompiledQuery& cq,
                     const std::string& dataDir) {
    MappedFile custFile, ordFile, liFile;
    if (!custFile.open(dataDir + "customer.tbl") ||
        !ordFile.open(dataDir + "orders.tbl") ||
        !liFile.open(dataDir + "lineitem.tbl")) {
        std::cerr << "Gen-Q18 SF100: Cannot open TBL files" << std::endl; return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto custIdx = buildLineIndex(custFile);
    auto ordIdx = buildLineIndex(ordFile);
    auto liIdx = buildLineIndex(liFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    size_t custRows = custIdx.size(), ordRows = ordIdx.size(), liRows = liIdx.size();
    printf("Gen-Q18 SF100: customer=%zu, orders=%zu, lineitem=%zu (index %.1f ms)\n",
           custRows, ordRows, liRows, indexBuildMs);

    // Load dimension tables fully
    auto bpT0 = std::chrono::high_resolution_clock::now();
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
    double buildParseMs = std::chrono::duration<double, std::milli>(bpT1 - bpT0).count();

    int max_orderkey = 0;
    for (size_t i = 0; i < ordRows; i++) max_orderkey = std::max(max_orderkey, o_orderkey[i]);
    uint qty_map_size = max_orderkey + 1;

    auto* aggPSO = findPSO(cq, "Q18_atomic_agg");
    if (!aggPSO) return;

    auto* pQtyMapBuf = device->newBuffer((size_t)qty_map_size * sizeof(float), MTL::ResourceStorageModeShared);
    memset(pQtyMapBuf->contents(), 0, (size_t)qty_map_size * sizeof(float));

    // Stream lineitem for qty aggregation
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 8, liRows);
    printf("Lineitem chunk size: %zu rows\n", chunkRows);

    struct Q18Slot { MTL::Buffer* orderkey; MTL::Buffer* quantity; };
    Q18Slot slots[2];
    for (int s = 0; s < 2; s++) {
        slots[s].orderkey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].quantity = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
    }

    auto timing = chunkedStreamLoop(
        cmdQueue, slots, 2, liRows, chunkRows,
        [&](Q18Slot& slot, size_t startRow, size_t rowCount) {
            parseIntColumnChunk(liFile, liIdx, startRow, rowCount, 0, (int*)slot.orderkey->contents());
            parseFloatColumnChunk(liFile, liIdx, startRow, rowCount, 4, (float*)slot.quantity->contents());
        },
        [&](Q18Slot& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf) {
            auto* enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(aggPSO);
            enc->setBuffer(slot.orderkey, 0, 0);
            enc->setBuffer(slot.quantity, 0, 1);
            enc->setBuffer(pQtyMapBuf, 0, 2);
            enc->setBytes(&chunkSize, sizeof(chunkSize), 3);
            enc->dispatchThreadgroups(MTL::Size::Make(2048, 1, 1), MTL::Size::Make(1024, 1, 1));
            enc->endEncoding();
            cmdBuf->commit();
        },
        [&]([[maybe_unused]] uint chunkSize, [[maybe_unused]] size_t chunkNum) {}
    );

    // CPU post: filter orders by qty > 300, join customer names
    auto postStart = std::chrono::high_resolution_clock::now();
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

    printf("\nTPC-H Gen-Q18 Results (Top 10 of LIMIT 100):\n");
    printf("+------------------+----------+----------+------------+-------------+---------+\n");
    printf("| c_name           | c_custkey| o_orderkey| o_orderdate| o_totalprice| sum_qty |\n");
    printf("+------------------+----------+----------+------------+-------------+---------+\n");
    size_t show = std::min((size_t)10, topK);
    for (size_t i = 0; i < show; i++) {
        printf("| %-16s | %8d | %9d | %10d | %11.2f | %7.2f |\n",
               results[i].c_name.c_str(), results[i].c_custkey, results[i].o_orderkey,
               results[i].o_orderdate, results[i].o_totalprice, results[i].sum_qty);
    }
    printf("+------------------+----------+----------+------------+-------------+---------+\n");
    printf("Total qualifying orders: %zu\n", results.size());
    auto postEnd = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    printf("\nGen-Q18 SF100 | %zu chunks | %zu lineitem\n", timing.chunkCount, liRows);
    printTimingSummary(indexBuildMs + buildParseMs + timing.parseMs, timing.gpuMs, postMs);

    releaseAll(pQtyMapBuf);
    for (int s = 0; s < 2; s++) releaseAll(slots[s].orderkey, slots[s].quantity);
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
    // SF-100 chunked streaming executors
    if (g_sf100_mode) {
        if (plan.name == "Q1")       { executeQ1SF100(device, cmdQueue, compiled, dataDir); return; }
        else if (plan.name == "Q3")  { executeQ3SF100(device, cmdQueue, compiled, dataDir); return; }
        else if (plan.name == "Q6")  { executeQ6SF100(device, cmdQueue, compiled, dataDir); return; }
        else if (plan.name == "Q13") { executeQ13SF100(device, cmdQueue, compiled, dataDir); return; }
        else if (plan.name == "Q14") { executeQ14SF100(device, cmdQueue, compiled, dataDir); return; }
        else if (plan.name == "Q18") { executeQ18SF100(device, cmdQueue, compiled, dataDir); return; }
        // Fall through for unimplemented SF-100 queries
        printf("WARNING: No SF-100 codegen executor for %s, using standard path\n", plan.name.c_str());
    }

    if (plan.name == "Q1")  executeQ1(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q6")  executeQ6(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q3")  executeQ3(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q14") executeQ14(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q13") executeQ13(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q4")  executeQ4(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q12") executeQ12(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q19") executeQ19(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q15") executeQ15(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q11") executeQ11(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q10") executeQ10(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q5")  executeQ5(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q7")  executeQ7(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q8")  executeQ8(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q17") executeQ17(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q22") executeQ22(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q2")  executeQ2(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q18") executeQ18(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q9")  executeQ9(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q16") executeQ16(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q20") executeQ20(device, cmdQueue, compiled, dataDir);
    else if (plan.name == "Q21") executeQ21(device, cmdQueue, compiled, dataDir);
    else throw std::runtime_error("No executor for plan: " + plan.name);
}

} // namespace codegen
