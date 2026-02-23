#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "Metal/Metal.hpp"
#include "Foundation/Foundation.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <functional>

// mmap for SF100 chunked streaming
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// Global dataset configuration
std::string g_dataset_path = "data/SF-1/"; // Default to SF-1
int g_scale_factor = 1; // 1, 10, or 100
bool g_sf100_mode = false; // true when running SF100 chunked execution

// ===================================================================
// SF100 CHUNKED EXECUTION INFRASTRUCTURE
// ===================================================================

// --- Chunk configuration ---
struct ChunkConfig {
    static constexpr size_t DEFAULT_CHUNK_ROWS = 10 * 1024 * 1024; // 10M rows
    static constexpr size_t MIN_CHUNK_ROWS = 1 * 1024 * 1024;      // 1M
    static constexpr size_t MAX_CHUNK_ROWS = 50 * 1024 * 1024;     // 50M
    static constexpr size_t NUM_BUFFERS = 2;                         // double-buffer

    static size_t adaptiveChunkSize(MTL::Device* device, size_t bytesPerRow, size_t totalRows) {
        size_t availableBytes = static_cast<size_t>(device->recommendedMaxWorkingSetSize() * 0.25);
        size_t perBufferBytes = availableBytes / NUM_BUFFERS;
        size_t maxRows = perBufferBytes / bytesPerRow;
        maxRows = std::max(maxRows, MIN_CHUNK_ROWS);
        maxRows = std::min(maxRows, MAX_CHUNK_ROWS);
        maxRows = std::min(maxRows, totalRows);
        return maxRows;
    }
};

// --- Memory-mapped TBL file for streaming ---
struct MappedFile {
    int fd = -1;
    void* data = nullptr;
    size_t size = 0;
    
    bool open(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { std::cerr << "Cannot open: " << path << std::endl; return false; }
        struct stat st;
        fstat(fd, &st);
        size = st.st_size;
        data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) { ::close(fd); fd = -1; data = nullptr; return false; }
        madvise(data, size, MADV_SEQUENTIAL);
        return true;
    }
    
    void close() {
        if (data && data != MAP_FAILED) { munmap(data, size); data = nullptr; }
        if (fd >= 0) { ::close(fd); fd = -1; }
    }
    
    ~MappedFile() { close(); }
};

// Count total lines in a mmap'd file
static size_t countLines(const MappedFile& mf) {
    size_t count = 0;
    const char* p = (const char*)mf.data;
    const char* end = p + mf.size;
    while (p < end) { if (*p++ == '\n') count++; }
    return count;
}

// Build line offset index for random access by row number
static std::vector<size_t> buildLineIndex(const MappedFile& mf) {
    std::vector<size_t> offsets;
    offsets.reserve(countLines(mf) + 1); // pre-size to avoid repeated reallocations
    offsets.push_back(0);
    const char* p = (const char*)mf.data;
    for (size_t i = 0; i < mf.size; i++) {
        if (p[i] == '\n' && i + 1 < mf.size) offsets.push_back(i + 1);
    }
    return offsets;
}

// Parse a chunk of int columns from mmap'd TBL data
static size_t parseIntColumnChunk(const MappedFile& mf, const std::vector<size_t>& lineIndex,
                                   size_t startRow, size_t rowCount, int columnIndex, int* output) {
    const char* base = (const char*)mf.data;
    const char* fileEnd = base + mf.size;
    size_t maxRow = std::min(startRow + rowCount, lineIndex.size());
    size_t parsed = 0;
    for (size_t r = startRow; r < maxRow; r++) {
        const char* line = base + lineIndex[r];
        int col = 0;
        const char* start = line;
        while (col <= columnIndex) {
            const char* end = start;
            while (end < fileEnd && *end != '|' && *end != '\n') end++;
            if (col == columnIndex) {
                output[parsed++] = atoi(start);
                break;
            }
            col++;
            if (end >= fileEnd) break;
            start = end + 1;
        }
    }
    return parsed;
}

// Parse a chunk of float columns
static size_t parseFloatColumnChunk(const MappedFile& mf, const std::vector<size_t>& lineIndex,
                                     size_t startRow, size_t rowCount, int columnIndex, float* output) {
    const char* base = (const char*)mf.data;
    const char* fileEnd = base + mf.size;
    size_t maxRow = std::min(startRow + rowCount, lineIndex.size());
    size_t parsed = 0;
    for (size_t r = startRow; r < maxRow; r++) {
        const char* line = base + lineIndex[r];
        int col = 0;
        const char* start = line;
        while (col <= columnIndex) {
            const char* end = start;
            while (end < fileEnd && *end != '|' && *end != '\n') end++;
            if (col == columnIndex) {
                output[parsed++] = strtof(start, nullptr);
                break;
            }
            col++;
            if (end >= fileEnd) break;
            start = end + 1;
        }
    }
    return parsed;
}

// Parse a chunk of date columns (YYYY-MM-DD -> YYYYMMDD int)
static size_t parseDateColumnChunk(const MappedFile& mf, const std::vector<size_t>& lineIndex,
                                    size_t startRow, size_t rowCount, int columnIndex, int* output) {
    const char* base = (const char*)mf.data;
    const char* fileEnd = base + mf.size;
    size_t maxRow = std::min(startRow + rowCount, lineIndex.size());
    size_t parsed = 0;
    for (size_t r = startRow; r < maxRow; r++) {
        const char* line = base + lineIndex[r];
        int col = 0;
        const char* start = line;
        while (col <= columnIndex) {
            const char* end = start;
            while (end < fileEnd && *end != '|' && *end != '\n') end++;
            if (col == columnIndex) {
                // Parse YYYY-MM-DD removing dashes
                int year = 0, month = 0, day = 0;
                const char* p = start;
                while (p < end && *p >= '0' && *p <= '9') { year = year * 10 + (*p - '0'); p++; }
                if (p < end) p++; // skip '-'
                while (p < end && *p >= '0' && *p <= '9') { month = month * 10 + (*p - '0'); p++; }
                if (p < end) p++; // skip '-'
                while (p < end && *p >= '0' && *p <= '9') { day = day * 10 + (*p - '0'); p++; }
                output[parsed++] = year * 10000 + month * 100 + day;
                break;
            }
            col++;
            if (end >= fileEnd) break;
            start = end + 1;
        }
    }
    return parsed;
}

// Parse a chunk of char columns (single char)
static size_t parseCharColumnChunk(const MappedFile& mf, const std::vector<size_t>& lineIndex,
                                    size_t startRow, size_t rowCount, int columnIndex, char* output) {
    const char* base = (const char*)mf.data;
    const char* fileEnd = base + mf.size;
    size_t maxRow = std::min(startRow + rowCount, lineIndex.size());
    size_t parsed = 0;
    for (size_t r = startRow; r < maxRow; r++) {
        const char* line = base + lineIndex[r];
        int col = 0;
        const char* start = line;
        while (col <= columnIndex) {
            const char* end = start;
            while (end < fileEnd && *end != '|' && *end != '\n') end++;
            if (col == columnIndex) {
                if (start < fileEnd)
                    output[parsed++] = *start;
                else
                    output[parsed++] = '\0';
                break;
            }
            col++;
            if (end >= fileEnd) break;
            start = end + 1;
        }
    }
    return parsed;
}

// Fixed-width char column parser: copies up to fixedWidth bytes per field, pads with \0
static size_t parseCharColumnChunkFixed(const MappedFile& mf, const std::vector<size_t>& lineIndex,
                                        size_t startRow, size_t rowCount, int columnIndex,
                                        int fixedWidth, char* output) {
    const char* base = (const char*)mf.data;
    const char* fileEnd = base + mf.size;
    size_t maxRow = std::min(startRow + rowCount, lineIndex.size());
    size_t parsed = 0;
    for (size_t r = startRow; r < maxRow; r++) {
        const char* line = base + lineIndex[r];
        int col = 0;
        const char* start = line;
        while (col <= columnIndex) {
            const char* end = start;
            while (end < fileEnd && *end != '|' && *end != '\n') end++;
            if (col == columnIndex) {
                int len = (int)(end - start);
                int copy = len < fixedWidth ? len : fixedWidth;
                char* dst = output + parsed * fixedWidth;
                memcpy(dst, start, copy);
                memset(dst + copy, '\0', fixedWidth - copy);
                parsed++;
                break;
            }
            col++;
            if (end >= fileEnd) break;
            start = end + 1;
        }
    }
    return parsed;
}


// ===================================================================
// SF100 CHUNKED Q1 EXECUTION
// ===================================================================
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
    NS::Error* error = nullptr;
    MTL::Function* s1Fn = library->newFunction(NS::String::string("q1_chunked_stage1", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* s1PSO = device->newComputePipelineState(s1Fn, &error);
    if (!s1PSO) { std::cerr << "Failed to create q1_chunked_stage1 PSO" << std::endl; return; }
    MTL::Function* s2Fn = library->newFunction(NS::String::string("q1_chunked_stage2", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* s2PSO = device->newComputePipelineState(s2Fn, &error);
    if (!s2PSO) { std::cerr << "Failed to create q1_chunked_stage2 PSO" << std::endl; return; }

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

    double totalGpuMs = 0.0, totalCpuParseMs = 0.0;
    double totalCpuPostMs = 0.0;
    size_t offset = 0;
    int slotIdx = 0;
    size_t chunkNum = 0;

    // --- Double-buffered pipeline: overlap CPU parse of chunk N+1 with GPU exec of chunk N ---

    // Pre-parse first chunk into slot 0
    size_t rowsThisChunk = std::min(chunkRows, totalRows);
    {
        ChunkSlot& slot = slots[0];
        auto loadStart = std::chrono::high_resolution_clock::now();
        parseDateColumnChunk(mf, lineIndex, 0, rowsThisChunk, 10, (int*)slot.shipdate->contents());
        parseCharColumnChunk(mf, lineIndex, 0, rowsThisChunk, 8,  (char*)slot.returnflag->contents());
        parseCharColumnChunk(mf, lineIndex, 0, rowsThisChunk, 9,  (char*)slot.linestatus->contents());
        parseFloatColumnChunk(mf, lineIndex, 0, rowsThisChunk, 4, (float*)slot.quantity->contents());
        parseFloatColumnChunk(mf, lineIndex, 0, rowsThisChunk, 5, (float*)slot.extprice->contents());
        parseFloatColumnChunk(mf, lineIndex, 0, rowsThisChunk, 6, (float*)slot.discount->contents());
        parseFloatColumnChunk(mf, lineIndex, 0, rowsThisChunk, 7, (float*)slot.tax->contents());
        auto loadEnd = std::chrono::high_resolution_clock::now();
        totalCpuParseMs += std::chrono::duration<double, std::milli>(loadEnd - loadStart).count();
    }

    while (offset < totalRows) {
        rowsThisChunk = std::min(chunkRows, totalRows - offset);
        ChunkSlot& slot = slots[slotIdx % NUM_SLOTS];

        // Zero partials (safe: GPU from previous iteration already waited on)
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

        // Dispatch GPU on current slot (data already parsed)
        uint chunkSize = (uint)rowsThisChunk;
        MTL::CommandBuffer* cmdBuf = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cmdBuf->computeCommandEncoder();

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

        // --- Double-buffer: parse NEXT chunk into alternate slot while GPU runs ---
        size_t nextOffset = offset + rowsThisChunk;
        if (nextOffset < totalRows) {
            size_t nextRows = std::min(chunkRows, totalRows - nextOffset);
            ChunkSlot& nextSlot = slots[(slotIdx + 1) % NUM_SLOTS];
            auto loadStart = std::chrono::high_resolution_clock::now();
            parseDateColumnChunk(mf, lineIndex, nextOffset, nextRows, 10, (int*)nextSlot.shipdate->contents());
            parseCharColumnChunk(mf, lineIndex, nextOffset, nextRows, 8,  (char*)nextSlot.returnflag->contents());
            parseCharColumnChunk(mf, lineIndex, nextOffset, nextRows, 9,  (char*)nextSlot.linestatus->contents());
            parseFloatColumnChunk(mf, lineIndex, nextOffset, nextRows, 4, (float*)nextSlot.quantity->contents());
            parseFloatColumnChunk(mf, lineIndex, nextOffset, nextRows, 5, (float*)nextSlot.extprice->contents());
            parseFloatColumnChunk(mf, lineIndex, nextOffset, nextRows, 6, (float*)nextSlot.discount->contents());
            parseFloatColumnChunk(mf, lineIndex, nextOffset, nextRows, 7, (float*)nextSlot.tax->contents());
            auto loadEnd = std::chrono::high_resolution_clock::now();
            totalCpuParseMs += std::chrono::duration<double, std::milli>(loadEnd - loadStart).count();
        }

        // Wait for GPU to finish this chunk
        cmdBuf->waitUntilCompleted();
        double gpuMs = (cmdBuf->GPUEndTime() - cmdBuf->GPUStartTime()) * 1000.0;
        totalGpuMs += gpuMs;

        // Accumulate chunk results into global accumulators
        auto postStart = std::chrono::high_resolution_clock::now();
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

        auto postEnd = std::chrono::high_resolution_clock::now();
        double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();
        totalCpuPostMs += postMs;

        chunkNum++;
        offset += rowsThisChunk;
        slotIdx++;
    }

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

    double totalCpuPostAllMs = totalCpuPostMs + cpuPostFinalMs;
    double allCpuParseMs = indexBuildMs + totalCpuParseMs;
    double totalExecMs = totalCpuPostAllMs + totalGpuMs;
    printf("\nSF100 Q1 | %zu chunks | %zu rows\n", chunkNum, totalRows);
    printf("  CPU Parsing (.tbl): %10.2f ms\n", allCpuParseMs);
    printf("  GPU Execution:      %10.2f ms\n", totalGpuMs);
    printf("  CPU Post Process:   %10.2f ms\n", totalCpuPostAllMs);
    printf("  Total Execution:    %10.2f ms  (GPU + CPU post)\n", totalExecMs);

    // Cleanup
    s1Fn->release(); s1PSO->release(); s2Fn->release(); s2PSO->release();
    for (int s = 0; s < NUM_SLOTS; s++) {
        slots[s].shipdate->release(); slots[s].returnflag->release(); slots[s].linestatus->release();
        slots[s].quantity->release(); slots[s].extprice->release(); slots[s].discount->release(); slots[s].tax->release();
    }
    p_qtyCents->release(); p_baseCents->release(); p_discCents->release(); p_chargeCents->release();
    p_discountBP->release(); p_counts->release();
    f_qtyCents->release(); f_baseCents->release(); f_discCents->release(); f_chargeCents->release();
    f_discountBP->release(); f_counts->release();
}

// ===================================================================
// SF100 CHUNKED Q6 EXECUTION
// ===================================================================
void runQ6BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q6 Benchmark (SF100 Chunked) ===" << std::endl;

    MappedFile mf;
    if (!mf.open(g_dataset_path + "lineitem.tbl")) {
        std::cerr << "Q6 SF100: Cannot mmap lineitem.tbl" << std::endl;
        return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto lineIndex = buildLineIndex(mf);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    size_t totalRows = lineIndex.size();
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 16, totalRows); // 4 cols * 4 bytes
    const uint num_tg = 2048;
    printf("Q6 SF100: %zu rows, chunk size: %zu (index %.1f ms)\n", totalRows, chunkRows, indexBuildMs);

    NS::Error* error = nullptr;
    MTL::Function* s1Fn = library->newFunction(NS::String::string("q6_chunked_stage1", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* s1PSO = device->newComputePipelineState(s1Fn, &error);
    if (!s1PSO) { std::cerr << "Failed q6_chunked_stage1 PSO" << std::endl; return; }
    MTL::Function* s2Fn = library->newFunction(NS::String::string("q6_chunked_stage2", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* s2PSO = device->newComputePipelineState(s2Fn, &error);
    if (!s2PSO) { std::cerr << "Failed q6_chunked_stage2 PSO" << std::endl; return; }

    // Double-buffered column buffers
    const int NUM_SLOTS = 2;
    struct Q6Slot { MTL::Buffer* shipdate; MTL::Buffer* discount; MTL::Buffer* quantity; MTL::Buffer* extprice; };
    Q6Slot slots[NUM_SLOTS];
    for (int s = 0; s < NUM_SLOTS; s++) {
        slots[s].shipdate = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        slots[s].discount = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].quantity = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        slots[s].extprice = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
    }

    MTL::Buffer* partials = device->newBuffer(num_tg * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* finalBuf = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    int start_date = 19940101, end_date = 19950101;
    float min_discount = 0.05f, max_discount = 0.07f, max_quantity = 24.0f;

    double globalRevenue = 0.0, totalGpuMs = 0.0, totalCpuParseMs = 0.0;
    double totalCpuPostMs = 0.0;
    size_t offset = 0, chunkNum = 0;

    // --- Double-buffered pipeline: overlap CPU parse of chunk N+1 with GPU exec of chunk N ---

    // Pre-parse first chunk into slot 0
    size_t rowsThisChunk = std::min(chunkRows, totalRows);
    {
        Q6Slot& slot = slots[0];
        auto parseStart = std::chrono::high_resolution_clock::now();
        parseDateColumnChunk(mf, lineIndex, 0, rowsThisChunk, 10, (int*)slot.shipdate->contents());
        parseFloatColumnChunk(mf, lineIndex, 0, rowsThisChunk, 6, (float*)slot.discount->contents());
        parseFloatColumnChunk(mf, lineIndex, 0, rowsThisChunk, 4, (float*)slot.quantity->contents());
        parseFloatColumnChunk(mf, lineIndex, 0, rowsThisChunk, 5, (float*)slot.extprice->contents());
        auto parseEnd = std::chrono::high_resolution_clock::now();
        totalCpuParseMs += std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();
    }

    while (offset < totalRows) {
        rowsThisChunk = std::min(chunkRows, totalRows - offset);
        Q6Slot& slot = slots[chunkNum % NUM_SLOTS];

        // Dispatch GPU on current slot (data already parsed)
        uint chunkSize = (uint)rowsThisChunk;
        MTL::CommandBuffer* cmdBuf = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cmdBuf->computeCommandEncoder();

        enc->setComputePipelineState(s1PSO);
        enc->setBuffer(slot.shipdate, 0, 0); enc->setBuffer(slot.discount, 0, 1);
        enc->setBuffer(slot.quantity, 0, 2); enc->setBuffer(slot.extprice, 0, 3);
        enc->setBuffer(partials, 0, 4);
        enc->setBytes(&chunkSize, sizeof(chunkSize), 5);
        enc->setBytes(&start_date, sizeof(start_date), 6);
        enc->setBytes(&end_date, sizeof(end_date), 7);
        enc->setBytes(&min_discount, sizeof(min_discount), 8);
        enc->setBytes(&max_discount, sizeof(max_discount), 9);
        enc->setBytes(&max_quantity, sizeof(max_quantity), 10);
        NS::UInteger tgSize = s1PSO->maxTotalThreadsPerThreadgroup();
        if (tgSize > 1024) tgSize = 1024;
        enc->dispatchThreadgroups(MTL::Size::Make(num_tg, 1, 1), MTL::Size::Make(tgSize, 1, 1));

        enc->setComputePipelineState(s2PSO);
        enc->setBuffer(partials, 0, 0); enc->setBuffer(finalBuf, 0, 1);
        enc->setBytes(&num_tg, sizeof(num_tg), 2);
        enc->dispatchThreads(MTL::Size::Make(1,1,1), MTL::Size::Make(1,1,1));
        enc->endEncoding();

        cmdBuf->commit();

        // --- Double-buffer: parse NEXT chunk into alternate slot while GPU runs ---
        size_t nextOffset = offset + rowsThisChunk;
        if (nextOffset < totalRows) {
            size_t nextRows = std::min(chunkRows, totalRows - nextOffset);
            Q6Slot& nextSlot = slots[(chunkNum + 1) % NUM_SLOTS];
            auto parseStart = std::chrono::high_resolution_clock::now();
            parseDateColumnChunk(mf, lineIndex, nextOffset, nextRows, 10, (int*)nextSlot.shipdate->contents());
            parseFloatColumnChunk(mf, lineIndex, nextOffset, nextRows, 6, (float*)nextSlot.discount->contents());
            parseFloatColumnChunk(mf, lineIndex, nextOffset, nextRows, 4, (float*)nextSlot.quantity->contents());
            parseFloatColumnChunk(mf, lineIndex, nextOffset, nextRows, 5, (float*)nextSlot.extprice->contents());
            auto parseEnd = std::chrono::high_resolution_clock::now();
            totalCpuParseMs += std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();
        }

        // Wait for GPU to finish this chunk
        cmdBuf->waitUntilCompleted();
        double gpuMs = (cmdBuf->GPUEndTime() - cmdBuf->GPUStartTime()) * 1000.0;
        totalGpuMs += gpuMs;

        auto postStart = std::chrono::high_resolution_clock::now();
        globalRevenue += *(float*)finalBuf->contents();
        auto postEnd = std::chrono::high_resolution_clock::now();
        double postMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();
        totalCpuPostMs += postMs;

        chunkNum++;
        offset += rowsThisChunk;
    }

    double allCpuParseMs = indexBuildMs + totalCpuParseMs;
    double totalExecMs = totalCpuPostMs + totalGpuMs;

    printf("TPC-H Q6 Result: Revenue = $%.2f\n", globalRevenue);
    printf("\nSF100 Q6 | %zu chunks | %zu rows\n", chunkNum, totalRows);
    printf("  CPU Parsing (.tbl): %10.2f ms\n", allCpuParseMs);
    printf("  GPU Execution:      %10.2f ms\n", totalGpuMs);
    printf("  CPU Post Process:   %10.2f ms\n", totalCpuPostMs);
    printf("  Total Execution:    %10.2f ms  (GPU + CPU post)\n", totalExecMs);

    s1Fn->release(); s1PSO->release(); s2Fn->release(); s2PSO->release();
    for (int s = 0; s < NUM_SLOTS; s++) {
        slots[s].shipdate->release(); slots[s].discount->release();
        slots[s].quantity->release(); slots[s].extprice->release();
    }
    partials->release(); finalBuf->release();
}

// ===================================================================
// SF100 Q3 / Q9 / Q13 STUBS — delegate to regular path with memory check
// ===================================================================
// For Q3/Q9/Q13 at SF100, we use the same chunked mmap approach for loading
// but fall back to the original GPU pipeline if memory allows, or print a
// warning if the dataset exceeds available GPU memory.

void runQ3BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ9BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ13BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);

// Forward declarations for original (SF1/SF10) benchmark functions
void runSelectionBenchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runAggregationBenchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runJoinBenchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ1Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ3Benchmark(MTL::Device* pDevice, MTL::CommandQueue* pCommandQueue, MTL::Library* pLibrary);
void runQ6Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ9Benchmark(MTL::Device* pDevice, MTL::CommandQueue* pCommandQueue, MTL::Library* pLibrary);
void runQ13Benchmark(MTL::Device* pDevice, MTL::CommandQueue* pCommandQueue, MTL::Library* pLibrary);

// ===================================================================
// END SF100 INFRASTRUCTURE
// ===================================================================

// --- Helper to Load Integer Column ---
std::vector<int> loadIntColumn(const std::string& filePath, int columnIndex) {
    std::vector<int> data;
    std::ifstream file(filePath);
    if (!file.is_open()) { std::cerr << "Error: Could not open file " << filePath << std::endl; return data; }
    std::string line;
    while (std::getline(file, line)) {
        std::string token; int currentCol = 0; size_t start = 0; size_t end = line.find('|');
        while (end != std::string::npos) {
            if (currentCol == columnIndex) { token = line.substr(start, end - start); data.push_back(std::stoi(token)); break; }
            start = end + 1; end = line.find('|', start); currentCol++;
        }
    }
    return data;
}

// --- Helper to Load Float Column ---
std::vector<float> loadFloatColumn(const std::string& filePath, int columnIndex) {
    std::vector<float> data;
    std::ifstream file(filePath);
    if (!file.is_open()) { std::cerr << "Error: Could not open file " << filePath << std::endl; return data; }
    std::string line;
    while (std::getline(file, line)) {
        std::string token; int currentCol = 0; size_t start = 0; size_t end = line.find('|');
        while (end != std::string::npos) {
            if (currentCol == columnIndex) { token = line.substr(start, end - start); data.push_back(std::stof(token)); break; }
            start = end + 1; end = line.find('|', start); currentCol++;
        }
    }
    return data;
}

// Helper to Load char columns
std::vector<char> loadCharColumn(const std::string& filePath, int columnIndex, int fixed_width = 0) {
    std::vector<char> data; std::ifstream file(filePath); if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << std::endl; return data;
    }
    std::string line; while (std::getline(file, line)) { std::string token; int currentCol = 0; size_t start = 0; size_t end = line.find('|');
        while (end != std::string::npos) { if (currentCol == columnIndex) { token = line.substr(start, end - start);
            if (fixed_width > 0) { for(size_t i=0; i < (size_t)fixed_width; ++i) data.push_back(i < token.length() ? token[i] : '\0'); }
            else { data.push_back(token[0]);
            }
            break;
        }
            start = end + 1; end = line.find('|', start); currentCol++;
        }
    }
    return data;
}

// Helper to Load date columns (as integers for simplicity, e.g., 19980315)
std::vector<int> loadDateColumn(const std::string& filePath, int columnIndex) {
    std::vector<int> data;
    std::ifstream file(filePath);
    if (!file.is_open()) { std::cerr << "Error: Could not open file " << filePath << std::endl; return data; }
    std::string line;
    while (std::getline(file, line)) {
        std::string token; int currentCol = 0; size_t start = 0; size_t end = line.find('|');
        while (end != std::string::npos) {
            if (currentCol == columnIndex) {
                token = line.substr(start, end - start);
                token.erase(std::remove(token.begin(), token.end(), '-'), token.end());
                data.push_back(std::stoi(token));
                break;
            }
            start = end + 1; end = line.find('|', start); currentCol++;
        }
    }
    return data;
}


// --- Selection Benchmark Test Function ---
void runSingleSelectionTest([[maybe_unused]] MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::ComputePipelineState* pipelineState,
                            MTL::Buffer* inBuffer, MTL::Buffer* resultBuffer,
                            const std::vector<int>& cpuData, int filterValue) {
    
    double gpuExecutionTime = 0.0;
    for(int iter = 0; iter < 3; ++iter) {
        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* commandEncoder = commandBuffer->computeCommandEncoder();
        commandEncoder->setComputePipelineState(pipelineState);
        commandEncoder->setBuffer(inBuffer, 0, 0);
        commandEncoder->setBuffer(resultBuffer, 0, 1);
        commandEncoder->setBytes(&filterValue, sizeof(filterValue), 2);
        
        MTL::Size gridSize = MTL::Size::Make(cpuData.size(), 1, 1);
        NS::UInteger threadGroupSize = pipelineState->maxTotalThreadsPerThreadgroup();
        if (threadGroupSize > cpuData.size()) { threadGroupSize = cpuData.size(); }
        MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);
        commandEncoder->dispatchThreads(gridSize, threadgroupSize);
        commandEncoder->endEncoding();
        
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        if (iter == 2) {
            gpuExecutionTime = commandBuffer->GPUEndTime() - commandBuffer->GPUStartTime();
        }
    }
    double dataSizeBytes = (double)cpuData.size() * sizeof(int);
    double dataSizeGB = dataSizeBytes / (1024.0 * 1024.0 * 1024.0);
    double bandwidth = dataSizeGB / gpuExecutionTime;
    
    unsigned int *resultData = (unsigned int *)resultBuffer->contents();
    unsigned int passCount = 0;
    for (size_t i = 0; i < cpuData.size(); ++i) { if (resultData[i] == 1) { passCount++; } }
    float selectivity = 100.0f * (float)passCount / (float)cpuData.size();
    
    std::cout << "--- Filter Value: < " << filterValue << " ---" << std::endl;
    std::cout << "Selectivity: " << selectivity << "% (" << passCount << " rows matched)" << std::endl;
    std::cout << "GPU execution time: " << gpuExecutionTime * 1000.0 << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl << std::endl;
}

// --- Main Function for Selection Benchmark ---
void runSelectionBenchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "--- Running Selection Benchmark ---" << std::endl;

    //Select tpch data file
    std::vector<int> cpuData = loadIntColumn(g_dataset_path + "lineitem.tbl", 1);
    if (cpuData.empty()) { return; }
    std::cout << "Loaded " << cpuData.size() << " rows for selection." << std::endl;

    NS::Error* error = nullptr;
    NS::String* functionName = NS::String::string("selection_kernel", NS::UTF8StringEncoding);
    MTL::Function* selectionFunction = library->newFunction(functionName);
    MTL::ComputePipelineState* pipelineState = device->newComputePipelineState(selectionFunction, &error);
    if (!pipelineState) { 
        std::cerr << "Failed to create selection pipeline state" << std::endl; 
        if (error) {
            std::cerr << "Error: " << error->localizedDescription()->utf8String() << std::endl;
        }
        return; 
    }

    const unsigned long dataSizeBytes = cpuData.size() * sizeof(int);
    MTL::Buffer* inBuffer = device->newBuffer(cpuData.data(), dataSizeBytes, MTL::ResourceStorageModeShared);
    MTL::Buffer* resultBuffer = device->newBuffer(cpuData.size() * sizeof(unsigned int), MTL::ResourceStorageModeShared);

    runSingleSelectionTest(device, commandQueue, pipelineState, inBuffer, resultBuffer, cpuData, 1000);
    runSingleSelectionTest(device, commandQueue, pipelineState, inBuffer, resultBuffer, cpuData, 10000);
    runSingleSelectionTest(device, commandQueue, pipelineState, inBuffer, resultBuffer, cpuData, 50000);
    
    // Cleanup
    selectionFunction->release();
    pipelineState->release();
    inBuffer->release();
    resultBuffer->release();
    functionName->release();
}


// --- Main Function for Aggregation Benchmark ---
void runAggregationBenchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "--- Running Aggregation Benchmark ---" << std::endl;

    //Select tpch data file
    std::vector<float> cpuData = loadFloatColumn(g_dataset_path + "lineitem.tbl", 4);
    if (cpuData.empty()) return;
    std::cout << "Loaded " << cpuData.size() << " rows for aggregation." << std::endl;
    const unsigned long dataSizeBytes = cpuData.size() * sizeof(float);
    uint dataSize = (uint)cpuData.size(); // The actual number of elements

    NS::Error* error = nullptr;
    NS::String* stage1FunctionName = NS::String::string("sum_kernel_stage1", NS::UTF8StringEncoding);
    MTL::Function* stage1Function = library->newFunction(stage1FunctionName);
    MTL::ComputePipelineState* stage1Pipeline = device->newComputePipelineState(stage1Function, &error);
    if (!stage1Pipeline) { 
        std::cerr << "Failed to create stage 1 pipeline state" << std::endl;
        if (error) {
            std::cerr << "Error: " << error->localizedDescription()->utf8String() << std::endl;
        }
        return; 
    }

    NS::String* stage2FunctionName = NS::String::string("sum_kernel_stage2", NS::UTF8StringEncoding);
    MTL::Function* stage2Function = library->newFunction(stage2FunctionName);
    MTL::ComputePipelineState* stage2Pipeline = device->newComputePipelineState(stage2Function, &error);
    if (!stage2Pipeline) { 
        std::cerr << "Failed to create stage 2 pipeline state" << std::endl;
        if (error) {
            std::cerr << "Error: " << error->localizedDescription()->utf8String() << std::endl;
        }
        return; 
    }

    const int numThreadgroups = 2048;
    MTL::Buffer* inBuffer = device->newBuffer(cpuData.data(), dataSizeBytes, MTL::ResourceStorageModeShared);
    MTL::Buffer* partialSumsBuffer = device->newBuffer(numThreadgroups * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* resultBuffer = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();

    // Use a single encoder for both stages to reduce encoder churn
    MTL::ComputeCommandEncoder* enc = commandBuffer->computeCommandEncoder();
    enc->setComputePipelineState(stage1Pipeline);
    enc->setBuffer(inBuffer, 0, 0);
    enc->setBuffer(partialSumsBuffer, 0, 1);
    enc->setBytes(&dataSize, sizeof(dataSize), 2);

    NS::UInteger stage1ThreadGroupSize = stage1Pipeline->maxTotalThreadsPerThreadgroup();
    if (stage1ThreadGroupSize > 1024) stage1ThreadGroupSize = 1024; // cap to match shared_memory[1024] in kernel
    MTL::Size stage1GridSize = MTL::Size::Make(numThreadgroups, 1, 1);
    MTL::Size stage1GroupSize = MTL::Size::Make(stage1ThreadGroupSize, 1, 1);
    enc->dispatchThreadgroups(stage1GridSize, stage1GroupSize);

    // Switch to stage 2 on the same encoder
    const uint numTGs = numThreadgroups;
    enc->setComputePipelineState(stage2Pipeline);
    enc->setBuffer(partialSumsBuffer, 0, 0);
    enc->setBuffer(resultBuffer, 0, 1);
    enc->setBytes(&numTGs, sizeof(numTGs), 2);
    enc->dispatchThreads(MTL::Size::Make(1, 1, 1), MTL::Size::Make(1, 1, 1));
    enc->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    double gpuExecutionTime = commandBuffer->GPUEndTime() - commandBuffer->GPUStartTime();
    double dataSizeGB = (double)dataSizeBytes / (1024.0 * 1024.0 * 1024.0);
    double bandwidth = dataSizeGB / gpuExecutionTime;

    float *finalSum = (float *)resultBuffer->contents();
    std::cout << "Final SUM(l_quantity): " << finalSum[0] << std::endl;
    std::cout << "GPU execution time: " << gpuExecutionTime * 1000.0 << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl << std::endl;
    
    // Cleanup
    stage1Function->release();
    stage1Pipeline->release();
    stage2Function->release();
    stage2Pipeline->release();
    inBuffer->release();
    partialSumsBuffer->release();
    resultBuffer->release();
    stage1FunctionName->release();
    stage2FunctionName->release();
}


// --- Main Function for Join Benchmark ---
void runJoinBenchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "--- Running Join Benchmark ---" << std::endl;
    
    // =================================================================
    // PHASE 1: BUILD
    // =================================================================
    
    // 1. Load Data for the build side (orders table)
    std::vector<int> buildKeys = loadIntColumn(g_dataset_path + "orders.tbl", 0);
    if (buildKeys.empty()) {
        std::cerr << "Error: Could not open 'orders.tbl'. Make sure it's in your " << g_dataset_path << " folder." << std::endl;
        return;
    }
    const uint buildDataSize = (uint)buildKeys.size();
    std::cout << "Loaded " << buildDataSize << " rows from orders.tbl for build phase." << std::endl;

    // 2. Setup Hash Table
    const uint hashTableSize = buildDataSize * 2;
    const unsigned long hashTableSizeBytes = hashTableSize * sizeof(int) * 2;
    std::vector<int> cpuHashTable(hashTableSize * 2, -1);

    // 3. Setup Build Kernel and Pipeline State
    NS::Error* error = nullptr;
    NS::String* buildFunctionName = NS::String::string("hash_join_build", NS::UTF8StringEncoding);
    MTL::Function* buildFunction = library->newFunction(buildFunctionName);
    MTL::ComputePipelineState* buildPipeline = device->newComputePipelineState(buildFunction, &error);
    if (!buildPipeline) { 
        std::cerr << "Failed to create build pipeline state" << std::endl;
        if (error) {
            std::cerr << "Error: " << error->localizedDescription()->utf8String() << std::endl;
        }
        return; 
    }

    // 4. Create Build Buffers
    MTL::Buffer* buildKeysBuffer = device->newBuffer(buildKeys.data(), buildKeys.size() * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* buildValuesBuffer = device->newBuffer(buildKeys.data(), buildKeys.size() * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* hashTableBuffer = device->newBuffer(cpuHashTable.data(), hashTableSizeBytes, MTL::ResourceStorageModeShared);

    // 5. Encode and Dispatch Build Kernel
    MTL::CommandBuffer* buildCommandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* buildEncoder = buildCommandBuffer->computeCommandEncoder();
    
    buildEncoder->setComputePipelineState(buildPipeline);
    buildEncoder->setBuffer(buildKeysBuffer, 0, 0);
    buildEncoder->setBuffer(buildValuesBuffer, 0, 1);
    buildEncoder->setBuffer(hashTableBuffer, 0, 2);
    buildEncoder->setBytes(&buildDataSize, sizeof(buildDataSize), 3);
    buildEncoder->setBytes(&hashTableSize, sizeof(hashTableSize), 4);

    MTL::Size buildGridSize = MTL::Size::Make(buildDataSize, 1, 1);
    NS::UInteger buildThreadGroupSize = buildPipeline->maxTotalThreadsPerThreadgroup();
    if (buildThreadGroupSize > buildDataSize) { buildThreadGroupSize = buildDataSize; }
    MTL::Size buildGroupSize = MTL::Size::Make(buildThreadGroupSize, 1, 1);

    buildEncoder->dispatchThreads(buildGridSize, buildGroupSize);
    buildEncoder->endEncoding();
    
    // 6. Execute Build Phase
    buildCommandBuffer->commit();

    // =================================================================
    // PHASE 2: PROBE
    // =================================================================
    
    // 7. Load Data for the probe side (lineitem table)
    // l_orderkey is the 1st column (index 0)
    std::vector<int> probeKeys = loadIntColumn(g_dataset_path + "lineitem.tbl", 0);
    if (probeKeys.empty()) {
        std::cerr << "Error: Could not open 'lineitem.tbl' for probe phase." << std::endl;
        return;
    }
    const uint probeDataSize = (uint)probeKeys.size();
    std::cout << "Loaded " << probeDataSize << " rows from lineitem.tbl for probe phase." << std::endl;

    // 8. Setup Probe Kernel and Pipeline State
    NS::String* probeFunctionName = NS::String::string("hash_join_probe", NS::UTF8StringEncoding);
    MTL::Function* probeFunction = library->newFunction(probeFunctionName);
    MTL::ComputePipelineState* probePipeline = device->newComputePipelineState(probeFunction, &error);
    if (!probePipeline) { 
        std::cerr << "Failed to create probe pipeline state" << std::endl;
        if (error) {
            std::cerr << "Error: " << error->localizedDescription()->utf8String() << std::endl;
        }
        return; 
    }

    // 9. Create Probe Buffers
    MTL::Buffer* probeKeysBuffer = device->newBuffer(probeKeys.data(), probeKeys.size() * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* matchCountBuffer = device->newBuffer(sizeof(unsigned int), MTL::ResourceStorageModeShared);
    // Clear the match count to zero
    memset(matchCountBuffer->contents(), 0, sizeof(unsigned int));

    // 10. Wait for build to finish, then start probe
    buildCommandBuffer->waitUntilCompleted(); // Ensure build is done before probe starts
    
    MTL::CommandBuffer* probeCommandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* probeEncoder = probeCommandBuffer->computeCommandEncoder();

    probeEncoder->setComputePipelineState(probePipeline);
    probeEncoder->setBuffer(probeKeysBuffer, 0, 0);
    probeEncoder->setBuffer(hashTableBuffer, 0, 1); // Reuse the hash table from build
    probeEncoder->setBuffer(matchCountBuffer, 0, 2);
    probeEncoder->setBytes(&probeDataSize, sizeof(probeDataSize), 3);
    probeEncoder->setBytes(&hashTableSize, sizeof(hashTableSize), 4);

    MTL::Size probeGridSize = MTL::Size::Make(probeDataSize, 1, 1);
    NS::UInteger probeThreadGroupSize = probePipeline->maxTotalThreadsPerThreadgroup();
    if (probeThreadGroupSize > probeDataSize) { probeThreadGroupSize = probeDataSize; }
    MTL::Size probeGroupSize = MTL::Size::Make(probeThreadGroupSize, 1, 1);
    probeEncoder->dispatchThreads(probeGridSize, probeGroupSize);
    probeEncoder->endEncoding();
    
    // 11. Execute Probe Phase
    probeCommandBuffer->commit();
    probeCommandBuffer->waitUntilCompleted();

    // =================================================================
    // FINAL RESULTS
    // =================================================================
    
    double buildTime = buildCommandBuffer->GPUEndTime() - buildCommandBuffer->GPUStartTime();
    double probeTime = probeCommandBuffer->GPUEndTime() - probeCommandBuffer->GPUStartTime();
    
    unsigned int* matchCount = (unsigned int*)matchCountBuffer->contents();

    std::cout << "Join complete. Found " << *matchCount << " total matches." << std::endl;
    std::cout << "Build Phase GPU time: " << buildTime * 1000.0 << " ms" << std::endl;
    std::cout << "Probe Phase GPU time: " << probeTime * 1000.0 << " ms" << std::endl;
    std::cout << "Total Join GPU time: " << (buildTime + probeTime) * 1000.0 << " ms" << std::endl << std::endl;
    
    // Cleanup
    buildFunction->release();
    buildPipeline->release();
    probeFunction->release();
    probePipeline->release();
    buildKeysBuffer->release();
    buildValuesBuffer->release();
    hashTableBuffer->release();
    probeKeysBuffer->release();
    matchCountBuffer->release();
    buildFunctionName->release();
    probeFunctionName->release();
}



// --- Main Function for TPC-H Q1 Benchmark ---
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
    NS::Error* error = nullptr;
    MTL::Function* stage1Fn = library->newFunction(NS::String::string("q1_bins_accumulate_int_stage1", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* stage1PSO = device->newComputePipelineState(stage1Fn, &error);
    if (!stage1PSO) { std::cerr << "Failed to create q1_bins_accumulate_int_stage1 PSO" << std::endl; return; }
    MTL::Function* stage2Fn = library->newFunction(NS::String::string("q1_bins_reduce_int_stage2", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* stage2PSO = device->newComputePipelineState(stage2Fn, &error);
    if (!stage2PSO) { std::cerr << "Failed to create q1_bins_reduce_int_stage2 PSO" << std::endl; return; }

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

    double q1TotalExecMs = q1_gpu_ms + q1_cpu_ms;
    printf("\nQ1 | %u rows\n", data_size);
    printf("  CPU Parsing (.tbl): %10.2f ms\n", q1CpuParseMs);
    printf("  GPU Execution:      %10.2f ms\n", q1_gpu_ms);
    printf("  CPU Post Process:   %10.2f ms\n", q1_cpu_ms);
    printf("  Total Execution:    %10.2f ms  (GPU + CPU post)\n", q1TotalExecMs);

    // Cleanup
    stage1Fn->release(); stage1PSO->release(); stage2Fn->release(); stage2PSO->release();
    shipdateBuffer->release(); flagBuffer->release(); statusBuffer->release();
    qtyBuffer->release(); priceBuffer->release(); discBuffer->release(); taxBuffer->release();
    p_sumQtyCents->release(); p_sumBaseCents->release(); p_sumDiscPriceCents->release(); p_sumChargeCents->release(); p_sumDiscountBP->release(); p_counts->release();
    f_sumQtyCents->release(); f_sumBaseCents->release(); f_sumDiscPriceCents->release(); f_sumChargeCents->release(); f_sumDiscountBP->release(); f_counts->release();
}



// C++ struct for reading GPU hash table results (matches Q3Aggregates/Q3CompactResult layout)
struct Q3Aggregates_CPU {
    int key;
    float revenue;
    unsigned int orderdate;
    unsigned int shippriority;
};


// --- Main Function for TPC-H Q3 Benchmark ---
void runQ3Benchmark(MTL::Device* pDevice, MTL::CommandQueue* pCommandQueue, MTL::Library* pLibrary) {
    std::cout << "\n--- Running TPC-H Query 3 Benchmark ---" << std::endl;

    // 1. Load data for all three tables
    auto q3ParseStart = std::chrono::high_resolution_clock::now();
    const std::string sf_path = g_dataset_path;
    auto c_custkey = loadIntColumn(sf_path + "customer.tbl", 0);
    auto c_mktsegment = loadCharColumn(sf_path + "customer.tbl", 6);

    auto o_orderkey = loadIntColumn(sf_path + "orders.tbl", 0);
    auto o_custkey = loadIntColumn(sf_path + "orders.tbl", 1);
    auto o_orderdate = loadDateColumn(sf_path + "orders.tbl", 4);
    auto o_shippriority = loadIntColumn(sf_path + "orders.tbl", 7);

    auto l_orderkey = loadIntColumn(sf_path + "lineitem.tbl", 0);
    auto l_shipdate = loadDateColumn(sf_path + "lineitem.tbl", 10);
    auto l_extendedprice = loadFloatColumn(sf_path + "lineitem.tbl", 5);
    auto l_discount = loadFloatColumn(sf_path + "lineitem.tbl", 6);
    auto q3ParseEnd = std::chrono::high_resolution_clock::now();
    double q3CpuParseMs = std::chrono::duration<double, std::milli>(q3ParseEnd - q3ParseStart).count();
    
    const uint customer_size = (uint)c_custkey.size();
    const uint orders_size = (uint)o_orderkey.size();
    const uint lineitem_size = (uint)l_orderkey.size();
    std::cout << "Loaded " << customer_size << " customers, " << orders_size << " orders, " << lineitem_size << " lineitem rows." << std::endl;

    // 2. Setup all kernels
    NS::Error* pError = nullptr;
    MTL::Function* pCustBuildFn = pLibrary->newFunction(NS::String::string("q3_build_customer_bitmap_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pCustBuildPipe = pDevice->newComputePipelineState(pCustBuildFn, &pError);

    MTL::Function* pOrdersBuildFn = pLibrary->newFunction(NS::String::string("q3_build_orders_map_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pOrdersBuildPipe = pDevice->newComputePipelineState(pOrdersBuildFn, &pError);
    
    MTL::Function* pFusedProbeAggFn = pLibrary->newFunction(NS::String::string("q3_probe_and_aggregate_direct_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pFusedProbeAggPipe = pDevice->newComputePipelineState(pFusedProbeAggFn, &pError);

    MTL::Function* pCompactFn = pLibrary->newFunction(NS::String::string("q3_compact_results_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pCompactPipe = pDevice->newComputePipelineState(pCompactFn, &pError);

    // 3. Create Buffers
    // Optimization 1: Bitmap for Customer (filter 'BUILDING')
    int max_custkey = 0;
    for(int k : c_custkey) max_custkey = std::max(max_custkey, k);
    const uint customer_bitmap_ints = (max_custkey + 31) / 32 + 1;
    MTL::Buffer* pCustomerBitmapBuffer = pDevice->newBuffer(customer_bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);
    std::memset(pCustomerBitmapBuffer->contents(), 0, customer_bitmap_ints * sizeof(uint));

    MTL::Buffer* pCustKeyBuffer = pDevice->newBuffer(c_custkey.data(), customer_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCustMktBuffer = pDevice->newBuffer(c_mktsegment.data(), customer_size * sizeof(char), MTL::ResourceStorageModeShared);

    // Optimization 2: Direct Map for Orders
    int max_orderkey = 0;
    for(int k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    const uint orders_map_size = max_orderkey + 1;
    MTL::Buffer* pOrdersMapBuffer = pDevice->newBuffer(orders_map_size * sizeof(int), MTL::ResourceStorageModeShared);
    // Initialize with -1
    std::memset(pOrdersMapBuffer->contents(), -1, orders_map_size * sizeof(int));

    MTL::Buffer* pOrdKeyBuffer = pDevice->newBuffer(o_orderkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdCustKeyBuffer = pDevice->newBuffer(o_custkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdDateBuffer = pDevice->newBuffer(o_orderdate.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdPrioBuffer = pDevice->newBuffer(o_shippriority.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    
    MTL::Buffer* pLineOrdKeyBuffer = pDevice->newBuffer(l_orderkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineShipDateBuffer = pDevice->newBuffer(l_shipdate.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLinePriceBuffer = pDevice->newBuffer(l_extendedprice.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineDiscBuffer = pDevice->newBuffer(l_discount.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);
    
    const uint num_threadgroups = 2048;
    const uint final_ht_size = orders_size * 2;
    MTL::Buffer* pFinalHTBuffer = pDevice->newBuffer(final_ht_size * sizeof(Q3Aggregates_CPU), MTL::ResourceStorageModeShared);
    std::memset(pFinalHTBuffer->contents(), 0, final_ht_size * sizeof(Q3Aggregates_CPU));

    // Dense output buffer for GPU compaction
    MTL::Buffer* pDenseBuffer = pDevice->newBuffer(final_ht_size * sizeof(Q3Aggregates_CPU), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCountBuffer = pDevice->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared);

    const int cutoff_date = 19950315;

    // 4. Dispatch full pipeline (2 warmup + 1 measured)
    double gpuExecutionTime = 0.0;
    
    for(int iter = 0; iter < 3; ++iter) {
        // Reset final HT for each iteration
        std::memset(pFinalHTBuffer->contents(), 0, final_ht_size * sizeof(Q3Aggregates_CPU));
        
        MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = pCommandBuffer->computeCommandEncoder();
        
        // Customer HT build (Bitmap)
        enc->setComputePipelineState(pCustBuildPipe);
        enc->setBuffer(pCustKeyBuffer, 0, 0);
        enc->setBuffer(pCustMktBuffer, 0, 1);
        enc->setBuffer(pCustomerBitmapBuffer, 0, 2);
        enc->setBytes(&customer_size, sizeof(customer_size), 3);
        {
            NS::UInteger threadGroupSize = pCustBuildPipe->maxTotalThreadsPerThreadgroup();
            if (threadGroupSize > 256) threadGroupSize = 256;
            MTL::Size threadgroupSize = MTL::Size(threadGroupSize, 1, 1);
            MTL::Size threadgroups = MTL::Size((customer_size + threadGroupSize - 1) / threadGroupSize, 1, 1);
            enc->dispatchThreadgroups(threadgroups, threadgroupSize);
        }

        // Orders HT build (Direct Map)
        enc->setComputePipelineState(pOrdersBuildPipe);
        enc->setBuffer(pOrdKeyBuffer, 0, 0);
        enc->setBuffer(pOrdDateBuffer, 0, 1);
        enc->setBuffer(pOrdersMapBuffer, 0, 2);
        enc->setBytes(&orders_size, sizeof(orders_size), 3);
        enc->setBytes(&cutoff_date, sizeof(cutoff_date), 4);
        {
            NS::UInteger threadGroupSize = pOrdersBuildPipe->maxTotalThreadsPerThreadgroup();
            if (threadGroupSize > 256) threadGroupSize = 256;
            MTL::Size threadgroupSize = MTL::Size(threadGroupSize, 1, 1);
            MTL::Size threadgroups = MTL::Size((orders_size + threadGroupSize - 1) / threadGroupSize, 1, 1);
            enc->dispatchThreadgroups(threadgroups, threadgroupSize);
        }

        // Fused probe + direct aggregation into final HT
        enc->setComputePipelineState(pFusedProbeAggPipe);
        enc->setBuffer(pLineOrdKeyBuffer, 0, 0);
        enc->setBuffer(pLineShipDateBuffer, 0, 1);
        enc->setBuffer(pLinePriceBuffer, 0, 2);
        enc->setBuffer(pLineDiscBuffer, 0, 3);
        enc->setBuffer(pCustomerBitmapBuffer, 0, 4);
        enc->setBuffer(pOrdersMapBuffer, 0, 5);
        enc->setBuffer(pOrdCustKeyBuffer, 0, 6);
        enc->setBuffer(pOrdDateBuffer, 0, 7);
        enc->setBuffer(pOrdPrioBuffer, 0, 8);
        enc->setBuffer(pFinalHTBuffer, 0, 9);
        enc->setBytes(&lineitem_size, sizeof(lineitem_size), 10);
        enc->setBytes(&cutoff_date, sizeof(cutoff_date), 11);
        enc->setBytes(&final_ht_size, sizeof(final_ht_size), 12);
        enc->dispatchThreadgroups(MTL::Size(num_threadgroups, 1, 1), MTL::Size(1024, 1, 1));
        enc->endEncoding();
        
        pCommandBuffer->commit();
        pCommandBuffer->waitUntilCompleted();
        
        if (iter == 2) {
             gpuExecutionTime = pCommandBuffer->GPUEndTime() - pCommandBuffer->GPUStartTime();
        }
    }
    
    // 6. GPU compaction: extract non-empty HT entries into dense buffer
    *(uint*)pCountBuffer->contents() = 0;
    MTL::CommandBuffer* compactCB = pCommandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* compactEnc = compactCB->computeCommandEncoder();
    compactEnc->setComputePipelineState(pCompactPipe);
    compactEnc->setBuffer(pFinalHTBuffer, 0, 0);
    compactEnc->setBuffer(pDenseBuffer, 0, 1);
    compactEnc->setBuffer(pCountBuffer, 0, 2);
    compactEnc->setBytes(&final_ht_size, sizeof(final_ht_size), 3);
    compactEnc->dispatchThreadgroups(MTL::Size((final_ht_size + 255) / 256, 1, 1), MTL::Size(256, 1, 1));
    compactEnc->endEncoding();
    compactCB->commit();
    compactCB->waitUntilCompleted();
    double compactGpuMs = (compactCB->GPUEndTime() - compactCB->GPUStartTime()) * 1000.0;

    uint resultCount = *(uint*)pCountBuffer->contents();
    Q3Aggregates_CPU* dense = (Q3Aggregates_CPU*)pDenseBuffer->contents();

    // CPU: partial_sort for top 10 only (O(N) instead of O(N log N))
    auto cpuMergeStart = std::chrono::high_resolution_clock::now();
    size_t topK = std::min((size_t)10, (size_t)resultCount);
    std::partial_sort(dense, dense + topK, dense + resultCount,
        [](const Q3Aggregates_CPU& a, const Q3Aggregates_CPU& b) {
            if (a.revenue != b.revenue) return a.revenue > b.revenue;
            return a.orderdate < b.orderdate;
        });
    auto cpuMergeEnd = std::chrono::high_resolution_clock::now();
    double cpuMergeMs = std::chrono::duration<double, std::milli>(cpuMergeEnd - cpuMergeStart).count();

    printf("\nTPC-H Query 3 Results (Top 10):\n");
    printf("+----------+------------+------------+--------------+\n");
    printf("| orderkey |   revenue  | orderdate  | shippriority |\n");
    printf("+----------+------------+------------+--------------+\n");
    for (size_t i = 0; i < topK; ++i) {
        printf("| %8d | $%10.2f | %10u | %12u |\n",
               dense[i].key, dense[i].revenue, dense[i].orderdate, dense[i].shippriority);
    }
    printf("+----------+------------+------------+--------------+\n");
    printf("Total results found: %u\n", resultCount);

    double q3GpuMs = gpuExecutionTime * 1000.0 + compactGpuMs;
    double q3TotalExecMs = q3GpuMs + cpuMergeMs;
    printf("\nQ3 | %u rows (lineitem)\n", lineitem_size);
    printf("  CPU Parsing (.tbl): %10.2f ms\n", q3CpuParseMs);
    printf("  GPU Execution:      %10.2f ms\n", q3GpuMs);
    printf("  CPU Post Process:   %10.2f ms\n", cpuMergeMs);
    printf("  Total Execution:    %10.2f ms  (GPU + CPU post)\n", q3TotalExecMs);
    
    //Cleanup
    pCustBuildFn->release();
    pCustBuildPipe->release();
    pOrdersBuildFn->release();
    pOrdersBuildPipe->release();
    pFusedProbeAggFn->release();
    pFusedProbeAggPipe->release();
    pCompactFn->release();
    pCompactPipe->release();

    pCustKeyBuffer->release();
    pCustMktBuffer->release();
    pCustomerBitmapBuffer->release();
    pOrdKeyBuffer->release();
    pOrdCustKeyBuffer->release();
    pOrdDateBuffer->release();
    pOrdPrioBuffer->release();
    pOrdersMapBuffer->release();
    pLineOrdKeyBuffer->release();
    pLineShipDateBuffer->release();
    pLinePriceBuffer->release();
    pLineDiscBuffer->release();
    pFinalHTBuffer->release();
    pDenseBuffer->release();
    pCountBuffer->release();
}


// --- Main Function for TPC-H Query 6 Benchmark ---
void runQ6Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "--- Running TPC-H Query 6 Benchmark ---" << std::endl;
    
    // Load required columns from lineitem table
    auto q6ParseStart = std::chrono::high_resolution_clock::now();
    std::vector<int> l_shipdate = loadDateColumn(g_dataset_path + "lineitem.tbl", 10);    // Column 10: l_shipdate
    std::vector<float> l_discount = loadFloatColumn(g_dataset_path + "lineitem.tbl", 6);  // Column 6: l_discount
    std::vector<float> l_quantity = loadFloatColumn(g_dataset_path + "lineitem.tbl", 4);  // Column 4: l_quantity
    std::vector<float> l_extendedprice = loadFloatColumn(g_dataset_path + "lineitem.tbl", 5); // Column 5: l_extendedprice
    auto q6ParseEnd = std::chrono::high_resolution_clock::now();
    double q6CpuParseMs = std::chrono::duration<double, std::milli>(q6ParseEnd - q6ParseStart).count();

    if (l_shipdate.empty() || l_discount.empty() || l_quantity.empty() || l_extendedprice.empty()) {
        std::cerr << "Error: Could not load required columns for Q6 benchmark" << std::endl;
        return;
    }

    uint dataSize = (uint)l_shipdate.size();
    std::cout << "Loaded " << dataSize << " rows for TPC-H Query 6." << std::endl;

    // Query parameters
    int start_date = 19940101;   // 1994-01-01
    int end_date = 19950101;     // 1995-01-01
    float min_discount = 0.05f;  // 5%
    float max_discount = 0.07f;  // 7%
    float max_quantity = 24.0f;

    NS::Error* error = nullptr;
    
    // Create stage 1 pipeline (filter and sum)
    NS::String* stage1FunctionName = NS::String::string("q6_filter_and_sum_stage1", NS::UTF8StringEncoding);
    MTL::Function* stage1Function = library->newFunction(stage1FunctionName);
    if (!stage1Function) {
        std::cerr << "Error: Could not find q6_filter_and_sum_stage1 function" << std::endl;
        return;
    }
    MTL::ComputePipelineState* stage1Pipeline = device->newComputePipelineState(stage1Function, &error);
    if (!stage1Pipeline) {
        std::cerr << "Failed to create Q6 stage 1 pipeline state" << std::endl;
        if (error) {
            std::cerr << "Error: " << error->localizedDescription()->utf8String() << std::endl;
        }
        return;
    }

    // Create stage 2 pipeline (final sum)
    NS::String* stage2FunctionName = NS::String::string("q6_final_sum_stage2", NS::UTF8StringEncoding);
    MTL::Function* stage2Function = library->newFunction(stage2FunctionName);
    if (!stage2Function) {
        std::cerr << "Error: Could not find q6_final_sum_stage2 function" << std::endl;
        return;
    }
    MTL::ComputePipelineState* stage2Pipeline = device->newComputePipelineState(stage2Function, &error);
    if (!stage2Pipeline) {
        std::cerr << "Failed to create Q6 stage 2 pipeline state" << std::endl;
        if (error) {
            std::cerr << "Error: " << error->localizedDescription()->utf8String() << std::endl;
        }
        return;
    }

    // Create GPU buffers
    const int numThreadgroups = 2048;
    MTL::Buffer* shipdateBuffer = device->newBuffer(l_shipdate.data(), dataSize * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* discountBuffer = device->newBuffer(l_discount.data(), dataSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* quantityBuffer = device->newBuffer(l_quantity.data(), dataSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* extendedpriceBuffer = device->newBuffer(l_extendedprice.data(), dataSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* partialRevenuesBuffer = device->newBuffer(numThreadgroups * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* finalRevenueBuffer = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    // Execute GPU kernels (2 warmup + 1 measured)
    double q6_gpu_s = 0.0;
    
    for(int iter = 0; iter < 3; ++iter) {
        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = commandBuffer->computeCommandEncoder();
        
        // Stage 1: Filter and compute partial revenue sums
        enc->setComputePipelineState(stage1Pipeline);
        enc->setBuffer(shipdateBuffer, 0, 0);
        enc->setBuffer(discountBuffer, 0, 1);
        enc->setBuffer(quantityBuffer, 0, 2);
        enc->setBuffer(extendedpriceBuffer, 0, 3);
        enc->setBuffer(partialRevenuesBuffer, 0, 4);
        enc->setBytes(&dataSize, sizeof(dataSize), 5);
        enc->setBytes(&start_date, sizeof(start_date), 6);
        enc->setBytes(&end_date, sizeof(end_date), 7);
        enc->setBytes(&min_discount, sizeof(min_discount), 8);
        enc->setBytes(&max_discount, sizeof(max_discount), 9);
        enc->setBytes(&max_quantity, sizeof(max_quantity), 10);

        NS::UInteger stage1ThreadGroupSize = stage1Pipeline->maxTotalThreadsPerThreadgroup();
        if (stage1ThreadGroupSize > 1024) stage1ThreadGroupSize = 1024; // cap to match shared_memory[1024] in kernel
        MTL::Size stage1GridSize = MTL::Size::Make(numThreadgroups, 1, 1);
        MTL::Size stage1GroupSize = MTL::Size::Make(stage1ThreadGroupSize, 1, 1);
        enc->dispatchThreadgroups(stage1GridSize, stage1GroupSize);

        // Stage 2: Final sum reduction on the same encoder
        const uint numTGs = numThreadgroups;
        enc->setComputePipelineState(stage2Pipeline);
        enc->setBuffer(partialRevenuesBuffer, 0, 0);
        enc->setBuffer(finalRevenueBuffer, 0, 1);
        enc->setBytes(&numTGs, sizeof(numTGs), 2);
        MTL::Size stage2GridSize = MTL::Size::Make(1, 1, 1);
        MTL::Size stage2GroupSize = MTL::Size::Make(1, 1, 1);
        enc->dispatchThreads(stage2GridSize, stage2GroupSize);
        enc->endEncoding();

        // Execute and measure time
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        
        if (iter == 2) {
             q6_gpu_s = commandBuffer->GPUEndTime() - commandBuffer->GPUStartTime();
        }
    }

    // CPU post (minimal) timing: fetching result
    auto q6_cpu_post_start = std::chrono::high_resolution_clock::now();

    // Get result
    float* resultData = (float*)finalRevenueBuffer->contents();
    float totalRevenue = resultData[0];

    auto q6_cpu_post_end = std::chrono::high_resolution_clock::now();
    double q6_cpu_ms = std::chrono::duration<double, std::milli>(q6_cpu_post_end - q6_cpu_post_start).count();

    std::cout << "TPC-H Query 6 Result:" << std::endl;
    std::cout << "Total Revenue: $" << std::fixed << std::setprecision(2) << totalRevenue << std::endl;

    double q6GpuMs = q6_gpu_s * 1000.0;
    double q6TotalExecMs = q6GpuMs + q6_cpu_ms;
    size_t totalDataBytes = dataSize * (sizeof(int) + 3 * sizeof(float));
    double bandwidth = (totalDataBytes / (1024.0 * 1024.0 * 1024.0)) / q6_gpu_s;
    printf("\nQ6 | %u rows\n", dataSize);
    printf("  CPU Parsing (.tbl): %10.2f ms\n", q6CpuParseMs);
    printf("  GPU Execution:      %10.2f ms\n", q6GpuMs);
    printf("  CPU Post Process:   %10.2f ms\n", q6_cpu_ms);
    printf("  Total Execution:    %10.2f ms  (GPU + CPU post)\n", q6TotalExecMs);
    printf("  Bandwidth:          %10.2f GB/s\n", bandwidth);

    // Cleanup
    stage1Function->release();
    stage1Pipeline->release();
    stage2Function->release();
    stage2Pipeline->release();
    shipdateBuffer->release();
    discountBuffer->release();
    quantityBuffer->release();
    extendedpriceBuffer->release();
    partialRevenuesBuffer->release();
    finalRevenueBuffer->release();
    stage1FunctionName->release();
    stage2FunctionName->release();
}


// C++ structs for reading final results
struct Q9Result {
    int nationkey;
    int year;
    float profit;
};

struct Q9Aggregates_CPU {
    uint key;
    float profit;
};


// --- Main Function for TPC-H Q9 Benchmark ---
void runQ9Benchmark(MTL::Device* pDevice, MTL::CommandQueue* pCommandQueue, MTL::Library* pLibrary) {
    std::cout << "\n--- Running TPC-H Query 9 Benchmark ---" << std::endl;

    const std::string sf_path = g_dataset_path;
    
    // 1. Load data for all SIX tables
    auto q9ParseStart = std::chrono::high_resolution_clock::now();
    auto p_partkey = loadIntColumn(sf_path + "part.tbl", 0);
    auto p_name = loadCharColumn(sf_path + "part.tbl", 1, 55);
    auto s_suppkey = loadIntColumn(sf_path + "supplier.tbl", 0);
    auto s_nationkey = loadIntColumn(sf_path + "supplier.tbl", 3);
    auto l_partkey = loadIntColumn(sf_path + "lineitem.tbl", 1);
    auto l_suppkey = loadIntColumn(sf_path + "lineitem.tbl", 2);
    auto l_orderkey = loadIntColumn(sf_path + "lineitem.tbl", 0);
    auto l_quantity = loadFloatColumn(sf_path + "lineitem.tbl", 4);
    auto l_extendedprice = loadFloatColumn(sf_path + "lineitem.tbl", 5);
    auto l_discount = loadFloatColumn(sf_path + "lineitem.tbl", 6);
    auto ps_partkey = loadIntColumn(sf_path + "partsupp.tbl", 0);
    auto ps_suppkey = loadIntColumn(sf_path + "partsupp.tbl", 1);
    auto ps_supplycost = loadFloatColumn(sf_path + "partsupp.tbl", 3);
    auto o_orderkey = loadIntColumn(sf_path + "orders.tbl", 0);
    auto o_orderdate = loadDateColumn(sf_path + "orders.tbl", 4);
    auto n_nationkey = loadIntColumn(sf_path + "nation.tbl", 0);
    auto n_name = loadCharColumn(sf_path + "nation.tbl", 1, 25);
    auto q9ParseEnd = std::chrono::high_resolution_clock::now();
    double q9CpuParseMs = std::chrono::duration<double, std::milli>(q9ParseEnd - q9ParseStart).count();

    // Create a map for nation names
    std::map<int, std::string> nation_names;
    for (size_t i = 0; i < n_nationkey.size(); ++i) {
        nation_names[n_nationkey[i]] = std::string(&n_name[i * 25], 25);
    }
    
    // Get sizes
    const uint part_size = (uint)p_partkey.size(), supplier_size = (uint)s_suppkey.size(), lineitem_size = (uint)l_partkey.size();
    const uint partsupp_size = (uint)ps_partkey.size(), orders_size = (uint)o_orderkey.size();
    std::cout << "Loaded data for all tables." << std::endl;
    std::cout << "Part size: " << part_size << ", Supplier size: " << supplier_size << ", Lineitem size: " << lineitem_size << std::endl;

    // Debug: Check for 'green' in p_name
    int green_count = 0;
    for (size_t i = 0; i < part_size; ++i) {
        bool match = false;
        for(int j = 0; j < 50; ++j) {
            if (p_name[i * 55 + j] == 'g' && p_name[i * 55 + j + 1] == 'r' &&
                p_name[i * 55 + j + 2] == 'e' && p_name[i * 55 + j + 3] == 'e' &&
                p_name[i * 55 + j + 4] == 'n') {
                match = true;
                break;
            }
        }
        if (match) green_count++;
    }
    std::cout << "Found " << green_count << " parts with 'green' in name (CPU check)." << std::endl;


    // 2. Setup all kernel pipelines
    NS::Error* pError = nullptr;
    MTL::Function* pPartBuildFn = pLibrary->newFunction(NS::String::string("q9_build_part_ht_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pPartBuildPipe = pDevice->newComputePipelineState(pPartBuildFn, &pError);
    if (!pPartBuildPipe) { std::cerr << "Failed to create pPartBuildPipe: " << pError->localizedDescription()->utf8String() << std::endl; return; }

    MTL::Function* pSuppBuildFn = pLibrary->newFunction(NS::String::string("q9_build_supplier_ht_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pSuppBuildPipe = pDevice->newComputePipelineState(pSuppBuildFn, &pError);
    if (!pSuppBuildPipe) { std::cerr << "Failed to create pSuppBuildPipe: " << pError->localizedDescription()->utf8String() << std::endl; return; }

    MTL::Function* pPartSuppBuildFn = pLibrary->newFunction(NS::String::string("q9_build_partsupp_ht_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pPartSuppBuildPipe = pDevice->newComputePipelineState(pPartSuppBuildFn, &pError);
    if (!pPartSuppBuildPipe) { std::cerr << "Failed to create pPartSuppBuildPipe: " << pError->localizedDescription()->utf8String() << std::endl; return; }

    MTL::Function* pOrdersBuildFn = pLibrary->newFunction(NS::String::string("q9_build_orders_ht_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pOrdersBuildPipe = pDevice->newComputePipelineState(pOrdersBuildFn, &pError);
    if (!pOrdersBuildPipe) { std::cerr << "Failed to create pOrdersBuildPipe: " << pError->localizedDescription()->utf8String() << std::endl; return; }

    MTL::Function* pProbeAggFn = pLibrary->newFunction(NS::String::string("q9_probe_and_local_agg_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pProbeAggPipe = pDevice->newComputePipelineState(pProbeAggFn, &pError);
    if (!pProbeAggPipe) { std::cerr << "Failed to create pProbeAggPipe: " << pError->localizedDescription()->utf8String() << std::endl; return; }

    MTL::Function* pMergeFn = pLibrary->newFunction(NS::String::string("q9_merge_results_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pMergePipe = pDevice->newComputePipelineState(pMergeFn, &pError);
    if (!pMergePipe) { std::cerr << "Failed to create pMergePipe: " << pError->localizedDescription()->utf8String() << std::endl; return; }

    // 3. Create all GPU buffers
    // Part Bitmap (Optimization 1)
    int max_partkey = 0;
    for(int k : p_partkey) max_partkey = std::max(max_partkey, k);
    std::cout << "Max PartKey: " << max_partkey << std::endl;
    const uint part_bitmap_ints = (max_partkey + 31) / 32 + 1;
    MTL::Buffer* pPartBitmapBuffer = pDevice->newBuffer(part_bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);
    std::memset(pPartBitmapBuffer->contents(), 0, part_bitmap_ints * sizeof(uint));
    
    MTL::Buffer* pPartKeyBuffer = pDevice->newBuffer(p_partkey.data(), part_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartNameBuffer = pDevice->newBuffer(p_name.data(), p_name.size() * sizeof(char), MTL::ResourceStorageModeShared);
    // Dummy size for compatibility
    const uint part_ht_size = 0; 

    // Supplier Direct Map (Optimization 2)
    int max_suppkey = 0;
    for(int k : s_suppkey) max_suppkey = std::max(max_suppkey, k);
    std::cout << "Max SuppKey: " << max_suppkey << std::endl;
    const uint supp_map_size = max_suppkey + 1;
    MTL::Buffer* pSuppMapBuffer = pDevice->newBuffer(supp_map_size * sizeof(int), MTL::ResourceStorageModeShared);
    // Initialize with -1 to be safe
    std::memset(pSuppMapBuffer->contents(), -1, supp_map_size * sizeof(int));

    
    MTL::Buffer* pSuppKeyBuffer = pDevice->newBuffer(s_suppkey.data(), supplier_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSuppNationKeyBuffer = pDevice->newBuffer(s_nationkey.data(), supplier_size * sizeof(int), MTL::ResourceStorageModeShared);
    // Dummy size for compatibility
    const uint supplier_ht_size = 0;
    
    const uint partsupp_ht_size = partsupp_size * 4; // larger table to reduce probe lengths
    // PartSuppEntry has 4 ints (partkey, suppkey, idx, pad); initialize all to -1 to mark empty
    std::vector<int> cpu_partsupp_ht(partsupp_ht_size * 4, -1);
    MTL::Buffer* pPsPartKeyBuffer = pDevice->newBuffer(ps_partkey.data(), partsupp_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsSuppKeyBuffer = pDevice->newBuffer(ps_suppkey.data(), partsupp_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsSupplyCostBuffer = pDevice->newBuffer(ps_supplycost.data(), partsupp_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartSuppHTBuffer = pDevice->newBuffer(cpu_partsupp_ht.data(), partsupp_ht_size * sizeof(int) * 4, MTL::ResourceStorageModeShared);
    
    const uint orders_ht_size = orders_size * 2;
    std::vector<int> cpu_orders_ht(orders_ht_size * 2, -1);
    MTL::Buffer* pOrdKeyBuffer = pDevice->newBuffer(o_orderkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdDateBuffer = pDevice->newBuffer(o_orderdate.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdersHTBuffer = pDevice->newBuffer(cpu_orders_ht.data(), orders_ht_size * sizeof(int) * 2, MTL::ResourceStorageModeShared);

    MTL::Buffer* pLinePartKeyBuffer = pDevice->newBuffer(l_partkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineSuppKeyBuffer = pDevice->newBuffer(l_suppkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineOrdKeyBuffer = pDevice->newBuffer(l_orderkey.data(), lineitem_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineQtyBuffer = pDevice->newBuffer(l_quantity.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLinePriceBuffer = pDevice->newBuffer(l_extendedprice.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pLineDiscBuffer = pDevice->newBuffer(l_discount.data(), lineitem_size * sizeof(float), MTL::ResourceStorageModeShared);

    const uint num_threadgroups = 2048, local_ht_size = 256, intermediate_size = num_threadgroups * local_ht_size;
    MTL::Buffer* pIntermediateBuffer = pDevice->newBuffer(intermediate_size * sizeof(Q9Aggregates_CPU), MTL::ResourceStorageModeShared);
    // Ensure intermediate buffer is zero-initialized so merge stage can early-out on empty slots
    std::memset(pIntermediateBuffer->contents(), 0, intermediate_size * sizeof(Q9Aggregates_CPU));
    const uint final_ht_size = 25 * 10; // 25 nations * ~10 years
    std::vector<uint> cpu_final_ht(final_ht_size * (sizeof(Q9Aggregates_CPU)/sizeof(uint)), 0);
    MTL::Buffer* pFinalHTBuffer = pDevice->newBuffer(cpu_final_ht.data(), final_ht_size * sizeof(Q9Aggregates_CPU), MTL::ResourceStorageModeShared);

    // 4. Dispatch the entire 6-stage pipeline (2 warmup + 1 measured)
    double q9_gpu_compute_time = 0.0;
    
    for(int iter = 0; iter < 3; ++iter) {
        // Reset Buffers
        std::memset(pPartBitmapBuffer->contents(), 0, part_bitmap_ints * sizeof(uint));
        std::memset(pSuppMapBuffer->contents(), -1, supp_map_size * sizeof(int));
        std::memset(pPartSuppHTBuffer->contents(), 0xFF, partsupp_ht_size * sizeof(int) * 4);
        std::memset(pOrdersHTBuffer->contents(), 0xFF, orders_ht_size * sizeof(int) * 2);
        std::memset(pIntermediateBuffer->contents(), 0, intermediate_size * sizeof(Q9Aggregates_CPU));
        std::memset(pFinalHTBuffer->contents(), 0, final_ht_size * sizeof(Q9Aggregates_CPU));
        
        MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();

        // Encoder 1: Build Phase (Stages 1-4)
        MTL::ComputeCommandEncoder* pBuildEnc = pCommandBuffer->computeCommandEncoder();
        
        // Stage 1: Part build (Bitmap)
        pBuildEnc->setComputePipelineState(pPartBuildPipe);
        pBuildEnc->setBuffer(pPartKeyBuffer, 0, 0); pBuildEnc->setBuffer(pPartNameBuffer, 0, 1);
        pBuildEnc->setBuffer(pPartBitmapBuffer, 0, 2); pBuildEnc->setBytes(&part_size, sizeof(part_size), 3);
        pBuildEnc->setBytes(&part_ht_size, sizeof(part_ht_size), 4);
        {
            NS::UInteger threadGroupSize = pPartBuildPipe->maxTotalThreadsPerThreadgroup();
            if (threadGroupSize > 256) threadGroupSize = 256;
            MTL::Size threadgroupSize = MTL::Size(threadGroupSize, 1, 1);
            MTL::Size threadgroups = MTL::Size((part_size + threadGroupSize - 1) / threadGroupSize, 1, 1);
            pBuildEnc->dispatchThreadgroups(threadgroups, threadgroupSize);
        }
        
        // Stage 2: Supplier build (Direct Map)
        pBuildEnc->setComputePipelineState(pSuppBuildPipe);
        pBuildEnc->setBuffer(pSuppKeyBuffer, 0, 0); pBuildEnc->setBuffer(pSuppNationKeyBuffer, 0, 1);
        pBuildEnc->setBuffer(pSuppMapBuffer, 0, 2); pBuildEnc->setBytes(&supplier_size, sizeof(supplier_size), 3);
        pBuildEnc->setBytes(&supplier_ht_size, sizeof(supplier_ht_size), 4);
        {
            NS::UInteger threadGroupSize = pSuppBuildPipe->maxTotalThreadsPerThreadgroup();
            if (threadGroupSize > 256) threadGroupSize = 256;
            MTL::Size threadgroupSize = MTL::Size(threadGroupSize, 1, 1);
            MTL::Size threadgroups = MTL::Size((supplier_size + threadGroupSize - 1) / threadGroupSize, 1, 1);
            pBuildEnc->dispatchThreadgroups(threadgroups, threadgroupSize);
        }
        
        // Stage 3: PartSupp build
        pBuildEnc->setComputePipelineState(pPartSuppBuildPipe);
        pBuildEnc->setBuffer(pPsPartKeyBuffer, 0, 0); pBuildEnc->setBuffer(pPsSuppKeyBuffer, 0, 1);
        pBuildEnc->setBuffer(pPartSuppHTBuffer, 0, 2); pBuildEnc->setBytes(&partsupp_size, sizeof(partsupp_size), 3);
        pBuildEnc->setBytes(&partsupp_ht_size, sizeof(partsupp_ht_size), 4);
        {
            NS::UInteger threadGroupSize = pPartSuppBuildPipe->maxTotalThreadsPerThreadgroup();
            if (threadGroupSize > 256) threadGroupSize = 256;
            MTL::Size threadgroupSize = MTL::Size(threadGroupSize, 1, 1);
            MTL::Size threadgroups = MTL::Size((partsupp_size + threadGroupSize - 1) / threadGroupSize, 1, 1);
            pBuildEnc->dispatchThreadgroups(threadgroups, threadgroupSize);
        }
        
        // Stage 4: Orders build
        pBuildEnc->setComputePipelineState(pOrdersBuildPipe);
        pBuildEnc->setBuffer(pOrdKeyBuffer, 0, 0); pBuildEnc->setBuffer(pOrdDateBuffer, 0, 1);
        pBuildEnc->setBuffer(pOrdersHTBuffer, 0, 2); pBuildEnc->setBytes(&orders_size, sizeof(orders_size), 3);
        pBuildEnc->setBytes(&orders_ht_size, sizeof(orders_ht_size), 4);
        {
            NS::UInteger threadGroupSize = pOrdersBuildPipe->maxTotalThreadsPerThreadgroup();
            if (threadGroupSize > 256) threadGroupSize = 256;
            MTL::Size threadgroupSize = MTL::Size(threadGroupSize, 1, 1);
            MTL::Size threadgroups = MTL::Size((orders_size + threadGroupSize - 1) / threadGroupSize, 1, 1);
            pBuildEnc->dispatchThreadgroups(threadgroups, threadgroupSize);
        }
        
        pBuildEnc->endEncoding();

        // Ensure build phase is complete before probe phase
        pCommandBuffer->commit();
        pCommandBuffer->waitUntilCompleted();
        
        double buildTime = 0;
        if (iter == 2) {
            buildTime = pCommandBuffer->GPUEndTime() - pCommandBuffer->GPUStartTime();
        }
        
        // Restart command buffer for probe phase
        pCommandBuffer = pCommandQueue->commandBuffer();

        // Encoder 2: Probe & Merge Phase (Stages 5-6)
        // Splitting encoders ensures memory consistency between builds and probe
        MTL::ComputeCommandEncoder* pProbeEnc = pCommandBuffer->computeCommandEncoder();
        
        // Stage 5: Probe + local aggregation
        pProbeEnc->setComputePipelineState(pProbeAggPipe);
        pProbeEnc->setBuffer(pLineSuppKeyBuffer, 0, 0); pProbeEnc->setBuffer(pLinePartKeyBuffer, 0, 1);
        pProbeEnc->setBuffer(pLineOrdKeyBuffer, 0, 2); pProbeEnc->setBuffer(pLinePriceBuffer, 0, 3);
        pProbeEnc->setBuffer(pLineDiscBuffer, 0, 4); pProbeEnc->setBuffer(pLineQtyBuffer, 0, 5);
        pProbeEnc->setBuffer(pPsSupplyCostBuffer, 0, 6); 
        pProbeEnc->setBuffer(pPartBitmapBuffer, 0, 7); // Bitmap
        pProbeEnc->setBuffer(pSuppMapBuffer, 0, 8);    // Direct Map
        pProbeEnc->setBuffer(pPartSuppHTBuffer, 0, 9);
        pProbeEnc->setBuffer(pOrdersHTBuffer, 0, 10); pProbeEnc->setBuffer(pIntermediateBuffer, 0, 11);
        pProbeEnc->setBytes(&lineitem_size, sizeof(lineitem_size), 12); pProbeEnc->setBytes(&part_ht_size, sizeof(part_ht_size), 13);
        pProbeEnc->setBytes(&supplier_ht_size, sizeof(supplier_ht_size), 14); pProbeEnc->setBytes(&partsupp_ht_size, sizeof(partsupp_ht_size), 15);
        pProbeEnc->setBytes(&orders_ht_size, sizeof(orders_ht_size), 16);
        pProbeEnc->dispatchThreadgroups(MTL::Size(num_threadgroups, 1, 1), MTL::Size(1024, 1, 1));
        
        // Stage 6: Merge
        pProbeEnc->setComputePipelineState(pMergePipe);
        pProbeEnc->setBuffer(pIntermediateBuffer, 0, 0); pProbeEnc->setBuffer(pFinalHTBuffer, 0, 1);
        pProbeEnc->setBytes(&intermediate_size, sizeof(intermediate_size), 2); pProbeEnc->setBytes(&final_ht_size, sizeof(final_ht_size), 3);
        pProbeEnc->dispatchThreads(MTL::Size(intermediate_size, 1, 1), MTL::Size(1024, 1, 1));
        
        pProbeEnc->endEncoding();

        // Execute and time total Q9
        pCommandBuffer->commit();
        pCommandBuffer->waitUntilCompleted();
        
        if (iter == 2) {
            double probeTime = pCommandBuffer->GPUEndTime() - pCommandBuffer->GPUStartTime();
            q9_gpu_compute_time = buildTime + probeTime;
        }
    }

    // 6. CPU post-processing: read, aggregate, and sort results
    auto q9_cpu_post_start = std::chrono::high_resolution_clock::now();
    Q9Aggregates_CPU* results = (Q9Aggregates_CPU*)pFinalHTBuffer->contents();
    std::vector<Q9Result> final_results;
    for (uint i = 0; i < final_ht_size; ++i) {
        if (results[i].key != 0) {
            int nationkey = (results[i].key >> 16) & 0xFFFF;
            int year = results[i].key & 0xFFFF;
            final_results.push_back({nationkey, year, results[i].profit});
        }
    }
    std::sort(final_results.begin(), final_results.end(), [](const Q9Result& a, const Q9Result& b) {
        if (a.nationkey != b.nationkey) return a.nationkey < b.nationkey;
        return a.year > b.year;
    });
    printf("\nTPC-H Query 9 Results (Top 15):\n");
    printf("+------------+------+---------------+\n");
    printf("| Nation     | Year |        Profit |\n");
    printf("+------------+------+---------------+\n");
    for (size_t i = 0; i < 15 && i < final_results.size(); ++i) {
        printf("| %-10s | %4d | $%13.2f |\n",
               nation_names[final_results[i].nationkey].c_str(), final_results[i].year, final_results[i].profit);
    }
    printf("+------------+------+---------------+\n");
    printf("Total results found: %lu\n", final_results.size());
    // Comparable view: aggregate by year to match DuckDB's o_year -> sum_profit output
    std::map<int, double> year_totals;
    for (const auto& r : final_results) {
        year_totals[r.year] += (double)r.profit;
    }
    printf("\nComparable TPC-H Q9 (yearly sum_profit):\n");
    printf("+--------+---------------+\n");
    printf("| o_year |   sum_profit  |\n");
    printf("+--------+---------------+\n");
    for (const auto& kv : year_totals) {
        printf("| %6d | %13.4f |\n", kv.first, kv.second);
    }
    printf("+--------+---------------+\n");
    auto q9_cpu_post_end = std::chrono::high_resolution_clock::now();
    double q9_cpu_ms = std::chrono::duration<double, std::milli>(q9_cpu_post_end - q9_cpu_post_start).count();

    double q9GpuMs = q9_gpu_compute_time * 1000.0;
    double q9TotalExecMs = q9GpuMs + q9_cpu_ms;
    printf("\nQ9 | %u rows (lineitem)\n", lineitem_size);
    printf("  CPU Parsing (.tbl): %10.2f ms\n", q9CpuParseMs);
    printf("  GPU Execution:      %10.2f ms\n", q9GpuMs);
    printf("  CPU Post Process:   %10.2f ms\n", q9_cpu_ms);
    printf("  Total Execution:    %10.2f ms  (GPU + CPU post)\n", q9TotalExecMs);
    
    // Release all functions and pipelines
    pPartBuildFn->release();
    pPartBuildPipe->release();
    pSuppBuildFn->release();
    pSuppBuildPipe->release();
    pPartSuppBuildFn->release();
    pPartSuppBuildPipe->release();
    pOrdersBuildFn->release();
    pOrdersBuildPipe->release();
    pProbeAggFn->release();
    pProbeAggPipe->release();
    pMergeFn->release();
    pMergePipe->release();
    
    // Release all buffers
    pPartKeyBuffer->release();
    pPartNameBuffer->release();
    pPartBitmapBuffer->release();
    pSuppKeyBuffer->release();
    pSuppNationKeyBuffer->release();
    pSuppMapBuffer->release();
    pPsPartKeyBuffer->release();
    pPsSuppKeyBuffer->release();
    pPsSupplyCostBuffer->release();
    pPartSuppHTBuffer->release();
    pOrdKeyBuffer->release();
    pOrdDateBuffer->release();
    pOrdersHTBuffer->release();
    pLinePartKeyBuffer->release();
    pLineSuppKeyBuffer->release();
    pLineOrdKeyBuffer->release();
    pLineQtyBuffer->release();
    pLinePriceBuffer->release();
    pLineDiscBuffer->release();
    pIntermediateBuffer->release();
    pFinalHTBuffer->release();
}


struct Q13Result {
    uint c_count;
    uint custdist;
};

// --- Main Function for TPC-H Q13 Benchmark ---
void runQ13Benchmark(MTL::Device* pDevice, MTL::CommandQueue* pCommandQueue, MTL::Library* pLibrary) {
    std::cout << "\n--- Running TPC-H Query 13 Benchmark ---" << std::endl;

    const std::string sf_path = g_dataset_path;
    
    // 1. Load data
    auto q13ParseStart = std::chrono::high_resolution_clock::now();
    auto o_custkey = loadIntColumn(sf_path + "orders.tbl", 1);
    auto o_comment = loadCharColumn(sf_path + "orders.tbl", 8, 100);
    auto c_custkey = loadIntColumn(sf_path + "customer.tbl", 0);
    auto q13ParseEnd = std::chrono::high_resolution_clock::now();
    double q13CpuParseMs = std::chrono::duration<double, std::milli>(q13ParseEnd - q13ParseStart).count();

    const uint orders_size = (uint)o_custkey.size();
    const uint customer_size = (uint)c_custkey.size();
    std::cout << "Loaded " << orders_size << " orders and " << customer_size << " customers." << std::endl;

    // 2. Setup kernels
    NS::Error* pError = nullptr;
    MTL::Function* pFusedCountFn = pLibrary->newFunction(NS::String::string("q13_fused_direct_count_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pFusedCountPipe = pDevice->newComputePipelineState(pFusedCountFn, &pError);

    MTL::Function* pHistFn = pLibrary->newFunction(NS::String::string("q13_build_histogram_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pHistPipe = pDevice->newComputePipelineState(pHistFn, &pError);

    // 3. Create Buffers
    const uint num_threadgroups = 2048;
    MTL::Buffer* pOrdCustKeyBuffer = pDevice->newBuffer(o_custkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdCommentBuffer = pDevice->newBuffer(o_comment.data(), o_comment.size() * sizeof(char), MTL::ResourceStorageModeShared);

    // Direct mapping output: per-customer order counts (index = custkey - 1).
    std::vector<uint> cpu_counts_per_customer(customer_size, 0u);
    MTL::Buffer* pCountsPerCustomerBuffer = pDevice->newBuffer(cpu_counts_per_customer.data(), customer_size * sizeof(uint), MTL::ResourceStorageModeShared);

    // Histogram buffer for GPU histogram kernel
    const uint hist_max_bins = 256;
    MTL::Buffer* pHistogramBuf = pDevice->newBuffer(hist_max_bins * sizeof(uint), MTL::ResourceStorageModeShared);

    // 4. Dispatch the fused GPU stage (2 warmup + 1 measured)
    double gpuExecutionTime = 0.0;
    
    for(int iter = 0; iter < 3; ++iter) {
        // Reset output buffers
        std::memset(pCountsPerCustomerBuffer->contents(), 0, customer_size * sizeof(uint));
        std::memset(pHistogramBuf->contents(), 0, hist_max_bins * sizeof(uint));
        
        MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
        
        // Encoder 1: count kernel
        MTL::ComputeCommandEncoder* enc1 = pCommandBuffer->computeCommandEncoder();
        enc1->setComputePipelineState(pFusedCountPipe);
        enc1->setBuffer(pOrdCustKeyBuffer, 0, 0);
        enc1->setBuffer(pOrdCommentBuffer, 0, 1);
        enc1->setBuffer(pCountsPerCustomerBuffer, 0, 2);
        enc1->setBytes(&orders_size, sizeof(orders_size), 3);
        enc1->setBytes(&customer_size, sizeof(customer_size), 4);
        enc1->dispatchThreadgroups(MTL::Size(num_threadgroups, 1, 1), MTL::Size(1024, 1, 1));
        enc1->endEncoding();

        // Encoder 2: histogram kernel (encoder boundary provides memory barrier)
        MTL::ComputeCommandEncoder* enc2 = pCommandBuffer->computeCommandEncoder();
        enc2->setComputePipelineState(pHistPipe);
        enc2->setBuffer(pCountsPerCustomerBuffer, 0, 0);
        enc2->setBuffer(pHistogramBuf, 0, 1);
        enc2->setBytes(&customer_size, sizeof(customer_size), 2);
        enc2->setBytes(&hist_max_bins, sizeof(hist_max_bins), 3);
        enc2->dispatchThreadgroups(MTL::Size(num_threadgroups, 1, 1), MTL::Size(1024, 1, 1));
        enc2->endEncoding();

        // 5. Execute GPU work
        pCommandBuffer->commit();
        pCommandBuffer->waitUntilCompleted();
        
        if (iter == 2) {
            gpuExecutionTime = pCommandBuffer->GPUEndTime() - pCommandBuffer->GPUStartTime();
        }
    }

    // 6. Read back histogram from GPU
    auto q13_cpu_merge_start = std::chrono::high_resolution_clock::now();
    uint* hist = (uint*)pHistogramBuf->contents();
    std::vector<Q13Result> final_results;
    for (uint i = 0; i < hist_max_bins; i++) {
        if (hist[i] > 0) {
            final_results.push_back({i, hist[i]});
        }
    }
    std::sort(final_results.begin(), final_results.end(), [](const Q13Result& a, const Q13Result& b) {
        if (a.custdist != b.custdist) return a.custdist > b.custdist;
        return a.c_count > b.c_count;
    });
    auto q13_cpu_merge_end = std::chrono::high_resolution_clock::now();
    double q13_cpu_merge_time = std::chrono::duration<double>(q13_cpu_merge_end - q13_cpu_merge_start).count();

    printf("\nTPC-H Query 13 Results (Comparable histogram):\n");
    printf("+---------+----------+\n");
    printf("| c_count | custdist |\n");
    printf("+---------+----------+\n");
    for(const auto& res : final_results) {
        printf("| %7u | %8u |\n", res.c_count, res.custdist);
    }
    printf("+---------+----------+\n");

    double q13GpuMs = gpuExecutionTime * 1000.0;
    double q13CpuPostMs = q13_cpu_merge_time * 1000.0;
    double q13TotalExecMs = q13GpuMs + q13CpuPostMs;
    printf("\nQ13 | %u orders | %u customers\n", orders_size, customer_size);
    printf("  CPU Parsing (.tbl): %10.2f ms\n", q13CpuParseMs);
    printf("  GPU Execution:      %10.2f ms\n", q13GpuMs);
    printf("  CPU Post Process:   %10.2f ms\n", q13CpuPostMs);
    printf("  Total Execution:    %10.2f ms  (GPU + CPU post)\n", q13TotalExecMs);

    // Release objects...
    pFusedCountFn->release();
    pFusedCountPipe->release();
    pHistFn->release();
    pHistPipe->release();
    pOrdCustKeyBuffer->release();
    pOrdCommentBuffer->release();
    pCountsPerCustomerBuffer->release();
    pHistogramBuf->release();
}


// ===================================================================
// SF100 Q3 — Chunked Streaming with mmap for join queries
// ===================================================================
void runQ3BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q3 Benchmark (SF100 Chunked) ===" << std::endl;

    // For Q3, we need customer (small), orders (medium), lineitem (huge).
    // Strategy: Load customer + orders fully (they fit ~22GB at SF100).
    // Stream lineitem in chunks.
    
    // Check if we have enough memory for the build side
    size_t maxMem = device->recommendedMaxWorkingSetSize();
    printf("GPU max working set: %zu MB\n", maxMem / (1024*1024));

    // Load small tables fully
    MappedFile custFile, ordFile, liFile;
    if (!custFile.open(g_dataset_path + "customer.tbl") ||
        !ordFile.open(g_dataset_path + "orders.tbl") ||
        !liFile.open(g_dataset_path + "lineitem.tbl")) {
        std::cerr << "Q3 SF100: Cannot open required TBL files" << std::endl;
        return;
    }

    // Build full customer/orders data (they fit in memory at SF100)
    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto custIndex = buildLineIndex(custFile);
    auto ordIndex = buildLineIndex(ordFile);
    auto liIndex = buildLineIndex(liFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();
    size_t custRows = custIndex.size(), ordRows = ordIndex.size(), liRows = liIndex.size();
    printf("Q3 SF100: customer=%zu, orders=%zu, lineitem=%zu rows (index %.1f ms)\n", custRows, ordRows, liRows, indexBuildMs);

    // Load customer: c_custkey(col 0), c_mktsegment(col 6, first char)
    double buildParseCpuMs = 0;
    auto bpT0 = std::chrono::high_resolution_clock::now();
    std::vector<int> c_custkey(custRows);
    std::vector<char> c_mktsegment(custRows);
    parseIntColumnChunk(custFile, custIndex, 0, custRows, 0, c_custkey.data());
    parseCharColumnChunk(custFile, custIndex, 0, custRows, 6, c_mktsegment.data());

    // Load orders: o_orderkey(col 0), o_custkey(col 1), o_orderdate(col 4), o_shippriority(col 7)
    std::vector<int> o_orderkey(ordRows), o_custkey(ordRows), o_orderdate(ordRows), o_shippriority(ordRows);
    parseIntColumnChunk(ordFile, ordIndex, 0, ordRows, 0, o_orderkey.data());
    parseIntColumnChunk(ordFile, ordIndex, 0, ordRows, 1, o_custkey.data());
    parseDateColumnChunk(ordFile, ordIndex, 0, ordRows, 4, o_orderdate.data());
    parseIntColumnChunk(ordFile, ordIndex, 0, ordRows, 7, o_shippriority.data());
    auto bpT1 = std::chrono::high_resolution_clock::now();
    buildParseCpuMs = std::chrono::duration<double, std::milli>(bpT1 - bpT0).count();

    // Use existing Q3 GPU pipeline for build phase (bitmap + direct map)
    NS::Error* pError = nullptr;
    MTL::Function* pCustBuildFn = library->newFunction(NS::String::string("q3_build_customer_bitmap_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pCustBuildPipe = device->newComputePipelineState(pCustBuildFn, &pError);
    MTL::Function* pOrdersBuildFn = library->newFunction(NS::String::string("q3_build_orders_map_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pOrdersBuildPipe = device->newComputePipelineState(pOrdersBuildFn, &pError);
    MTL::Function* pFusedProbeAggFn = library->newFunction(NS::String::string("q3_probe_and_aggregate_direct_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pFusedProbeAggPipe = device->newComputePipelineState(pFusedProbeAggFn, &pError);
    MTL::Function* pCompactFn = library->newFunction(NS::String::string("q3_compact_results_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pCompactPipe = device->newComputePipelineState(pCompactFn, &pError);

    if (!pCustBuildPipe || !pOrdersBuildPipe || !pFusedProbeAggPipe || !pCompactPipe) {
        std::cerr << "Q3 SF100: Failed to create pipeline states" << std::endl;
        return;
    }

    // Build phase buffers (loaded once)
    const uint customer_size = (uint)custRows, orders_size = (uint)ordRows;
    int max_custkey = 0;
    for (auto k : c_custkey) max_custkey = std::max(max_custkey, k);
    const uint customer_bitmap_ints = (max_custkey + 31) / 32 + 1;
    MTL::Buffer* pCustBitmapBuf = device->newBuffer(customer_bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);
    memset(pCustBitmapBuf->contents(), 0, customer_bitmap_ints * sizeof(uint));
    MTL::Buffer* pCustKeyBuf = device->newBuffer(c_custkey.data(), customer_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCustMktBuf = device->newBuffer(c_mktsegment.data(), customer_size * sizeof(char), MTL::ResourceStorageModeShared);

    int max_orderkey = 0;
    for (auto k : o_orderkey) max_orderkey = std::max(max_orderkey, k);
    const uint orders_map_size = max_orderkey + 1;
    
    // Check if orders map fits in memory
    size_t ordersMapBytes = (size_t)orders_map_size * sizeof(int);
    printf("Orders map: %u entries (%.1f MB)\n", orders_map_size, ordersMapBytes / (1024.0*1024.0));
    if (ordersMapBytes > maxMem * 0.3) {
        std::cerr << "Q3 SF100: Orders direct map too large (" << ordersMapBytes/(1024*1024) << " MB). Need radix-partitioned join." << std::endl;
        std::cerr << "  This is expected for SF100. Radix join path coming soon." << std::endl;
        pCustBuildFn->release(); pCustBuildPipe->release();
        pOrdersBuildFn->release(); pOrdersBuildPipe->release();
        pFusedProbeAggFn->release(); pFusedProbeAggPipe->release();
        pCustKeyBuf->release(); pCustMktBuf->release(); pCustBitmapBuf->release();
        return;
    }

    MTL::Buffer* pOrdersMapBuf = device->newBuffer(ordersMapBytes, MTL::ResourceStorageModeShared);
    memset(pOrdersMapBuf->contents(), -1, ordersMapBytes);

    MTL::Buffer* pOrdKeyBuf = device->newBuffer(o_orderkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdCustBuf = device->newBuffer(o_custkey.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdDateBuf = device->newBuffer(o_orderdate.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdPrioBuf = device->newBuffer(o_shippriority.data(), orders_size * sizeof(int), MTL::ResourceStorageModeShared);

    const int cutoff_date = 19950315;

    // Execute build phase once
    double buildGpuMs = 0;
    {
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

        enc->setComputePipelineState(pCustBuildPipe);
        enc->setBuffer(pCustKeyBuf, 0, 0); enc->setBuffer(pCustMktBuf, 0, 1);
        enc->setBuffer(pCustBitmapBuf, 0, 2); enc->setBytes(&customer_size, sizeof(customer_size), 3);
        enc->dispatchThreadgroups(MTL::Size((customer_size + 255)/256, 1, 1), MTL::Size(256, 1, 1));

        enc->setComputePipelineState(pOrdersBuildPipe);
        enc->setBuffer(pOrdKeyBuf, 0, 0); enc->setBuffer(pOrdDateBuf, 0, 1);
        enc->setBuffer(pOrdersMapBuf, 0, 2); enc->setBytes(&orders_size, sizeof(orders_size), 3);
        enc->setBytes(&cutoff_date, sizeof(cutoff_date), 4);
        enc->dispatchThreadgroups(MTL::Size((orders_size + 255)/256, 1, 1), MTL::Size(256, 1, 1));

        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
        buildGpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
        printf("Build phase done (GPU: %.2f ms)\n", buildGpuMs);
    }

    // Stream lineitem in chunks: probe + aggregate directly into final HT
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 20, liRows); // 4 cols ~20B/row
    printf("Lineitem chunk size: %zu rows\n", chunkRows);

    struct Q3ChunkSlot {
        MTL::Buffer* orderkey; MTL::Buffer* shipdate; MTL::Buffer* extprice; MTL::Buffer* discount;
    };
    Q3ChunkSlot liSlots[2];
    for (int s = 0; s < 2; s++) {
        liSlots[s].orderkey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        liSlots[s].shipdate = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        liSlots[s].extprice = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        liSlots[s].discount = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
    }

    // Final hash table for GPU aggregation (persists across chunks)
    const uint q3_final_ht_size = (uint)std::max((size_t)(ordRows / 64), (size_t)(1 << 20));
    MTL::Buffer* pFinalHTBuf = device->newBuffer((size_t)q3_final_ht_size * sizeof(Q3Aggregates_CPU), MTL::ResourceStorageModeShared);
    std::memset(pFinalHTBuf->contents(), 0, (size_t)q3_final_ht_size * sizeof(Q3Aggregates_CPU));

    // Dense output buffer for GPU compaction
    MTL::Buffer* pDenseBuf = device->newBuffer((size_t)q3_final_ht_size * sizeof(Q3Aggregates_CPU), MTL::ResourceStorageModeShared);
    MTL::Buffer* pCountBuf = device->newBuffer(sizeof(uint), MTL::ResourceStorageModeShared);

    double totalGpuMs = 0.0, totalCpuParseMs = 0.0;
    size_t offset = 0, chunkNum = 0;

    // --- Double-buffered pipeline: overlap CPU parse of chunk N+1 with GPU exec of chunk N ---

    // Pre-parse first chunk into slot 0
    size_t rowsThisChunk = std::min(chunkRows, liRows);
    {
        Q3ChunkSlot& slot = liSlots[0];
        auto parseStart = std::chrono::high_resolution_clock::now();
        parseIntColumnChunk(liFile, liIndex, 0, rowsThisChunk, 0, (int*)slot.orderkey->contents());
        parseDateColumnChunk(liFile, liIndex, 0, rowsThisChunk, 10, (int*)slot.shipdate->contents());
        parseFloatColumnChunk(liFile, liIndex, 0, rowsThisChunk, 5, (float*)slot.extprice->contents());
        parseFloatColumnChunk(liFile, liIndex, 0, rowsThisChunk, 6, (float*)slot.discount->contents());
        auto parseEnd = std::chrono::high_resolution_clock::now();
        totalCpuParseMs += std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();
    }

    while (offset < liRows) {
        rowsThisChunk = std::min(chunkRows, liRows - offset);
        Q3ChunkSlot& slot = liSlots[chunkNum % 2];

        // Dispatch fused probe+aggregate on current slot
        uint lineitem_size = (uint)rowsThisChunk;
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

        enc->setComputePipelineState(pFusedProbeAggPipe);
        enc->setBuffer(slot.orderkey, 0, 0); enc->setBuffer(slot.shipdate, 0, 1);
        enc->setBuffer(slot.extprice, 0, 2); enc->setBuffer(slot.discount, 0, 3);
        enc->setBuffer(pCustBitmapBuf, 0, 4); enc->setBuffer(pOrdersMapBuf, 0, 5);
        enc->setBuffer(pOrdCustBuf, 0, 6); enc->setBuffer(pOrdDateBuf, 0, 7);
        enc->setBuffer(pOrdPrioBuf, 0, 8);
        enc->setBuffer(pFinalHTBuf, 0, 9);
        enc->setBytes(&lineitem_size, sizeof(lineitem_size), 10);
        enc->setBytes(&cutoff_date, sizeof(cutoff_date), 11);
        enc->setBytes(&q3_final_ht_size, sizeof(q3_final_ht_size), 12);
        enc->dispatchThreadgroups(MTL::Size(2048, 1, 1), MTL::Size(1024, 1, 1));
        enc->endEncoding();
        cb->commit();

        // --- Double-buffer: parse NEXT chunk into alternate slot while GPU runs ---
        size_t nextOffset = offset + rowsThisChunk;
        if (nextOffset < liRows) {
            size_t nextRows = std::min(chunkRows, liRows - nextOffset);
            Q3ChunkSlot& nextSlot = liSlots[(chunkNum + 1) % 2];
            auto parseStart = std::chrono::high_resolution_clock::now();
            parseIntColumnChunk(liFile, liIndex, nextOffset, nextRows, 0, (int*)nextSlot.orderkey->contents());
            parseDateColumnChunk(liFile, liIndex, nextOffset, nextRows, 10, (int*)nextSlot.shipdate->contents());
            parseFloatColumnChunk(liFile, liIndex, nextOffset, nextRows, 5, (float*)nextSlot.extprice->contents());
            parseFloatColumnChunk(liFile, liIndex, nextOffset, nextRows, 6, (float*)nextSlot.discount->contents());
            auto parseEnd = std::chrono::high_resolution_clock::now();
            totalCpuParseMs += std::chrono::duration<double, std::milli>(parseEnd - parseStart).count();
        }

        // Wait for GPU to finish this chunk
        cb->waitUntilCompleted();
        double gpuMs = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
        totalGpuMs += gpuMs;

        chunkNum++;
        offset += rowsThisChunk;
    }

    // GPU compaction: extract non-empty HT entries into dense buffer
    *(uint*)pCountBuf->contents() = 0;
    {
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pCompactPipe);
        enc->setBuffer(pFinalHTBuf, 0, 0);
        enc->setBuffer(pDenseBuf, 0, 1);
        enc->setBuffer(pCountBuf, 0, 2);
        enc->setBytes(&q3_final_ht_size, sizeof(q3_final_ht_size), 3);
        enc->dispatchThreadgroups(MTL::Size((q3_final_ht_size + 255) / 256, 1, 1), MTL::Size(256, 1, 1));
        enc->endEncoding();
        cb->commit();
        cb->waitUntilCompleted();
        totalGpuMs += (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
    }

    uint resultCount = *(uint*)pCountBuf->contents();
    Q3Aggregates_CPU* dense = (Q3Aggregates_CPU*)pDenseBuf->contents();

    // CPU: partial_sort for top 10 only
    auto sortStart = std::chrono::high_resolution_clock::now();
    size_t topK = std::min((size_t)10, (size_t)resultCount);
    std::partial_sort(dense, dense + topK, dense + resultCount,
        [](const Q3Aggregates_CPU& a, const Q3Aggregates_CPU& b) {
            if (a.revenue != b.revenue) return a.revenue > b.revenue;
            return a.orderdate < b.orderdate;
        });
    auto sortEnd = std::chrono::high_resolution_clock::now();
    double sortMs = std::chrono::duration<double, std::milli>(sortEnd - sortStart).count();

    printf("\nTPC-H Q3 Results (Top 10):\n");
    printf("+----------+------------+------------+--------------+\n");
    printf("| orderkey |   revenue  | orderdate  | shippriority |\n");
    printf("+----------+------------+------------+--------------+\n");
    for (size_t i = 0; i < topK; i++) {
        printf("| %8d | $%10.2f | %10u | %12u |\n",
               dense[i].key, dense[i].revenue, dense[i].orderdate, dense[i].shippriority);
    }
    printf("+----------+------------+------------+--------------+\n");

    double totalCpuPostAllMs = sortMs;
    double allGpuMs = totalGpuMs + buildGpuMs;
    double allCpuParseMs = indexBuildMs + buildParseCpuMs + totalCpuParseMs;
    double totalExecMs = totalCpuPostAllMs + allGpuMs;

    printf("\nSF100 Q3 | %zu chunks | %zu rows\n", chunkNum, liRows);
    printf("  CPU Parsing (.tbl): %10.2f ms\n", allCpuParseMs);
    printf("  GPU Execution:      %10.2f ms\n", allGpuMs);
    printf("  CPU Post Process:   %10.2f ms\n", totalCpuPostAllMs);
    printf("  Total Execution:    %10.2f ms  (GPU + CPU post)\n", totalExecMs);

    // Cleanup
    pCustBuildFn->release(); pCustBuildPipe->release();
    pOrdersBuildFn->release(); pOrdersBuildPipe->release();
    pFusedProbeAggFn->release(); pFusedProbeAggPipe->release();
    pCompactFn->release(); pCompactPipe->release();
    pCustKeyBuf->release(); pCustMktBuf->release(); pCustBitmapBuf->release();
    pOrdersMapBuf->release();
    pOrdKeyBuf->release(); pOrdCustBuf->release(); pOrdDateBuf->release(); pOrdPrioBuf->release();
    for (int s = 0; s < 2; s++) {
        liSlots[s].orderkey->release(); liSlots[s].shipdate->release();
        liSlots[s].extprice->release(); liSlots[s].discount->release();
    }
    pFinalHTBuf->release();
    pDenseBuf->release();
    pCountBuf->release();
}


// ===================================================================
// SF100 Q9 — Hybrid Streaming: Build HTs sequentially, Stream Lineitem
// ===================================================================
void runQ9BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q9 Benchmark (SF100 Hybrid Streaming) ===" << std::endl;

    size_t maxMem = device->recommendedMaxWorkingSetSize();
    printf("GPU max working set: %zu MB\n", maxMem / (1024*1024));

    // ── Open all mmap files ──
    MappedFile partFile, suppFile, psFile, ordFile, natFile, liFile;
    if (!partFile.open(g_dataset_path + "part.tbl") || !suppFile.open(g_dataset_path + "supplier.tbl") ||
        !psFile.open(g_dataset_path + "partsupp.tbl") || !ordFile.open(g_dataset_path + "orders.tbl") ||
        !natFile.open(g_dataset_path + "nation.tbl") || !liFile.open(g_dataset_path + "lineitem.tbl")) {
        std::cerr << "Q9 SF100: Cannot open required files" << std::endl;
        return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto partIdx = buildLineIndex(partFile), suppIdx = buildLineIndex(suppFile);
    auto psIdx = buildLineIndex(psFile), ordIdx = buildLineIndex(ordFile);
    auto natIdx = buildLineIndex(natFile), liIdx = buildLineIndex(liFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();

    size_t partRows = partIdx.size(), suppRows = suppIdx.size();
    size_t psRows = psIdx.size(), ordRows = ordIdx.size();
    size_t liRows = liIdx.size();
    printf("Q9 SF100: part=%zu, supplier=%zu, partsupp=%zu, orders=%zu, lineitem=%zu (index %.1f ms)\n",
           partRows, suppRows, psRows, ordRows, liRows, indexBuildMs);

    // ── Load nation data (tiny — always fits) ──
    std::vector<int> n_nationkey(natIdx.size()), n_name_raw(natIdx.size());
    parseIntColumnChunk(natFile, natIdx, 0, natIdx.size(), 0, n_nationkey.data());
    // Load nation names as char column for mapping later
    std::vector<char> n_name_chars(natIdx.size() * 25, ' ');
    parseCharColumnChunkFixed(natFile, natIdx, 0, natIdx.size(), 1, 25, n_name_chars.data());
    std::map<int, std::string> nation_names;
    for (size_t i = 0; i < natIdx.size(); ++i) {
        nation_names[n_nationkey[i]] = std::string(&n_name_chars[i * 25], 25);
        // trim trailing spaces
        auto& s = nation_names[n_nationkey[i]];
        s.erase(s.find_last_not_of(' ') + 1);
    }

    // ── Setup all kernel pipelines ──
    NS::Error* pError = nullptr;
    MTL::Function* pPartBuildFn = library->newFunction(NS::String::string("q9_build_part_ht_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pPartBuildPipe = device->newComputePipelineState(pPartBuildFn, &pError);
    MTL::Function* pSuppBuildFn = library->newFunction(NS::String::string("q9_build_supplier_ht_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pSuppBuildPipe = device->newComputePipelineState(pSuppBuildFn, &pError);
    MTL::Function* pPartSuppBuildFn = library->newFunction(NS::String::string("q9_build_partsupp_ht_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pPartSuppBuildPipe = device->newComputePipelineState(pPartSuppBuildFn, &pError);
    MTL::Function* pOrdersBuildFn = library->newFunction(NS::String::string("q9_build_orders_ht_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pOrdersBuildPipe = device->newComputePipelineState(pOrdersBuildFn, &pError);
    MTL::Function* pProbeAggFn = library->newFunction(NS::String::string("q9_probe_and_local_agg_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pProbeAggPipe = device->newComputePipelineState(pProbeAggFn, &pError);
    MTL::Function* pMergeFn = library->newFunction(NS::String::string("q9_merge_results_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pMergePipe = device->newComputePipelineState(pMergeFn, &pError);

    if (!pPartBuildPipe || !pSuppBuildPipe || !pPartSuppBuildPipe || !pOrdersBuildPipe || !pProbeAggPipe || !pMergePipe) {
        std::cerr << "Q9 SF100: Failed to create one or more pipelines" << std::endl;
        return;
    }

    // Timing accumulators
    double totalCpuParseMs = 0, totalGpuMs = 0, totalCpuPostMs = 0;

    // ══════════════════════════════════════════════════════════════
    // PHASE 1: Build hash tables — load one table at a time,
    //          parse on CPU, upload to GPU, free CPU RAM immediately.
    // ══════════════════════════════════════════════════════════════
    printf("\n--- Phase 1: Build Hash Tables (sequential, memory-conscious) ---\n");

    // --- 1a. Part Bitmap ---
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<int> p_partkey(partRows);
    std::vector<char> p_name(partRows * 55);
    parseIntColumnChunk(partFile, partIdx, 0, partRows, 0, p_partkey.data());
    parseCharColumnChunkFixed(partFile, partIdx, 0, partRows, 1, 55, p_name.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    totalCpuParseMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

    int max_partkey = 0;
    for (size_t i = 0; i < partRows; i++) max_partkey = std::max(max_partkey, p_partkey[i]);
    const uint part_bitmap_ints = (max_partkey + 31) / 32 + 1;
    const uint part_ht_size = 0; // bitmap mode

    MTL::Buffer* pPartKeyBuf = device->newBuffer(p_partkey.data(), partRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartNameBuf = device->newBuffer(p_name.data(), p_name.size(), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartBitmapBuf = device->newBuffer(part_bitmap_ints * sizeof(uint), MTL::ResourceStorageModeShared);
    memset(pPartBitmapBuf->contents(), 0, part_bitmap_ints * sizeof(uint));

    {
        uint ps = (uint)partRows;
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pPartBuildPipe);
        enc->setBuffer(pPartKeyBuf, 0, 0);
        enc->setBuffer(pPartNameBuf, 0, 1);
        enc->setBuffer(pPartBitmapBuf, 0, 2);
        enc->setBytes(&ps, sizeof(ps), 3);
        enc->setBytes(&part_ht_size, sizeof(part_ht_size), 4);
        NS::UInteger tgs = std::min((NS::UInteger)256, pPartBuildPipe->maxTotalThreadsPerThreadgroup());
        enc->dispatchThreadgroups(MTL::Size((partRows + tgs - 1) / tgs, 1, 1), MTL::Size(tgs, 1, 1));
        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
        totalGpuMs += (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
    }
    // Free CPU-side vectors and GPU input buffers
    { std::vector<int>().swap(p_partkey); std::vector<char>().swap(p_name); }
    pPartKeyBuf->release(); pPartNameBuf->release();
    printf("  Part bitmap built (%u ints, max_partkey=%d)\n", part_bitmap_ints, max_partkey);

    // --- 1b. Supplier Direct Map ---
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<int> s_suppkey(suppRows), s_nationkey(suppRows);
    parseIntColumnChunk(suppFile, suppIdx, 0, suppRows, 0, s_suppkey.data());
    parseIntColumnChunk(suppFile, suppIdx, 0, suppRows, 3, s_nationkey.data());
    t1 = std::chrono::high_resolution_clock::now();
    totalCpuParseMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

    int max_suppkey = 0;
    for (size_t i = 0; i < suppRows; i++) max_suppkey = std::max(max_suppkey, s_suppkey[i]);
    const uint supp_map_size = max_suppkey + 1;
    const uint supplier_ht_size = 0; // direct map mode

    MTL::Buffer* pSuppKeyBuf = device->newBuffer(s_suppkey.data(), suppRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSuppNatBuf = device->newBuffer(s_nationkey.data(), suppRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pSuppMapBuf = device->newBuffer(supp_map_size * sizeof(int), MTL::ResourceStorageModeShared);
    memset(pSuppMapBuf->contents(), 0xFF, supp_map_size * sizeof(int)); // -1

    {
        uint ss = (uint)suppRows;
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pSuppBuildPipe);
        enc->setBuffer(pSuppKeyBuf, 0, 0);
        enc->setBuffer(pSuppNatBuf, 0, 1);
        enc->setBuffer(pSuppMapBuf, 0, 2);
        enc->setBytes(&ss, sizeof(ss), 3);
        enc->setBytes(&supplier_ht_size, sizeof(supplier_ht_size), 4);
        NS::UInteger tgs = std::min((NS::UInteger)256, pSuppBuildPipe->maxTotalThreadsPerThreadgroup());
        enc->dispatchThreadgroups(MTL::Size((suppRows + tgs - 1) / tgs, 1, 1), MTL::Size(tgs, 1, 1));
        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
        totalGpuMs += (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
    }
    { std::vector<int>().swap(s_suppkey); std::vector<int>().swap(s_nationkey); }
    pSuppKeyBuf->release(); pSuppNatBuf->release();
    printf("  Supplier direct map built (size=%u)\n", supp_map_size);

    // --- 1c. Orders Hash Table ---
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<int> o_orderkey(ordRows), o_orderdate(ordRows);
    parseIntColumnChunk(ordFile, ordIdx, 0, ordRows, 0, o_orderkey.data());
    parseDateColumnChunk(ordFile, ordIdx, 0, ordRows, 4, o_orderdate.data());
    t1 = std::chrono::high_resolution_clock::now();
    totalCpuParseMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

    const uint orders_ht_size = (uint)ordRows * 2;
    MTL::Buffer* pOrdKeyBuf = device->newBuffer(o_orderkey.data(), ordRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdDateBuf = device->newBuffer(o_orderdate.data(), ordRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pOrdersHTBuf = device->newBuffer(orders_ht_size * sizeof(int) * 2, MTL::ResourceStorageModeShared);
    memset(pOrdersHTBuf->contents(), 0xFF, orders_ht_size * sizeof(int) * 2);

    {
        uint os = (uint)ordRows;
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pOrdersBuildPipe);
        enc->setBuffer(pOrdKeyBuf, 0, 0);
        enc->setBuffer(pOrdDateBuf, 0, 1);
        enc->setBuffer(pOrdersHTBuf, 0, 2);
        enc->setBytes(&os, sizeof(os), 3);
        enc->setBytes(&orders_ht_size, sizeof(orders_ht_size), 4);
        NS::UInteger tgs = std::min((NS::UInteger)256, pOrdersBuildPipe->maxTotalThreadsPerThreadgroup());
        enc->dispatchThreadgroups(MTL::Size((ordRows + tgs - 1) / tgs, 1, 1), MTL::Size(tgs, 1, 1));
        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
        totalGpuMs += (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
    }
    { std::vector<int>().swap(o_orderkey); std::vector<int>().swap(o_orderdate); }
    pOrdKeyBuf->release(); pOrdDateBuf->release();
    printf("  Orders HT built (ht_size=%u)\n", orders_ht_size);

    // --- 1d. PartSupp Hash Table ---
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<int> ps_partkey(psRows), ps_suppkey(psRows);
    std::vector<float> ps_supplycost(psRows);
    parseIntColumnChunk(psFile, psIdx, 0, psRows, 0, ps_partkey.data());
    parseIntColumnChunk(psFile, psIdx, 0, psRows, 1, ps_suppkey.data());
    parseFloatColumnChunk(psFile, psIdx, 0, psRows, 3, ps_supplycost.data());
    t1 = std::chrono::high_resolution_clock::now();
    totalCpuParseMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

    const uint partsupp_ht_size = (uint)psRows * 4;
    MTL::Buffer* pPsPartKeyBuf = device->newBuffer(ps_partkey.data(), psRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsSuppKeyBuf = device->newBuffer(ps_suppkey.data(), psRows * sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPsSupplyCostBuf = device->newBuffer(ps_supplycost.data(), psRows * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* pPartSuppHTBuf = device->newBuffer(partsupp_ht_size * sizeof(int) * 4, MTL::ResourceStorageModeShared);
    memset(pPartSuppHTBuf->contents(), 0xFF, partsupp_ht_size * sizeof(int) * 4);

    {
        uint pss = (uint)psRows;
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pPartSuppBuildPipe);
        enc->setBuffer(pPsPartKeyBuf, 0, 0);
        enc->setBuffer(pPsSuppKeyBuf, 0, 1);
        enc->setBuffer(pPartSuppHTBuf, 0, 2);
        enc->setBytes(&pss, sizeof(pss), 3);
        enc->setBytes(&partsupp_ht_size, sizeof(partsupp_ht_size), 4);
        NS::UInteger tgs = std::min((NS::UInteger)256, pPartSuppBuildPipe->maxTotalThreadsPerThreadgroup());
        enc->dispatchThreadgroups(MTL::Size((psRows + tgs - 1) / tgs, 1, 1), MTL::Size(tgs, 1, 1));
        enc->endEncoding();
        cb->commit(); cb->waitUntilCompleted();
        totalGpuMs += (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;
    }
    { std::vector<int>().swap(ps_partkey); std::vector<int>().swap(ps_suppkey); }
    // Keep ps_supplycost — needed by probe kernel
    pPsPartKeyBuf->release(); pPsSuppKeyBuf->release();
    printf("  PartSupp HT built (ht_size=%u)\n", partsupp_ht_size);

    printf("Phase 1 complete. GPU HTs resident: bitmap(%.1f MB) + suppmap(%.1f MB) + orders_ht(%.1f MB) + partsupp_ht(%.1f MB)\n",
           part_bitmap_ints * 4.0 / (1024*1024), supp_map_size * 4.0 / (1024*1024),
           orders_ht_size * 8.0 / (1024*1024), partsupp_ht_size * 16.0 / (1024*1024));

    // ══════════════════════════════════════════════════════════════
    // PHASE 2: Stream lineitem in chunks through Probe + Merge
    // ══════════════════════════════════════════════════════════════
    printf("\n--- Phase 2: Stream lineitem (%zu rows) ---\n", liRows);

    // Per-chunk lineitem columns: partkey, suppkey, orderkey, quantity, extprice, discount
    // ~28 bytes/row → use adaptive chunk size
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 28, liRows);
    size_t numChunks = (liRows + chunkRows - 1) / chunkRows;
    printf("Chunk size: %zu rows, %zu chunks\n", chunkRows, numChunks);

    // Allocate double-buffered lineitem chunk GPU buffers
    const int Q9_NUM_SLOTS = 2;
    struct Q9LiSlot {
        MTL::Buffer* partKey; MTL::Buffer* suppKey; MTL::Buffer* ordKey;
        MTL::Buffer* qty; MTL::Buffer* price; MTL::Buffer* disc;
    };
    Q9LiSlot liSlots[Q9_NUM_SLOTS];
    for (int s = 0; s < Q9_NUM_SLOTS; s++) {
        liSlots[s].partKey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        liSlots[s].suppKey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        liSlots[s].ordKey  = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        liSlots[s].qty     = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        liSlots[s].price   = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
        liSlots[s].disc    = device->newBuffer(chunkRows * sizeof(float), MTL::ResourceStorageModeShared);
    }

    // Intermediate + final aggregation buffers (persistent across chunks)
    const uint num_threadgroups = 2048, local_ht_size = 256;
    const uint intermediate_size = num_threadgroups * local_ht_size;
    MTL::Buffer* pIntermediateBuf = device->newBuffer(intermediate_size * sizeof(Q9Aggregates_CPU), MTL::ResourceStorageModeShared);
    const uint final_ht_size = 25 * 10; // 25 nations * ~10 years
    MTL::Buffer* pFinalHTBuf = device->newBuffer(final_ht_size * sizeof(Q9Aggregates_CPU), MTL::ResourceStorageModeShared);
    memset(pFinalHTBuf->contents(), 0, final_ht_size * sizeof(Q9Aggregates_CPU));

    // --- Double-buffered pipeline: overlap CPU parse of chunk N+1 with GPU exec of chunk N ---

    // Pre-parse first chunk into slot 0
    {
        uint firstChunk = (uint)std::min(chunkRows, liRows);
        Q9LiSlot& slot = liSlots[0];
        t0 = std::chrono::high_resolution_clock::now();
        parseIntColumnChunk(liFile, liIdx, 0, firstChunk, 1, (int*)slot.partKey->contents());
        parseIntColumnChunk(liFile, liIdx, 0, firstChunk, 2, (int*)slot.suppKey->contents());
        parseIntColumnChunk(liFile, liIdx, 0, firstChunk, 0, (int*)slot.ordKey->contents());
        parseFloatColumnChunk(liFile, liIdx, 0, firstChunk, 4, (float*)slot.qty->contents());
        parseFloatColumnChunk(liFile, liIdx, 0, firstChunk, 5, (float*)slot.price->contents());
        parseFloatColumnChunk(liFile, liIdx, 0, firstChunk, 6, (float*)slot.disc->contents());
        t1 = std::chrono::high_resolution_clock::now();
        totalCpuParseMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    for (size_t c = 0; c < numChunks; c++) {
        size_t start = c * chunkRows;
        size_t end = std::min(start + chunkRows, liRows);
        uint thisChunk = (uint)(end - start);
        // Reset intermediate buffer for this chunk
        memset(pIntermediateBuf->contents(), 0, intermediate_size * sizeof(Q9Aggregates_CPU));

        // Dispatch probe + merge on current slot (data already parsed)
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();

        // Probe + local agg
        enc->setComputePipelineState(pProbeAggPipe);
        enc->setBuffer(liSlots[c % Q9_NUM_SLOTS].suppKey, 0, 0);
        enc->setBuffer(liSlots[c % Q9_NUM_SLOTS].partKey, 0, 1);
        enc->setBuffer(liSlots[c % Q9_NUM_SLOTS].ordKey, 0, 2);
        enc->setBuffer(liSlots[c % Q9_NUM_SLOTS].price, 0, 3);
        enc->setBuffer(liSlots[c % Q9_NUM_SLOTS].disc, 0, 4);
        enc->setBuffer(liSlots[c % Q9_NUM_SLOTS].qty, 0, 5);
        enc->setBuffer(pPsSupplyCostBuf, 0, 6);
        enc->setBuffer(pPartBitmapBuf, 0, 7);
        enc->setBuffer(pSuppMapBuf, 0, 8);
        enc->setBuffer(pPartSuppHTBuf, 0, 9);
        enc->setBuffer(pOrdersHTBuf, 0, 10);
        enc->setBuffer(pIntermediateBuf, 0, 11);
        enc->setBytes(&thisChunk, sizeof(thisChunk), 12);
        enc->setBytes(&part_ht_size, sizeof(part_ht_size), 13);
        enc->setBytes(&supplier_ht_size, sizeof(supplier_ht_size), 14);
        enc->setBytes(&partsupp_ht_size, sizeof(partsupp_ht_size), 15);
        enc->setBytes(&orders_ht_size, sizeof(orders_ht_size), 16);
        enc->dispatchThreadgroups(MTL::Size(num_threadgroups, 1, 1), MTL::Size(1024, 1, 1));

        // Merge into final HT
        enc->setComputePipelineState(pMergePipe);
        enc->setBuffer(pIntermediateBuf, 0, 0);
        enc->setBuffer(pFinalHTBuf, 0, 1);
        enc->setBytes(&intermediate_size, sizeof(intermediate_size), 2);
        enc->setBytes(&final_ht_size, sizeof(final_ht_size), 3);
        enc->dispatchThreads(MTL::Size(intermediate_size, 1, 1), MTL::Size(1024, 1, 1));
        enc->endEncoding();

        cb->commit();

        // --- Double-buffer: parse NEXT chunk into alternate slot while GPU runs ---
        if (c + 1 < numChunks) {
            size_t nextStart = (c + 1) * chunkRows;
            size_t nextEnd = std::min(nextStart + chunkRows, liRows);
            uint nextChunk = (uint)(nextEnd - nextStart);
            Q9LiSlot& nextSlot = liSlots[(c + 1) % Q9_NUM_SLOTS];
            t0 = std::chrono::high_resolution_clock::now();
            parseIntColumnChunk(liFile, liIdx, nextStart, nextChunk, 1, (int*)nextSlot.partKey->contents());
            parseIntColumnChunk(liFile, liIdx, nextStart, nextChunk, 2, (int*)nextSlot.suppKey->contents());
            parseIntColumnChunk(liFile, liIdx, nextStart, nextChunk, 0, (int*)nextSlot.ordKey->contents());
            parseFloatColumnChunk(liFile, liIdx, nextStart, nextChunk, 4, (float*)nextSlot.qty->contents());
            parseFloatColumnChunk(liFile, liIdx, nextStart, nextChunk, 5, (float*)nextSlot.price->contents());
            parseFloatColumnChunk(liFile, liIdx, nextStart, nextChunk, 6, (float*)nextSlot.disc->contents());
            t1 = std::chrono::high_resolution_clock::now();
            totalCpuParseMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        // Wait for GPU
        cb->waitUntilCompleted();
        totalGpuMs += (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;

        if ((c + 1) % 10 == 0 || c + 1 == numChunks) {
            printf("  Chunk %zu/%zu done\n", c + 1, numChunks);
        }
    }

    // ══════════════════════════════════════════════════════════════
    // PHASE 3: Read back results, map nation names, sort, print
    // ══════════════════════════════════════════════════════════════
    auto postStart = std::chrono::high_resolution_clock::now();
    Q9Aggregates_CPU* results = (Q9Aggregates_CPU*)pFinalHTBuf->contents();
    std::vector<Q9Result> final_results;
    for (uint i = 0; i < final_ht_size; ++i) {
        if (results[i].key != 0) {
            int nationkey = (results[i].key >> 16) & 0xFFFF;
            int year = results[i].key & 0xFFFF;
            final_results.push_back({nationkey, year, results[i].profit});
        }
    }
    std::sort(final_results.begin(), final_results.end(), [](const Q9Result& a, const Q9Result& b) {
        if (a.nationkey != b.nationkey) return a.nationkey < b.nationkey;
        return a.year > b.year;
    });

    printf("\nTPC-H Query 9 Results (Top 15):\n");
    printf("+------------+------+---------------+\n");
    printf("| Nation     | Year |        Profit |\n");
    printf("+------------+------+---------------+\n");
    for (size_t i = 0; i < 15 && i < final_results.size(); ++i) {
        printf("| %-10s | %4d | $%13.2f |\n",
               nation_names[final_results[i].nationkey].c_str(), final_results[i].year, final_results[i].profit);
    }
    printf("+------------+------+---------------+\n");
    printf("Total results found: %lu\n", final_results.size());

    // Comparable view: aggregate by year
    std::map<int, double> year_totals;
    for (const auto& r : final_results) year_totals[r.year] += (double)r.profit;
    printf("\nComparable TPC-H Q9 (yearly sum_profit):\n");
    printf("+--------+---------------+\n");
    printf("| o_year |   sum_profit  |\n");
    printf("+--------+---------------+\n");
    for (const auto& kv : year_totals) printf("| %6d | %13.4f |\n", kv.first, kv.second);
    printf("+--------+---------------+\n");

    auto postEnd = std::chrono::high_resolution_clock::now();
    totalCpuPostMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    double totalExecMs = totalGpuMs + totalCpuPostMs;

    double allCpuParseMs = indexBuildMs + totalCpuParseMs;
    printf("\nSF100 Q9 | %zu chunks | %zu rows\n", numChunks, liRows);
    printf("  CPU Parsing (.tbl): %10.2f ms\n", allCpuParseMs);
    printf("  GPU Execution:      %10.2f ms\n", totalGpuMs);
    printf("  CPU Post Process:   %10.2f ms\n", totalCpuPostMs);
    printf("  Total Execution:    %10.2f ms  (GPU + CPU post)\n", totalExecMs);

    // Cleanup
    pPartBuildFn->release(); pPartBuildPipe->release();
    pSuppBuildFn->release(); pSuppBuildPipe->release();
    pPartSuppBuildFn->release(); pPartSuppBuildPipe->release();
    pOrdersBuildFn->release(); pOrdersBuildPipe->release();
    pProbeAggFn->release(); pProbeAggPipe->release();
    pMergeFn->release(); pMergePipe->release();
    pPartBitmapBuf->release(); pSuppMapBuf->release();
    pOrdersHTBuf->release(); pPartSuppHTBuf->release();
    pPsSupplyCostBuf->release();
    for (int s = 0; s < Q9_NUM_SLOTS; s++) {
        liSlots[s].partKey->release(); liSlots[s].suppKey->release(); liSlots[s].ordKey->release();
        liSlots[s].qty->release(); liSlots[s].price->release(); liSlots[s].disc->release();
    }
    pIntermediateBuf->release(); pFinalHTBuf->release();
}


// ===================================================================
// SF100 Q13 — Chunked Streaming for Orders Table
// ===================================================================
void runQ13BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library) {
    std::cout << "\n=== Running TPC-H Q13 Benchmark (SF100 Chunked) ===" << std::endl;

    MappedFile custFile, ordFile;
    if (!custFile.open(g_dataset_path + "customer.tbl") || !ordFile.open(g_dataset_path + "orders.tbl")) {
        std::cerr << "Q13 SF100: Cannot open files" << std::endl;
        return;
    }

    auto idxT0 = std::chrono::high_resolution_clock::now();
    auto custIdx = buildLineIndex(custFile);
    auto ordIdx = buildLineIndex(ordFile);
    auto idxT1 = std::chrono::high_resolution_clock::now();
    double indexBuildMs = std::chrono::duration<double, std::milli>(idxT1 - idxT0).count();
    size_t custRows = custIdx.size(), ordRows = ordIdx.size();
    printf("Q13 SF100: customer=%zu, orders=%zu (index %.1f ms)\n", custRows, ordRows, indexBuildMs);

    // Timing accumulators
    double totalCpuParseMs = 0, totalGpuMs = 0, totalCpuPostMs = 0;

    // Output: per-customer order counts (persistent across chunks, atomics accumulate)
    MTL::Buffer* pCountsBuf = device->newBuffer(custRows * sizeof(uint), MTL::ResourceStorageModeShared);
    memset(pCountsBuf->contents(), 0, custRows * sizeof(uint));

    NS::Error* pError = nullptr;
    MTL::Function* pFn = library->newFunction(NS::String::string("q13_fused_direct_count_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pPipe = device->newComputePipelineState(pFn, &pError);
    if (!pPipe) { std::cerr << "Q13 SF100: pipeline creation failed" << std::endl; return; }

    MTL::Function* pHistFn = library->newFunction(NS::String::string("q13_build_histogram_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pHistPipe = device->newComputePipelineState(pHistFn, &pError);

    // Histogram buffer for GPU histogram kernel
    const uint hist_max_bins = 256;
    MTL::Buffer* pHistogramBuf = device->newBuffer(hist_max_bins * sizeof(uint), MTL::ResourceStorageModeShared);

    // Stream orders in chunks: custkey(4 bytes) + comment(100 bytes) = 104 bytes/row
    size_t chunkRows = ChunkConfig::adaptiveChunkSize(device, 104, ordRows);
    size_t numChunks = (ordRows + chunkRows - 1) / chunkRows;
    printf("Q13 chunk size: %zu rows, %zu chunks\n", chunkRows, numChunks);

    // Allocate double-buffered chunk buffers for custkey + comment
    const int Q13_NUM_SLOTS = 2;
    struct Q13ChunkSlot { MTL::Buffer* custKey; MTL::Buffer* comment; };
    Q13ChunkSlot q13Slots[Q13_NUM_SLOTS];
    for (int s = 0; s < Q13_NUM_SLOTS; s++) {
        q13Slots[s].custKey = device->newBuffer(chunkRows * sizeof(int), MTL::ResourceStorageModeShared);
        q13Slots[s].comment = device->newBuffer(chunkRows * 100 * sizeof(char), MTL::ResourceStorageModeShared);
    }

    const uint num_threadgroups = 2048;
    uint cs = (uint)custRows;

    // --- Double-buffered pipeline: overlap CPU parse of chunk N+1 with GPU exec of chunk N ---

    // Pre-parse first chunk into slot 0
    {
        uint firstChunk = (uint)std::min(chunkRows, ordRows);
        auto t0 = std::chrono::high_resolution_clock::now();
        parseIntColumnChunk(ordFile, ordIdx, 0, firstChunk, 1, (int*)q13Slots[0].custKey->contents());
        parseCharColumnChunkFixed(ordFile, ordIdx, 0, firstChunk, 8, 100, (char*)q13Slots[0].comment->contents());
        auto t1 = std::chrono::high_resolution_clock::now();
        totalCpuParseMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    for (size_t c = 0; c < numChunks; c++) {
        size_t start = c * chunkRows;
        size_t end = std::min(start + chunkRows, ordRows);
        uint thisChunk = (uint)(end - start);
        Q13ChunkSlot& slot = q13Slots[c % Q13_NUM_SLOTS];

        // Dispatch kernel on current slot (data already parsed)
        MTL::CommandBuffer* cb = commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pPipe);
        enc->setBuffer(slot.custKey, 0, 0);
        enc->setBuffer(slot.comment, 0, 1);
        enc->setBuffer(pCountsBuf, 0, 2);
        enc->setBytes(&thisChunk, sizeof(thisChunk), 3);
        enc->setBytes(&cs, sizeof(cs), 4);
        enc->dispatchThreadgroups(MTL::Size(num_threadgroups, 1, 1), MTL::Size(1024, 1, 1));
        enc->endEncoding();

        cb->commit();

        // --- Double-buffer: parse NEXT chunk into alternate slot while GPU runs ---
        if (c + 1 < numChunks) {
            size_t nextStart = (c + 1) * chunkRows;
            size_t nextEnd = std::min(nextStart + chunkRows, ordRows);
            uint nextChunk = (uint)(nextEnd - nextStart);
            Q13ChunkSlot& nextSlot = q13Slots[(c + 1) % Q13_NUM_SLOTS];
            auto t0 = std::chrono::high_resolution_clock::now();
            parseIntColumnChunk(ordFile, ordIdx, nextStart, nextChunk, 1, (int*)nextSlot.custKey->contents());
            parseCharColumnChunkFixed(ordFile, ordIdx, nextStart, nextChunk, 8, 100, (char*)nextSlot.comment->contents());
            auto t1 = std::chrono::high_resolution_clock::now();
            totalCpuParseMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        // Wait for GPU
        cb->waitUntilCompleted();
        totalGpuMs += (cb->GPUEndTime() - cb->GPUStartTime()) * 1000.0;

        if ((c + 1) % 5 == 0 || c + 1 == numChunks) {
            printf("  Chunk %zu/%zu done\n", c + 1, numChunks);
        }
    }

    // GPU post-processing: dispatch histogram kernel on per-customer counts
    std::memset(pHistogramBuf->contents(), 0, hist_max_bins * sizeof(uint));
    uint cs_all = (uint)custRows;
    MTL::CommandBuffer* histCb = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* histEnc = histCb->computeCommandEncoder();
    histEnc->setComputePipelineState(pHistPipe);
    histEnc->setBuffer(pCountsBuf, 0, 0);
    histEnc->setBuffer(pHistogramBuf, 0, 1);
    histEnc->setBytes(&cs_all, sizeof(cs_all), 2);
    histEnc->setBytes(&hist_max_bins, sizeof(hist_max_bins), 3);
    histEnc->dispatchThreadgroups(MTL::Size(num_threadgroups, 1, 1), MTL::Size(1024, 1, 1));
    histEnc->endEncoding();
    histCb->commit();
    histCb->waitUntilCompleted();
    totalGpuMs += (histCb->GPUEndTime() - histCb->GPUStartTime()) * 1000.0;

    // CPU post-processing: read back histogram from GPU + sort
    auto postStart = std::chrono::high_resolution_clock::now();
    uint* hist = (uint*)pHistogramBuf->contents();
    std::vector<Q13Result> final_results;
    for (uint i = 0; i < hist_max_bins; i++) {
        if (hist[i] > 0) {
            final_results.push_back({i, hist[i]});
        }
    }
    std::sort(final_results.begin(), final_results.end(), [](const Q13Result& a, const Q13Result& b) {
        if (a.custdist != b.custdist) return a.custdist > b.custdist;
        return a.c_count > b.c_count;
    });

    printf("\nTPC-H Query 13 Results (Comparable histogram):\n");
    printf("+---------+----------+\n");
    printf("| c_count | custdist |\n");
    printf("+---------+----------+\n");
    for (const auto& res : final_results) {
        printf("| %7u | %8u |\n", res.c_count, res.custdist);
    }
    printf("+---------+----------+\n");

    auto postEnd = std::chrono::high_resolution_clock::now();
    totalCpuPostMs = std::chrono::duration<double, std::milli>(postEnd - postStart).count();

    double totalExecMs = totalGpuMs + totalCpuPostMs;

    double allCpuParseMs = indexBuildMs + totalCpuParseMs;
    printf("\nSF100 Q13 | %zu chunks | %zu rows\n", numChunks, ordRows);
    printf("  CPU Parsing (.tbl): %10.2f ms\n", allCpuParseMs);
    printf("  GPU Execution:      %10.2f ms\n", totalGpuMs);
    printf("  CPU Post Process:   %10.2f ms\n", totalCpuPostMs);
    printf("  Total Execution:    %10.2f ms  (GPU + CPU post)\n", totalExecMs);

    // Cleanup
    pFn->release(); pPipe->release();
    pHistFn->release(); pHistPipe->release();
    for (int s = 0; s < Q13_NUM_SLOTS; s++) {
        q13Slots[s].custKey->release(); q13Slots[s].comment->release();
    }
    pCountsBuf->release();
    pHistogramBuf->release();
}


void showHelp() {
    std::cout << "GPU Database Metal Benchmark" << std::endl;
    std::cout << "Usage: GPUDBMetalBenchmark [sf1|sf10|sf100] [query]" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "Available queries:" << std::endl;
    std::cout << "  all           - Run all benchmarks (default)" << std::endl;
    std::cout << "  selection     - Run selection benchmark" << std::endl;
    std::cout << "  aggregation   - Run aggregation benchmark" << std::endl;
    std::cout << "  join          - Run join benchmark" << std::endl;
    std::cout << "  q1            - Run TPC-H Query 1 (Pricing Summary Report)" << std::endl;
    std::cout << "  q3            - Run TPC-H Query 3 (Shipping Priority)" << std::endl;
    std::cout << "  q6            - Run TPC-H Query 6 (Forecasting Revenue Change)" << std::endl;
    std::cout << "  q9            - Run TPC-H Query 9 (Product Type Profit Measure)" << std::endl;
    std::cout << "  q13           - Run TPC-H Query 13 (Customer Distribution)" << std::endl;
    std::cout << "  help          - Show this help message" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "Scale Factors:" << std::endl;
    std::cout << "  sf1           - TPC-H SF-1 (~6M lineitem rows)" << std::endl;
    std::cout << "  sf10          - TPC-H SF-10 (~60M lineitem rows)" << std::endl;
    std::cout << "  sf100         - TPC-H SF-100 (~600M lineitem rows, chunked streaming)" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  GPUDBMetalBenchmark        # Run all benchmarks on SF-1" << std::endl;
    std::cout << "  GPUDBMetalBenchmark q1     # Run only TPC-H Query 1" << std::endl;
    std::cout << "  GPUDBMetalBenchmark sf100 q1  # Run Q1 on SF-100 (chunked)" << std::endl;
    std::cout << "  GPUDBMetalBenchmark sf100 q6  # Run Q6 on SF-100 (chunked)" << std::endl;
}

// --- Main Entry Point ---
int main(int argc, const char * argv[]) {
    // Parse command line arguments
    // Supports either:
    //   GPUDBMetalBenchmark q13
    //   GPUDBMetalBenchmark sf10 q13
    //   GPUDBMetalBenchmark q13 sf10
    //   GPUDBMetalBenchmark sf100 q1   (chunked streaming mode)
    std::string query = "all"; // default to running all benchmarks
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "help" || arg == "--help" || arg == "-h") {
            showHelp();
            return 0;
        }
        if (arg == "sf1") {
            g_dataset_path = "data/SF-1/";
            g_scale_factor = 1;
            g_sf100_mode = false;
            continue;
        }
        if (arg == "sf10") {
            g_dataset_path = "data/SF-10/";
            g_scale_factor = 10;
            g_sf100_mode = false;
            continue;
        }
        if (arg == "sf100") {
            g_dataset_path = "data/SF-100/";
            g_scale_factor = 100;
            g_sf100_mode = true;
            continue;
        }
        // Otherwise treat as the query selector.
        query = arg;
    }

    NS::AutoreleasePool* pAutoreleasePool = NS::AutoreleasePool::alloc()->init();
    
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    // Hint Metal to compile pipelines more aggressively in parallel
    if (device) {
        device->setShouldMaximizeConcurrentCompilation(true);
    }

    // Print GPU info for SF100 mode
    if (g_sf100_mode) {
        std::cout << "=== SF100 Chunked Streaming Mode ===" << std::endl;
        std::cout << "GPU: " << device->name()->utf8String() << std::endl;
        printf("Max Working Set: %llu MB\n", (unsigned long long)(device->recommendedMaxWorkingSetSize() / (1024*1024)));
        printf("Current Allocated: %llu MB\n", (unsigned long long)(device->currentAllocatedSize() / (1024*1024)));
        std::cout << "Data path: " << g_dataset_path << std::endl;
    }

    MTL::CommandQueue* commandQueue = device->newCommandQueue();
    
    NS::Error* error = nullptr;
    MTL::Library* library = device->newDefaultLibrary();
    if (!library) {
        // Try to load from specific path
        NS::String* libraryPath = NS::String::string("default.metallib", NS::UTF8StringEncoding);
        library = device->newLibrary(libraryPath, &error);
        libraryPath->release();
    }
    if (!library) {
        // Compile from source at runtime
        std::ifstream metalFile("kernels/DatabaseKernels.metal");
        if (metalFile.is_open()) {
            std::string metalSource((std::istreambuf_iterator<char>(metalFile)), std::istreambuf_iterator<char>());
            metalFile.close();
            NS::String* sourceStr = NS::String::string(metalSource.c_str(), NS::UTF8StringEncoding);
            MTL::CompileOptions* opts = MTL::CompileOptions::alloc()->init();
            library = device->newLibrary(sourceStr, opts, &error);
            opts->release();
            if (library) {
                std::cout << "Metal library compiled from source at runtime" << std::endl;
            }
        }
    }
    if (!library) {
        std::cerr << "Error loading .metal library from default, file path, and source" << std::endl;
        if (error) {
            std::cerr << "Error details: " << error->localizedDescription()->utf8String() << std::endl;
        }
        pAutoreleasePool->release();
        return 1;
    }

    // SF100 mode: use chunked execution paths
    if (g_sf100_mode) {
        if (query == "all") {
            runQ1BenchmarkSF100(device, commandQueue, library);
            runQ6BenchmarkSF100(device, commandQueue, library);
            runQ3BenchmarkSF100(device, commandQueue, library);
            runQ9BenchmarkSF100(device, commandQueue, library);
            runQ13BenchmarkSF100(device, commandQueue, library);
        } else if (query == "q1") {
            runQ1BenchmarkSF100(device, commandQueue, library);
        } else if (query == "q3") {
            runQ3BenchmarkSF100(device, commandQueue, library);
        } else if (query == "q6") {
            runQ6BenchmarkSF100(device, commandQueue, library);
        } else if (query == "q9") {
            runQ9BenchmarkSF100(device, commandQueue, library);
        } else if (query == "q13") {
            runQ13BenchmarkSF100(device, commandQueue, library);
        } else {
            std::cerr << "SF100 mode supports: q1, q3, q6, q9, q13, all" << std::endl;
            return 1;
        }
    } else {
        // Standard SF1/SF10 mode
        if (query == "all") {
            runSelectionBenchmark(device, commandQueue, library);
            runAggregationBenchmark(device, commandQueue, library);
            runJoinBenchmark(device, commandQueue, library);
            runQ1Benchmark(device, commandQueue, library);
            runQ3Benchmark(device, commandQueue, library);
            runQ6Benchmark(device, commandQueue, library);
            runQ9Benchmark(device, commandQueue, library);
            runQ13Benchmark(device, commandQueue, library);
        } else if (query == "selection") {
            runSelectionBenchmark(device, commandQueue, library);
        } else if (query == "aggregation") {
            runAggregationBenchmark(device, commandQueue, library);
        } else if (query == "join") {
            runJoinBenchmark(device, commandQueue, library);
        } else if (query == "q1") {
            runQ1Benchmark(device, commandQueue, library);
        } else if (query == "q3") {
            runQ3Benchmark(device, commandQueue, library);
        } else if (query == "q6") {
            runQ6Benchmark(device, commandQueue, library);
        } else if (query == "q9") {
            runQ9Benchmark(device, commandQueue, library);
        } else if (query == "q13") {
            runQ13Benchmark(device, commandQueue, library);
        } else {
            std::cerr << "Unknown query: " << query << std::endl;
            std::cerr << "Use 'help' to see available options." << std::endl;
            return 1;
        }
    }
    
    return 0;
}
