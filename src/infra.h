#pragma once

// Metal-cpp headers (no PRIVATE_IMPLEMENTATION here — that lives in infra.cpp)
#include "Metal/Metal.hpp"
#include "Foundation/Foundation.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <functional>
#include <span>

// mmap for SF100 chunked streaming
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// Round up to next power of 2 (host-side, must match GPU-side next_pow2)
inline uint nextPow2(uint v) {
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}

// Global dataset configuration
extern std::string g_dataset_path;
extern bool g_sf100_mode;

// Current query / scale-factor labels — set by main.cpp before dispatching each
// benchmark so printTimingSummary() can emit a machine-readable TIMING_CSV line.
extern std::string g_current_query;
extern std::string g_current_sf;

// ===================================================================
// DETAILED TIMING — per-query breakdown accumulated while the benchmark runs
// ===================================================================
// Shape mirrors GPUDBMetalCodeGen/src/infra.h::DetailedTiming so the same
// log-parsing / CSV-analysis scripts work across both projects. LC has no
// analyze/plan/codegen/compile phases (queries are hand-written C++), so
// those fields stay at 0. LC-specific helpers (createPipeline,
// createTimedSharedBuffer, recordKernelTime) auto-accumulate psoMs /
// bufferAllocMs / phaseKernelMs without queries needing to opt in.
struct DetailedTiming {
    std::string queryName;
    std::string scaleFactor;

    // Compile-overhead phases (all zero for LC: no codegen pipeline).
    double analyzeMs      = 0.0;
    double planMs         = 0.0;
    double codegenMs      = 0.0;
    double compileMs      = 0.0;
    double psoMs          = 0.0;   // Metal PSO creation (LC's only compile-ish cost)

    // Data-load phase.
    double dataLoadMs     = 0.0;   // query-observed read of .tbl/.colbin
    std::string loadSource = "none"; // "colbin", "tbl", "mixed", or "none"
    size_t      loadBytes  = 0;
    double ingestMs       = 0.0;   // one-time .tbl->column ingest (excluded from e2e)

    // Query-execution phase.
    double bufferAllocMs  = 0.0;
    double gpuTotalMs     = 0.0;
    double postMs         = 0.0;
    std::vector<std::pair<std::string, double>> phaseKernelMs;

    void reset() {
        queryName.clear(); scaleFactor.clear();
        analyzeMs = planMs = codegenMs = compileMs = psoMs = 0.0;
        dataLoadMs = ingestMs = 0.0;
        loadSource = "none"; loadBytes = 0;
        bufferAllocMs = gpuTotalMs = postMs = 0.0;
        phaseKernelMs.clear();
    }
};

// Global accumulator — main.cpp calls g_detailed_timing.reset() before each
// benchmark. Helpers (createPipeline, createTimedSharedBuffer, addKernelTime)
// add to it automatically, so individual queries don't need to thread a
// pointer through their code.
extern DetailedTiming g_detailed_timing;

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
inline size_t countLines(const MappedFile& mf) {
    size_t count = 0;
    const char* p = (const char*)mf.data;
    const char* end = p + mf.size;
    while (p < end) { if (*p++ == '\n') count++; }
    return count;
}

// Build line offset index for random access by row number
inline std::vector<size_t> buildLineIndex(const MappedFile& mf) {
    std::vector<size_t> offsets;
    offsets.reserve(countLines(mf) + 1); // pre-size to avoid repeated reallocations
    offsets.push_back(0);
    const char* p = (const char*)mf.data;
    for (size_t i = 0; i < mf.size; i++) {
        if (p[i] == '\n' && i + 1 < mf.size) offsets.push_back(i + 1);
    }
    return offsets;
}

// Generic column chunk parser: walks lines, finds column, calls extract callback
template<typename ExtractFn>
inline size_t parseColumnChunkGeneric(const MappedFile& mf, const std::vector<size_t>& lineIndex,
                                       size_t startRow, size_t rowCount, int columnIndex, ExtractFn extract) {
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
            if (col == columnIndex) { extract(start, end, parsed++); break; }
            col++;
            if (end >= fileEnd) break;
            start = end + 1;
        }
    }
    return parsed;
}

inline size_t parseIntColumnChunk(const MappedFile& mf, const std::vector<size_t>& lineIndex,
                                   size_t startRow, size_t rowCount, int columnIndex, int* output) {
    return parseColumnChunkGeneric(mf, lineIndex, startRow, rowCount, columnIndex,
        [output](const char* s, const char*, size_t i) { output[i] = atoi(s); });
}

inline size_t parseFloatColumnChunk(const MappedFile& mf, const std::vector<size_t>& lineIndex,
                                     size_t startRow, size_t rowCount, int columnIndex, float* output) {
    return parseColumnChunkGeneric(mf, lineIndex, startRow, rowCount, columnIndex,
        [output](const char* s, const char*, size_t i) { output[i] = strtof(s, nullptr); });
}

inline size_t parseDateColumnChunk(const MappedFile& mf, const std::vector<size_t>& lineIndex,
                                    size_t startRow, size_t rowCount, int columnIndex, int* output) {
    return parseColumnChunkGeneric(mf, lineIndex, startRow, rowCount, columnIndex,
        [output](const char* s, const char* e, size_t i) {
            int y = 0, m = 0, d = 0; const char* p = s;
            while (p < e && *p >= '0' && *p <= '9') { y = y * 10 + (*p - '0'); p++; }
            if (p < e) p++;
            while (p < e && *p >= '0' && *p <= '9') { m = m * 10 + (*p - '0'); p++; }
            if (p < e) p++;
            while (p < e && *p >= '0' && *p <= '9') { d = d * 10 + (*p - '0'); p++; }
            output[i] = y * 10000 + m * 100 + d;
        });
}

inline size_t parseCharColumnChunk(const MappedFile& mf, const std::vector<size_t>& lineIndex,
                                    size_t startRow, size_t rowCount, int columnIndex, char* output) {
    return parseColumnChunkGeneric(mf, lineIndex, startRow, rowCount, columnIndex,
        [output](const char* s, const char* e, size_t i) { output[i] = (s < e) ? *s : '\0'; });
}

inline size_t parseCharColumnChunkFixed(const MappedFile& mf, const std::vector<size_t>& lineIndex,
                                        size_t startRow, size_t rowCount, int columnIndex,
                                        int fixedWidth, char* output) {
    return parseColumnChunkGeneric(mf, lineIndex, startRow, rowCount, columnIndex,
        [output, fixedWidth](const char* s, const char* e, size_t i) {
            int len = (int)(e - s), cp = len < fixedWidth ? len : fixedWidth;
            char* dst = output + i * fixedWidth;
            memcpy(dst, s, cp); memset(dst + cp, '\0', fixedWidth - cp);
        });
}

// ===================================================================
// REUSABLE HELPERS
// ===================================================================

// --- Metal Pipeline Creation ---
// Auto-accumulates elapsed time into g_detailed_timing.psoMs so detailed
// timing works across all 22 queries without per-query instrumentation.
inline MTL::ComputePipelineState* createPipeline(MTL::Device* device, MTL::Library* library, const char* name) {
    auto t0 = std::chrono::high_resolution_clock::now();
    NS::Error* error = nullptr;
    auto fn = library->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
    if (!fn) { std::cerr << "Kernel not found: " << name << std::endl; return nullptr; }
    auto pso = device->newComputePipelineState(fn, &error);
    fn->release();
    if (!pso) { std::cerr << "Failed to create pipeline: " << name << std::endl; }
    auto t1 = std::chrono::high_resolution_clock::now();
    g_detailed_timing.psoMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
    return pso;
}

// --- Opt-in buffer-allocation timer ---
// Wrapper around device->newBuffer that accumulates time into
// g_detailed_timing.bufferAllocMs. Use in place of device->newBuffer(...)
// when you want that phase tracked. Existing raw newBuffer calls still work.
inline MTL::Buffer* createTimedSharedBuffer(MTL::Device* device, const void* data, size_t bytes) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto buf = data
        ? device->newBuffer(data, bytes, MTL::ResourceStorageModeShared)
        : device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    auto t1 = std::chrono::high_resolution_clock::now();
    g_detailed_timing.bufferAllocMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
    return buf;
}
inline MTL::Buffer* createTimedSharedBuffer(MTL::Device* device, size_t bytes) {
    return createTimedSharedBuffer(device, nullptr, bytes);
}

// --- Per-kernel breakdown helper ---
// Record a named kernel's elapsed GPU time into phaseKernelMs.
inline void recordKernelTime(const std::string& name, double ms) {
    g_detailed_timing.phaseKernelMs.emplace_back(name, ms);
}

// --- Variadic Release ---
template<typename... Args> void releaseAll(Args*... args) { (args->release(), ...); }

// --- String Trimming ---
inline std::string trimFixed(const char* chars, size_t index, int width) {
    std::string s(chars + index * width, width);
    s.erase(s.find_last_not_of(std::string("\0 ", 2)) + 1);
    return s;
}

// --- Nation/Region Utilities ---
inline int findRegionKey(const std::vector<int>& regionkeys, const char* name_chars,
                         int width, const std::string& target) {
    for (size_t i = 0; i < regionkeys.size(); i++) {
        if (trimFixed(name_chars, i, width) == target) return regionkeys[i];
    }
    return -1;
}

inline std::map<int, std::string> buildNationNames(const std::vector<int>& nationkeys,
                                                    const char* name_chars, int width) {
    std::map<int, std::string> names;
    for (size_t i = 0; i < nationkeys.size(); i++)
        names[nationkeys[i]] = trimFixed(name_chars, i, width);
    return names;
}

inline std::vector<int> filterNationsByRegion(const std::vector<int>& nationkeys,
                                               const std::vector<int>& regionkeys, int target_regionkey) {
    std::vector<int> result;
    for (size_t i = 0; i < nationkeys.size(); i++)
        if (regionkeys[i] == target_regionkey) result.push_back(nationkeys[i]);
    return result;
}

inline uint buildNationBitmap(const std::vector<int>& nationkeys,
                               const std::vector<int>& regionkeys, int target_regionkey) {
    uint bitmap = 0;
    for (size_t i = 0; i < nationkeys.size(); i++)
        if (regionkeys[i] == target_regionkey) bitmap |= (1u << nationkeys[i]);
    return bitmap;
}

// ===================================================================
// MULTI-COLUMN SINGLE-PASS LOADER (for SF1/SF10)
// ===================================================================

enum class ColType { INT, FLOAT, DATE, CHAR1, CHAR_FIXED };

struct ColSpec {
    int          columnIndex;
    ColType      type;
    int          fixedWidth;  // only for CHAR_FIXED
    
    ColSpec(int idx, ColType t, int fw = 0) : columnIndex(idx), type(t), fixedWidth(fw) {}
};

struct LoadedColumns {
    std::vector<std::vector<int>>   intCols;
    std::vector<std::vector<float>> floatCols;
    std::vector<std::vector<char>>  charCols;
    
    // Maps from (columnIndex, type) → index into intCols/floatCols/charCols
    std::unordered_map<int, size_t> intMap, floatMap, charMap;
    
    const std::vector<int>&   ints(int col)   const { return intCols[intMap.at(col)]; }
    const std::vector<float>& floats(int col) const { return floatCols[floatMap.at(col)]; }
    const std::vector<char>&  chars(int col)  const { return charCols[charMap.at(col)]; }
    
    std::vector<int>&   ints(int col)   { return intCols[intMap.at(col)]; }
    std::vector<float>& floats(int col) { return floatCols[floatMap.at(col)]; }
    std::vector<char>&  chars(int col)  { return charCols[charMap.at(col)]; }
};

inline LoadedColumns loadColumnsMultiText(const std::string& filePath, const std::vector<ColSpec>& specs) {
    LoadedColumns result;
    
    // Pre-allocate storage slots and build maps
    struct ColHandler {
        int    columnIndex;
        ColType type;
        int    fixedWidth;
        size_t storageIdx;
    };
    std::vector<ColHandler> handlers;
    int maxCol = 0;
    
    for (const auto& s : specs) {
        ColHandler h;
        h.columnIndex = s.columnIndex;
        h.type = s.type;
        h.fixedWidth = s.fixedWidth;
        maxCol = std::max(maxCol, s.columnIndex);
        
        switch (s.type) {
            case ColType::INT:
            case ColType::DATE:
                h.storageIdx = result.intCols.size();
                result.intMap[s.columnIndex] = h.storageIdx;
                result.intCols.emplace_back();
                break;
            case ColType::FLOAT:
                h.storageIdx = result.floatCols.size();
                result.floatMap[s.columnIndex] = h.storageIdx;
                result.floatCols.emplace_back();
                break;
            case ColType::CHAR1:
            case ColType::CHAR_FIXED:
                h.storageIdx = result.charCols.size();
                result.charMap[s.columnIndex] = h.storageIdx;
                result.charCols.emplace_back();
                break;
        }
        handlers.push_back(h);
    }
    
    // Build fast lookup: columnIndex → handler index
    std::vector<int> colToHandler(maxCol + 1, -1);
    for (size_t i = 0; i < handlers.size(); i++)
        colToHandler[handlers[i].columnIndex] = (int)i;
    
    std::ifstream file(filePath);
    if (!file.is_open()) { std::cerr << "Error: Could not open file " << filePath << std::endl; return result; }
    
    std::string line;
    while (std::getline(file, line)) {
        int col = 0;
        size_t start = 0;
        size_t end = line.find('|');
        while (end != std::string::npos && col <= maxCol) {
            if (colToHandler[col] >= 0) {
                auto& h = handlers[colToHandler[col]];
                const char* tok = line.c_str() + start;
                size_t tokLen = end - start;
                
                switch (h.type) {
                    case ColType::INT:
                        result.intCols[h.storageIdx].push_back(atoi(tok));
                        break;
                    case ColType::FLOAT:
                        result.floatCols[h.storageIdx].push_back(strtof(tok, nullptr));
                        break;
                    case ColType::DATE: {
                        // Parse YYYY-MM-DD without string copies
                        int y = 0, m = 0, d = 0;
                        const char* p = tok;
                        const char* tokEnd = tok + tokLen;
                        while (p < tokEnd && *p >= '0' && *p <= '9') { y = y * 10 + (*p - '0'); p++; }
                        if (p < tokEnd) p++; // skip '-'
                        while (p < tokEnd && *p >= '0' && *p <= '9') { m = m * 10 + (*p - '0'); p++; }
                        if (p < tokEnd) p++; // skip '-'
                        while (p < tokEnd && *p >= '0' && *p <= '9') { d = d * 10 + (*p - '0'); p++; }
                        result.intCols[h.storageIdx].push_back(y * 10000 + m * 100 + d);
                        break;
                    }
                    case ColType::CHAR1:
                        result.charCols[h.storageIdx].push_back(tokLen > 0 ? tok[0] : '\0');
                        break;
                    case ColType::CHAR_FIXED: {
                        auto& v = result.charCols[h.storageIdx];
                        int cp = (int)tokLen < h.fixedWidth ? (int)tokLen : h.fixedWidth;
                        v.insert(v.end(), tok, tok + cp);
                        v.insert(v.end(), h.fixedWidth - cp, '\0');
                        break;
                    }
                }
            }
            start = end + 1;
            end = line.find('|', start);
            col++;
        }
    }
    return result;
}

// ===================================================================
// LOAD-SOURCE TRACKER (shared with GPUDBMetalCodeGen)
// ===================================================================
// Per-thread counters populated by loadColumnsMultiAuto so the detailed
// timing printout can report whether a query was served from .colbin or .tbl.
// Reset at the start of each query via loadStats().reset().
struct LoadStats {
    size_t bytes       = 0;
    int    tblCalls    = 0;
    int    colbinCalls = 0;
    double excludedMs  = 0.0;   // one-time .tbl -> column ingest, excluded from e2e
    void reset() { bytes = 0; tblCalls = 0; colbinCalls = 0; excludedMs = 0.0; }
    void recordBinary(size_t b) { bytes += b; colbinCalls++; }
    void recordText(size_t b)   { bytes += b; tblCalls++; }
    void recordExcluded(double ms) { excludedMs += ms; }
    std::string source() const {
        if (tblCalls && colbinCalls) return "mixed";
        if (colbinCalls)             return "colbin";
        if (tblCalls)                return "tbl";
        return "none";
    }
};
inline LoadStats& loadStats() { thread_local LoadStats s; return s; }

// Forward decl so the colbin namespace can reference the auto loader.
inline LoadedColumns loadColumnsMultiAuto(const std::string& filePath,
                                           const std::vector<ColSpec>& specs);

// ===================================================================
// COLUMNAR BINARY FORMAT (.colbin)
// -------------------------------------------------------------------
// Layout:
//   [magic 8B "TPCHCB01"]
//   [u32 version][u32 n_cols][u64 n_rows]
//   [u64 source_size][i64 source_mtime_ns][u64 pad]  -> 48 B header
//   n_cols * ColDesc (each 32 B)
//   [payloads, each aligned to 16 KiB so they can be wrapped directly
//    as MTL::Buffer via newBufferWithBytesNoCopy — Apple Silicon page
//    size is 16 KiB, and MTLBuffer requires page-aligned pointers.]
// Version bumps:
//   v1 — 16 B alignment (copy-into-vector path only)
//   v2 — 16 KiB alignment (enables zero-copy MTL::Buffer wrapping)
// ===================================================================
namespace colbin {
static constexpr char     MAGIC[8]   = {'T','P','C','H','C','B','0','1'};
static constexpr uint32_t VERSION    = 2;
static constexpr size_t   ALIGN      = 16384;   // Apple Silicon page size

struct FileHeader {
    char     magic[8];
    uint32_t version;
    uint32_t n_cols;
    uint64_t n_rows;
    uint64_t source_size;
    int64_t  source_mtime_ns;
    uint64_t _pad;
};
static_assert(sizeof(FileHeader) == 48, "FileHeader size drift");

struct ColDesc {
    int32_t  columnIndex;
    uint8_t  dtype;
    uint8_t  _pad0;
    uint16_t fixedWidth;
    uint64_t offset;
    uint64_t size_bytes;
    uint64_t _pad1;
};
static_assert(sizeof(ColDesc) == 32, "ColDesc size drift");

inline uint8_t encodeType(ColType t) {
    switch (t) {
        case ColType::INT:        return 0;
        case ColType::FLOAT:      return 1;
        case ColType::DATE:       return 2;
        case ColType::CHAR1:      return 3;
        case ColType::CHAR_FIXED: return 4;
    }
    return 0;
}
inline ColType decodeType(uint8_t b) {
    switch (b) {
        case 0: return ColType::INT;
        case 1: return ColType::FLOAT;
        case 2: return ColType::DATE;
        case 3: return ColType::CHAR1;
        default: return ColType::CHAR_FIXED;
    }
}

inline std::string binaryPath(const std::string& tblPath) {
    auto pos = tblPath.rfind('.');
    std::string base = (pos == std::string::npos) ? tblPath : tblPath.substr(0, pos);
    return base + ".colbin";
}

inline bool statFile(const std::string& p, size_t& sz, int64_t& mtimeNs) {
    struct stat st{};
    if (::stat(p.c_str(), &st) != 0) return false;
    sz = (size_t)st.st_size;
#ifdef __APPLE__
    mtimeNs = (int64_t)st.st_mtimespec.tv_sec * 1000000000LL + st.st_mtimespec.tv_nsec;
#else
    mtimeNs = (int64_t)st.st_mtim.tv_sec * 1000000000LL + st.st_mtim.tv_nsec;
#endif
    return true;
}

inline bool loadColumnsFromBinary(const std::string& tblPath,
                                  const std::vector<ColSpec>& specs,
                                  LoadedColumns& out)
{
    const std::string cp = binaryPath(tblPath);
    size_t tblSize = 0; int64_t tblMtime = 0;
    if (!statFile(tblPath, tblSize, tblMtime)) return false;

    MappedFile mf;
    if (!mf.open(cp)) return false;
    if (mf.size < sizeof(FileHeader)) return false;

    const char* base = (const char*)mf.data;
    FileHeader hdr;
    memcpy(&hdr, base, sizeof(hdr));
    if (memcmp(hdr.magic, MAGIC, 8) != 0) return false;
    if (hdr.version != VERSION) return false;
    if (hdr.source_size != tblSize) return false;
    if (hdr.source_mtime_ns != tblMtime) return false;
    if (sizeof(FileHeader) + hdr.n_cols * sizeof(ColDesc) > mf.size) return false;

    const ColDesc* descs = (const ColDesc*)(base + sizeof(FileHeader));
    std::unordered_map<int, const ColDesc*> byIdx;
    byIdx.reserve(hdr.n_cols);
    for (uint32_t i = 0; i < hdr.n_cols; i++) byIdx[descs[i].columnIndex] = &descs[i];

    LoadedColumns result;
    const size_t n = hdr.n_rows;
    for (const auto& s : specs) {
        auto it = byIdx.find(s.columnIndex);
        if (it == byIdx.end()) return false;
        const ColDesc* d = it->second;
        if (d->offset + d->size_bytes > mf.size) return false;
        if (decodeType(d->dtype) != s.type) return false;
        if (s.type == ColType::CHAR_FIXED && d->fixedWidth != s.fixedWidth) return false;
        const char* src = base + d->offset;
        switch (s.type) {
            case ColType::INT:
            case ColType::DATE: {
                if (d->size_bytes != n * sizeof(int32_t)) return false;
                size_t idx = result.intCols.size();
                result.intMap[s.columnIndex] = idx;
                result.intCols.emplace_back(n);
                memcpy(result.intCols[idx].data(), src, d->size_bytes);
                break;
            }
            case ColType::FLOAT: {
                if (d->size_bytes != n * sizeof(float)) return false;
                size_t idx = result.floatCols.size();
                result.floatMap[s.columnIndex] = idx;
                result.floatCols.emplace_back(n);
                memcpy(result.floatCols[idx].data(), src, d->size_bytes);
                break;
            }
            case ColType::CHAR1: {
                if (d->size_bytes != n) return false;
                size_t idx = result.charCols.size();
                result.charMap[s.columnIndex] = idx;
                result.charCols.emplace_back(n);
                memcpy(result.charCols[idx].data(), src, d->size_bytes);
                break;
            }
            case ColType::CHAR_FIXED: {
                if (d->size_bytes != n * (size_t)s.fixedWidth) return false;
                size_t idx = result.charCols.size();
                result.charMap[s.columnIndex] = idx;
                result.charCols.emplace_back(d->size_bytes);
                memcpy(result.charCols[idx].data(), src, d->size_bytes);
                break;
            }
        }
    }
    out = std::move(result);
    return true;
}

inline bool writeColbin(const std::string& tblPath,
                         const std::vector<ColSpec>& specs,
                         const LoadedColumns& parsed)
{
    size_t tblSize = 0; int64_t tblMtime = 0;
    if (!statFile(tblPath, tblSize, tblMtime)) return false;
    const std::string cp  = binaryPath(tblPath);
    const std::string tmp = cp + ".tmp";
    if (specs.empty()) return false;

    size_t nRows = 0;
    {
        const auto& s = specs.front();
        switch (s.type) {
            case ColType::INT:
            case ColType::DATE:
                nRows = parsed.ints(s.columnIndex).size(); break;
            case ColType::FLOAT:
                nRows = parsed.floats(s.columnIndex).size(); break;
            case ColType::CHAR1:
                nRows = parsed.chars(s.columnIndex).size(); break;
            case ColType::CHAR_FIXED: {
                const auto& v = parsed.chars(s.columnIndex);
                nRows = s.fixedWidth > 0 ? v.size() / (size_t)s.fixedWidth : v.size();
                break;
            }
        }
    }

    std::vector<ColDesc> descs(specs.size());
    size_t cursor = sizeof(FileHeader) + specs.size() * sizeof(ColDesc);
    cursor = (cursor + ALIGN - 1) & ~(ALIGN - 1);
    for (size_t i = 0; i < specs.size(); i++) {
        const auto& s = specs[i];
        ColDesc& d = descs[i];
        d.columnIndex = s.columnIndex;
        d.dtype       = encodeType(s.type);
        d._pad0       = 0;
        d.fixedWidth  = (uint16_t)s.fixedWidth;
        d._pad1       = 0;
        size_t bytes = 0;
        switch (s.type) {
            case ColType::INT:
            case ColType::DATE:       bytes = nRows * sizeof(int32_t); break;
            case ColType::FLOAT:      bytes = nRows * sizeof(float);   break;
            case ColType::CHAR1:      bytes = nRows;                   break;
            case ColType::CHAR_FIXED: bytes = nRows * (size_t)s.fixedWidth; break;
        }
        d.size_bytes = bytes;
        d.offset     = cursor;
        cursor += bytes;
        cursor = (cursor + ALIGN - 1) & ~(ALIGN - 1);
    }
    size_t total = cursor;

    int fd = ::open(tmp.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
    if (fd < 0) { std::cerr << "colbin: cannot open " << tmp << "\n"; return false; }
    if (::ftruncate(fd, (off_t)total) != 0) {
        std::cerr << "colbin: ftruncate failed on " << tmp << "\n";
        ::close(fd); ::unlink(tmp.c_str()); return false;
    }

    FileHeader hdr{};
    memcpy(hdr.magic, MAGIC, 8);
    hdr.version         = VERSION;
    hdr.n_cols          = (uint32_t)specs.size();
    hdr.n_rows          = nRows;
    hdr.source_size     = tblSize;
    hdr.source_mtime_ns = tblMtime;
    hdr._pad            = 0;
    if (::pwrite(fd, &hdr, sizeof(hdr), 0) != (ssize_t)sizeof(hdr)) {
        ::close(fd); ::unlink(tmp.c_str()); return false;
    }
    if (::pwrite(fd, descs.data(), descs.size() * sizeof(ColDesc), sizeof(FileHeader))
         != (ssize_t)(descs.size() * sizeof(ColDesc))) {
        ::close(fd); ::unlink(tmp.c_str()); return false;
    }

    for (size_t i = 0; i < specs.size(); i++) {
        const auto& s = specs[i];
        const ColDesc& d = descs[i];
        const void* src = nullptr;
        switch (s.type) {
            case ColType::INT:
            case ColType::DATE:
                src = parsed.ints(s.columnIndex).data(); break;
            case ColType::FLOAT:
                src = parsed.floats(s.columnIndex).data(); break;
            case ColType::CHAR1:
            case ColType::CHAR_FIXED:
                src = parsed.chars(s.columnIndex).data(); break;
        }
        size_t remaining = d.size_bytes;
        off_t off = (off_t)d.offset;
        const char* p = (const char*)src;
        while (remaining > 0) {
            size_t chunk = remaining > (size_t)(1u << 30) ? (size_t)(1u << 30) : remaining;
            ssize_t w = ::pwrite(fd, p, chunk, off);
            if (w <= 0) {
                std::cerr << "colbin: payload write failed at col " << s.columnIndex << "\n";
                ::close(fd); ::unlink(tmp.c_str()); return false;
            }
            p += w; off += w; remaining -= (size_t)w;
        }
    }

    ::fsync(fd);
    ::close(fd);
    if (::rename(tmp.c_str(), cp.c_str()) != 0) {
        ::unlink(tmp.c_str()); return false;
    }
    return true;
}

} // namespace colbin

// ===================================================================
// ZERO-COPY COLUMN BUFFERS (optional, requires .colbin v2 with page-aligned payloads)
// -------------------------------------------------------------------
// Instead of memcpy'ing column data through std::vector and then a second
// memcpy into an MTLBuffer, we mmap the .colbin file once and ask Metal to
// wrap each page-aligned payload directly as an MTL::Buffer (via
// newBufferWithBytesNoCopy). On Apple Silicon the GPU shares page tables
// with the CPU, so these buffers are immediately GPU-readable without any
// copy or coherence pass. This eliminates both the tbl-parse cost and the
// host->buffer staging cost on the warm path.
//
// Lifetime: MappedColumns owns the mmap. All buffers it vends remain valid
// as long as the MappedColumns object (or a shared_ptr copy of it) lives.
// The MTLBuffer deallocator is nullptr: Metal won't touch the memory; we
// unmap in the MappedColumns destructor once all buffers are released.
// ===================================================================
struct MappedColumns {
    // Column lookup by index (matches ColSpec.columnIndex).
    std::unordered_map<int, MTL::Buffer*> buffers;
    size_t nRows = 0;

    // mmap owner. Keep alive at least until every vended MTLBuffer has been
    // released. A std::shared_ptr captured in the deallocator block would be
    // cleanest; for now we keep the mapping alive until this struct dies.
    void*  mapBase = nullptr;
    size_t mapSize = 0;

    MappedColumns() = default;
    MappedColumns(const MappedColumns&) = delete;
    MappedColumns& operator=(const MappedColumns&) = delete;
    MappedColumns(MappedColumns&& o) noexcept
      : buffers(std::move(o.buffers)), nRows(o.nRows),
        mapBase(o.mapBase), mapSize(o.mapSize) {
        o.mapBase = nullptr; o.mapSize = 0;
    }
    MappedColumns& operator=(MappedColumns&& o) noexcept {
        if (this != &o) {
            reset();
            buffers  = std::move(o.buffers);
            nRows    = o.nRows;
            mapBase  = o.mapBase;
            mapSize  = o.mapSize;
            o.mapBase = nullptr; o.mapSize = 0;
        }
        return *this;
    }
    ~MappedColumns() { reset(); }

    MTL::Buffer* get(int col) const {
        auto it = buffers.find(col);
        return it == buffers.end() ? nullptr : it->second;
    }
    bool valid() const { return mapBase != nullptr && !buffers.empty(); }

    void reset() {
        for (auto& [k, b] : buffers) if (b) b->release();
        buffers.clear();
        if (mapBase && mapSize) ::munmap(mapBase, mapSize);
        mapBase = nullptr; mapSize = 0;
    }
};

namespace colbin {

// Load a set of columns as MTLBuffers wrapping the mmap'd .colbin payload,
// with zero memory copies. Returns an invalid MappedColumns on any failure
// (caller should fall back to loadColumnsMultiAuto).
inline MappedColumns loadColumnsAsBuffers(MTL::Device* device,
                                           const std::string& tblPath,
                                           const std::vector<ColSpec>& specs) {
    MappedColumns out;
    if (!device) return out;

    const std::string cp = binaryPath(tblPath);
    size_t tblSize = 0; int64_t tblMtime = 0;
    if (!statFile(tblPath, tblSize, tblMtime)) return out;

    int fd = ::open(cp.c_str(), O_RDONLY);
    if (fd < 0) return out;

    struct stat st{};
    if (::fstat(fd, &st) != 0) { ::close(fd); return out; }
    size_t fileSize = (size_t)st.st_size;
    if (fileSize < sizeof(FileHeader)) { ::close(fd); return out; }

    void* base = ::mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) return out;

    FileHeader hdr;
    memcpy(&hdr, base, sizeof(hdr));
    if (memcmp(hdr.magic, MAGIC, 8) != 0 ||
        hdr.version != VERSION ||
        hdr.source_size != tblSize ||
        hdr.source_mtime_ns != tblMtime) {
        ::munmap(base, fileSize);
        return out;
    }
    if (sizeof(FileHeader) + hdr.n_cols * sizeof(ColDesc) > fileSize) {
        ::munmap(base, fileSize);
        return out;
    }

    const ColDesc* descs = (const ColDesc*)((const char*)base + sizeof(FileHeader));
    std::unordered_map<int, const ColDesc*> byIdx;
    byIdx.reserve(hdr.n_cols);
    for (uint32_t i = 0; i < hdr.n_cols; i++) byIdx[descs[i].columnIndex] = &descs[i];

    for (const auto& s : specs) {
        auto it = byIdx.find(s.columnIndex);
        if (it == byIdx.end())                 { ::munmap(base, fileSize); return MappedColumns{}; }
        const ColDesc* d = it->second;
        if (decodeType(d->dtype) != s.type)    { ::munmap(base, fileSize); return MappedColumns{}; }
        if (s.type == ColType::CHAR_FIXED && d->fixedWidth != s.fixedWidth) {
            ::munmap(base, fileSize); return MappedColumns{};
        }
        if (d->offset % ALIGN != 0) {
            // Misaligned payload (likely v1 file): cannot zero-copy.
            ::munmap(base, fileSize); return MappedColumns{};
        }
        if (d->offset + d->size_bytes > fileSize) {
            ::munmap(base, fileSize); return MappedColumns{};
        }
        void* colPtr = (char*)base + d->offset;
        MTL::Buffer* buf = device->newBuffer(colPtr, d->size_bytes,
                                              MTL::ResourceStorageModeShared,
                                              nullptr /* no deallocator: we own the mmap */);
        if (!buf) {
            ::munmap(base, fileSize);
            return MappedColumns{};
        }
        out.buffers[s.columnIndex] = buf;
    }
    out.nRows   = hdr.n_rows;
    out.mapBase = base;
    out.mapSize = fileSize;
    return out;
}

} // namespace colbin (zero-copy API re-opened)

inline bool zeroCopyEnabled() {
    static const bool v = []() {
        const char* e = ::getenv("GPUDB_NO_ZEROCOPY");
        return !(e && e[0] == '1');
    }();
    return v;
}

// ===================================================================
// QueryColumns — unified column source for GPU queries.
// -------------------------------------------------------------------
// Backed by either zero-copy mmap (default, .colbin v2) or .tbl/.colbin
// copy-path fallback. Exposes a single API regardless of mode:
//
//     auto cols = loadQueryColumns(device, path, specs);
//     size_t n = cols.rows();
//     MTL::Buffer* buf = cols.buffer(colIdx);     // GPU-ready, no extra copy
//     const int*   p  = cols.ints(colIdx);       // CPU-readable pointer
//     const float* q  = cols.floats(colIdx);
//     const char*  c  = cols.chars(colIdx);
//
// Ownership: cols releases all buffers and munmaps on destruction, so the
// typical pattern is one QueryColumns per table, declared at function scope.
// GPUDB_NO_ZEROCOPY=1 forces the legacy copy path for A/B testing.
// ===================================================================
class QueryColumns {
public:
    QueryColumns() = default;
    QueryColumns(const QueryColumns&) = delete;
    QueryColumns& operator=(const QueryColumns&) = delete;
    QueryColumns(QueryColumns&& o) noexcept { moveFrom(std::move(o)); }
    QueryColumns& operator=(QueryColumns&& o) noexcept {
        if (this != &o) { releaseOwned(); moveFrom(std::move(o)); }
        return *this;
    }
    ~QueryColumns() { releaseOwned(); }

    size_t       rows()                   const { return rows_; }
    MTL::Buffer* buffer(int col)          const { auto it = buffers_.find(col); return it==buffers_.end()?nullptr:it->second; }
    const int*   ints  (int col)          const { auto it = intPtrs_.find(col);   return it==intPtrs_.end()?nullptr:it->second; }
    const float* floats(int col)          const { auto it = floatPtrs_.find(col); return it==floatPtrs_.end()?nullptr:it->second; }
    const char*  chars (int col)          const { auto it = charPtrs_.find(col);  return it==charPtrs_.end()?nullptr:it->second; }
    std::span<const int>   intSpan  (int col) const { return { ints(col),   ints(col)   ? rows_ : 0 }; }
    std::span<const float> floatSpan(int col) const { return { floats(col), floats(col) ? rows_ : 0 }; }
    // charSpan width depends on schema; caller supplies stride:
    std::span<const char>  charSpan (int col, size_t stride = 1) const { return { chars(col), chars(col) ? rows_ * stride : 0 }; }
    bool         zeroCopy()               const { return mapped_.valid(); }

    // Internal builders (used by loadQueryColumns below).
    void _adoptMapped(MappedColumns&& mc, const std::vector<ColSpec>& specs);
    void _adoptCopied(MTL::Device* dev, LoadedColumns&& lc, const std::vector<ColSpec>& specs);

private:
    MappedColumns                              mapped_;      // active when zero-copy
    LoadedColumns                              copied_;      // active when fallback
    std::unordered_map<int, MTL::Buffer*>      buffers_;
    std::unordered_map<int, const int*>        intPtrs_;
    std::unordered_map<int, const float*>      floatPtrs_;
    std::unordered_map<int, const char*>       charPtrs_;
    std::vector<MTL::Buffer*>                  ownedBuffers_; // released on dtor (copy path)
    size_t                                     rows_ = 0;

    void releaseOwned() {
        for (auto* b : ownedBuffers_) if (b) b->release();
        ownedBuffers_.clear();
        buffers_.clear(); intPtrs_.clear(); floatPtrs_.clear(); charPtrs_.clear();
        rows_ = 0;
    }
    void moveFrom(QueryColumns&& o) {
        mapped_       = std::move(o.mapped_);
        copied_       = std::move(o.copied_);
        buffers_      = std::move(o.buffers_);
        intPtrs_      = std::move(o.intPtrs_);
        floatPtrs_    = std::move(o.floatPtrs_);
        charPtrs_     = std::move(o.charPtrs_);
        ownedBuffers_ = std::move(o.ownedBuffers_);
        rows_         = o.rows_;
        o.rows_ = 0;
    }
};

inline void QueryColumns::_adoptMapped(MappedColumns&& mc, const std::vector<ColSpec>& specs) {
    rows_   = mc.nRows;
    // Copy buffer pointers into our map (MappedColumns keeps ownership of the
    // MTLBuffer until mapped_ is destroyed; we must NOT release them ourselves).
    for (const auto& s : specs) {
        MTL::Buffer* buf = mc.get(s.columnIndex);
        if (!buf) continue;
        buffers_[s.columnIndex] = buf;
        const char* base = (const char*)buf->contents();
        switch (s.type) {
            case ColType::INT:
            case ColType::DATE:       intPtrs_  [s.columnIndex] = (const int*)base;   break;
            case ColType::FLOAT:      floatPtrs_[s.columnIndex] = (const float*)base; break;
            case ColType::CHAR1:
            case ColType::CHAR_FIXED: charPtrs_ [s.columnIndex] = (const char*)base;  break;
        }
    }
    mapped_ = std::move(mc);
}

inline void QueryColumns::_adoptCopied(MTL::Device* dev, LoadedColumns&& lc, const std::vector<ColSpec>& specs) {
    copied_ = std::move(lc);
    // Determine row count from first spec.
    if (!specs.empty()) {
        const auto& s0 = specs.front();
        switch (s0.type) {
            case ColType::INT:
            case ColType::DATE:       rows_ = copied_.ints(s0.columnIndex).size(); break;
            case ColType::FLOAT:      rows_ = copied_.floats(s0.columnIndex).size(); break;
            case ColType::CHAR1:      rows_ = copied_.chars(s0.columnIndex).size(); break;
            case ColType::CHAR_FIXED: rows_ = s0.fixedWidth > 0
                                              ? copied_.chars(s0.columnIndex).size() / (size_t)s0.fixedWidth
                                              : copied_.chars(s0.columnIndex).size();
                                      break;
        }
    }
    for (const auto& s : specs) {
        size_t bytes = 0;
        const void* src = nullptr;
        switch (s.type) {
            case ColType::INT:
            case ColType::DATE: {
                const auto& v = copied_.ints(s.columnIndex);
                bytes = v.size() * sizeof(int); src = v.data();
                intPtrs_[s.columnIndex] = v.data();
                break;
            }
            case ColType::FLOAT: {
                const auto& v = copied_.floats(s.columnIndex);
                bytes = v.size() * sizeof(float); src = v.data();
                floatPtrs_[s.columnIndex] = v.data();
                break;
            }
            case ColType::CHAR1:
            case ColType::CHAR_FIXED: {
                const auto& v = copied_.chars(s.columnIndex);
                bytes = v.size(); src = v.data();
                charPtrs_[s.columnIndex] = v.data();
                break;
            }
        }
        MTL::Buffer* buf = dev->newBuffer(src, bytes, MTL::ResourceStorageModeShared);
        buffers_[s.columnIndex] = buf;
        ownedBuffers_.push_back(buf);
    }
}

// Primary query-column loader. Attempts zero-copy mmap when enabled (default);
// falls back to the copy path transparently. Records into LoadStats so the
// timing printout reflects the true source.
inline QueryColumns loadQueryColumns(MTL::Device* device,
                                      const std::string& tblPath,
                                      const std::vector<ColSpec>& specs) {
    QueryColumns qc;
    if (zeroCopyEnabled()) {
        MappedColumns mc = colbin::loadColumnsAsBuffers(device, tblPath, specs);
        if (mc.valid()) {
            // Size charge for LoadStats: sum of referenced column sizes.
            size_t bytes = 0;
            for (auto& [k, b] : mc.buffers) if (b) bytes += b->length();
            loadStats().recordBinary(bytes);
            qc._adoptMapped(std::move(mc), specs);
            return qc;
        }
    }
    LoadedColumns lc = loadColumnsMultiAuto(tblPath, specs);
    qc._adoptCopied(device, std::move(lc), specs);
    return qc;
}

// Auto loader: binary-first with transparent .tbl fallback.
// Set GPUDB_NO_BINARY=1 to force the .tbl parser (diagnostics).
inline LoadedColumns loadColumnsMultiAuto(const std::string& filePath,
                                           const std::vector<ColSpec>& specs) {
    static const bool binaryEnabled = []() {
        const char* e = ::getenv("GPUDB_NO_BINARY");
        return !(e && e[0] == '1');
    }();
    if (binaryEnabled) {
        LoadedColumns fromBinary;
        if (colbin::loadColumnsFromBinary(filePath, specs, fromBinary)) {
            size_t bsz = 0; int64_t bmt = 0;
            (void)colbin::statFile(colbin::binaryPath(filePath), bsz, bmt);
            loadStats().recordBinary(bsz);
            return fromBinary;
        }
        static std::set<std::string> warned;
        if (warned.insert(filePath).second) {
            std::cerr << "[infra] .colbin missing/stale for " << filePath
                      << " -- using .tbl parser (run `make colbin-sfN` to accelerate).\n";
        }
    }
    struct stat st{};
    size_t sz = (::stat(filePath.c_str(), &st) == 0) ? (size_t)st.st_size : 0;
    loadStats().recordText(sz);
    auto t0 = std::chrono::high_resolution_clock::now();
    LoadedColumns out = loadColumnsMultiText(filePath, specs);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ingestMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    loadStats().recordExcluded(ingestMs);
    std::cerr << "[tbl-ingest] " << filePath << ": "
              << std::fixed << std::setprecision(1) << ingestMs
              << " ms (one-time, excluded from e2e)\n";
    return out;
}

// Back-compat: queries still call loadColumnsMulti(...); route through
// the auto (binary-first) loader so they transparently gain the speedup.
inline LoadedColumns loadColumnsMulti(const std::string& filePath,
                                       const std::vector<ColSpec>& specs) {
    return loadColumnsMultiAuto(filePath, specs);
}

// --- Standard Column Loaders (file-based, for SF1/SF10) ---
template<typename T, typename ParseFn>
inline std::vector<T> loadColumn(const std::string& filePath, int columnIndex, ParseFn parse) {
    std::vector<T> data;
    std::ifstream file(filePath);
    if (!file.is_open()) { std::cerr << "Error: Could not open file " << filePath << std::endl; return data; }
    std::string line;
    while (std::getline(file, line)) {
        int currentCol = 0; size_t start = 0; size_t end = line.find('|');
        while (end != std::string::npos) {
            if (currentCol == columnIndex) { parse(data, line.substr(start, end - start)); break; }
            start = end + 1; end = line.find('|', start); currentCol++;
        }
    }
    return data;
}

inline std::vector<int> loadIntColumn(const std::string& filePath, int columnIndex) {
    return loadColumn<int>(filePath, columnIndex, [](auto& v, const std::string& t) { v.push_back(std::stoi(t)); });
}
inline std::vector<float> loadFloatColumn(const std::string& filePath, int columnIndex) {
    return loadColumn<float>(filePath, columnIndex, [](auto& v, const std::string& t) { v.push_back(std::stof(t)); });
}
inline std::vector<int> loadDateColumn(const std::string& filePath, int columnIndex) {
    return loadColumn<int>(filePath, columnIndex, [](auto& v, std::string t) {
        t.erase(std::remove(t.begin(), t.end(), '-'), t.end());
        v.push_back(std::stoi(t));
    });
}
inline std::vector<char> loadCharColumn(const std::string& filePath, int columnIndex, int fixed_width = 0) {
    std::vector<char> data; std::ifstream file(filePath);
    if (!file.is_open()) { std::cerr << "Error: Could not open file " << filePath << std::endl; return data; }
    std::string line;
    while (std::getline(file, line)) {
        int currentCol = 0; size_t start = 0; size_t end = line.find('|');
        while (end != std::string::npos) {
            if (currentCol == columnIndex) {
                std::string token = line.substr(start, end - start);
                if (fixed_width > 0) { for (int i = 0; i < fixed_width; ++i) data.push_back(i < (int)token.length() ? token[i] : '\0'); }
                else { data.push_back(token[0]); }
                break;
            }
            start = end + 1; end = line.find('|', start); currentCol++;
        }
    }
    return data;
}

// ===================================================================
// SHARED TABLE LOADERS
// ===================================================================

// --- Nation Table ---
struct NationData {
    std::vector<int>  nationkey;
    std::vector<char> name;      // fixed-width 25 chars per row
    std::vector<int>  regionkey;
    static constexpr int NAME_WIDTH = 25;
};

inline NationData loadNation(const std::string& sf_path, bool with_regionkey = false) {
    NationData d;
    std::vector<ColSpec> specs = {{0, ColType::INT}, {1, ColType::CHAR_FIXED, NationData::NAME_WIDTH}};
    if (with_regionkey) specs.push_back({2, ColType::INT});
    auto cols = loadColumnsMulti(sf_path + "nation.tbl", specs);
    d.nationkey = std::move(cols.ints(0));
    d.name      = std::move(cols.chars(1));
    if (with_regionkey) d.regionkey = std::move(cols.ints(2));
    return d;
}

// --- Region Table ---
struct RegionData {
    std::vector<int>  regionkey;
    std::vector<char> name;
    static constexpr int NAME_WIDTH = 25;
};

inline RegionData loadRegion(const std::string& sf_path) {
    RegionData d;
    auto cols = loadColumnsMulti(sf_path + "region.tbl", {{0, ColType::INT}, {1, ColType::CHAR_FIXED, RegionData::NAME_WIDTH}});
    d.regionkey = std::move(cols.ints(0));
    d.name      = std::move(cols.chars(1));
    return d;
}

// --- Supplier Table (basic: suppkey + nationkey) ---
struct SupplierBasic {
    std::vector<int> suppkey;
    std::vector<int> nationkey;
};

inline SupplierBasic loadSupplierBasic(const std::string& sf_path) {
    SupplierBasic d;
    auto cols = loadColumnsMulti(sf_path + "supplier.tbl", {{0, ColType::INT}, {3, ColType::INT}});
    d.suppkey   = std::move(cols.ints(0));
    d.nationkey = std::move(cols.ints(3));
    return d;
}

// --- Single nation key lookup by name ---
inline int findNationKey(const NationData& nat, const std::string& target) {
    for (size_t i = 0; i < nat.nationkey.size(); i++)
        if (trimFixed(nat.name.data(), i, NationData::NAME_WIDTH) == target) return nat.nationkey[i];
    return -1;
}

// --- CPU Bitmap Builder ---
// Builds a bitmap for keys satisfying a predicate. Returns (bitmap_vector, bitmap_ints, max_key).
struct CPUBitmap {
    std::vector<uint> data;
    uint ints;
    int max_key;
};

template<typename Pred>
inline CPUBitmap buildCPUBitmap(const std::vector<int>& keys, Pred pred) {
    CPUBitmap b;
    b.max_key = 0;
    for (int k : keys) b.max_key = std::max(b.max_key, k);
    b.ints = (b.max_key + 31) / 32 + 1;
    b.data.assign(b.ints, 0);
    for (size_t i = 0; i < keys.size(); i++) {
        if (pred(i)) {
            int k = keys[i];
            b.data[k / 32] |= (1u << (k % 32));
        }
    }
    return b;
}

// Overload: bitmap where every key qualifies (no predicate).
inline CPUBitmap buildCPUBitmap(const std::vector<int>& keys) {
    return buildCPUBitmap(keys, [](size_t) { return true; });
}

// Overload: accept raw pointer + count (for QueryColumns::ints + rows()).
template<typename Pred>
inline CPUBitmap buildCPUBitmap(const int* keys, size_t count, Pred pred) {
    CPUBitmap b;
    b.max_key = 0;
    for (size_t i = 0; i < count; i++) b.max_key = std::max(b.max_key, keys[i]);
    b.ints = (b.max_key + 31) / 32 + 1;
    b.data.assign(b.ints, 0);
    for (size_t i = 0; i < count; i++) {
        if (pred(i)) {
            int k = keys[i];
            b.data[k / 32] |= (1u << (k % 32));
        }
    }
    return b;
}
inline CPUBitmap buildCPUBitmap(const int* keys, size_t count) {
    return buildCPUBitmap(keys, count, [](size_t) { return true; });
}

// Upload a CPU bitmap to a Metal buffer (caller must release).
inline MTL::Buffer* uploadBitmap(MTL::Device* device, const CPUBitmap& bm) {
    auto buf = device->newBuffer(bm.ints * sizeof(uint), MTL::ResourceStorageModeShared);
    memcpy(buf->contents(), bm.data.data(), bm.ints * sizeof(uint));
    return buf;
}

// ===================================================================
// TIMING & BUFFER HELPERS
// ===================================================================

// --- Timing Summary ---
// Prints both the original 4-line summary (preserves compatibility with
// existing log parsers) AND a detailed breakdown that includes the auto-
// tracked stages from g_detailed_timing (PSO creation, per-kernel GPU time,
// opt-in buffer allocation). Also emits two machine-readable lines for
// benchmark scripts to harvest. Format mirrors GPUDBMetalCodeGen's
// printDetailedTimingSummary so the same analysis scripts work on both
// projects:
//
//   TIMING_CSV,<sf>,<query>,<analyze>,<plan>,<codegen>,<compile>,<pso>,
//               <data_load>,<buffer_alloc>,<gpu>,<post>,<query_compute>,
//               <end_to_end>,<load_source>,<load_bytes>,<load_mibps>,
//               <ingest_ms>,<query_compute_ms_dup>
//
// For LC, analyze/plan/codegen/compile are always 0.
inline void printDetailedTimingSummary(const DetailedTiming& dt) {
    const double compileOverhead = dt.analyzeMs + dt.planMs + dt.codegenMs +
                                    dt.compileMs + dt.psoMs;
    const double queryCompute    = dt.bufferAllocMs + dt.gpuTotalMs + dt.postMs;
    const double endToEnd        = compileOverhead + dt.dataLoadMs + queryCompute;

    auto bar = []() { printf("  +------------------------------+------------------+\n"); };
    auto row = [](const char* label, double ms) {
        printf("  | %-28s | %13.3f ms |\n", label, ms);
    };
    auto rowHL = [](const char* label, double ms) {
        printf("  | %-28s | %13.3f ms | <<\n", label, ms);
    };

    printf("\n  Detailed Timing");
    if (!dt.queryName.empty())   printf(" — %s", dt.queryName.c_str());
    if (!dt.scaleFactor.empty()) printf(" @ %s", dt.scaleFactor.c_str());
    printf("\n");

    // --- Compile Overhead ---
    if (compileOverhead > 0.0) {
        bar();
        printf("  | %-28s | %-16s |\n", "Compile Overhead", "");
        bar();
        if (dt.analyzeMs > 0.0) row("  Query Analyze",   dt.analyzeMs);
        if (dt.planMs    > 0.0) row("  Planning",        dt.planMs);
        if (dt.codegenMs > 0.0) row("  Code Generation", dt.codegenMs);
        if (dt.compileMs > 0.0) row("  Metal Compile",   dt.compileMs);
        if (dt.psoMs     > 0.0) row("  PSO Creation",    dt.psoMs);
    }

    // --- Data Load (I/O) ---
    bar();
    char dlLabel[64];
    snprintf(dlLabel, sizeof(dlLabel), "Data Load (I/O, %s)", dt.loadSource.c_str());
    printf("  | %-28s | %-16s |\n", dlLabel, "");
    bar();
    row("  Load", dt.dataLoadMs);
    if (dt.dataLoadMs > 0.0 && dt.loadBytes > 0) {
        double mib = (double)dt.loadBytes / (1024.0 * 1024.0);
        double mibps = mib / (dt.dataLoadMs / 1000.0);
        char tp[64];
        snprintf(tp, sizeof(tp), "  Throughput (%.1f MiB)", mib);
        printf("  | %-28s | %10.1f MiB/s |\n", tp, mibps);
    }
    if (dt.ingestMs > 0.0) row("  One-time .tbl ingest*", dt.ingestMs);

    // --- Query Execution ---
    bar();
    printf("  | %-28s | %-16s |\n", "Query Execution", "");
    bar();
    if (dt.bufferAllocMs > 0.0) row("  Buffer Setup", dt.bufferAllocMs);
    if (!dt.phaseKernelMs.empty()) {
        for (const auto& [name, ms] : dt.phaseKernelMs) {
            char label[64];
            snprintf(label, sizeof(label), "  kernel %s", name.c_str());
            row(label, ms);
        }
    }
    row("  GPU Compute",          dt.gpuTotalMs);
    row("  CPU Post",             dt.postMs);
    rowHL("  Query Compute",      queryCompute);

    // --- Totals ---
    bar();
    printf("  | %-28s | %-16s |\n", "Totals", "");
    bar();
    if (compileOverhead > 0.0) row("  Compile Overhead", compileOverhead);
    row("  Data Load",                   dt.dataLoadMs);
    row("  Query Compute",               queryCompute);
    rowHL("  End-to-End",                endToEnd);
    bar();
    if (dt.ingestMs > 0.0) {
        printf("  * one-time .tbl ingest cost, excluded from End-to-End.\n");
        printf("    Run `make colbin-sfN` to cache columnar binaries (.colbin) and\n");
        printf("    eliminate this cost on subsequent runs.\n");
    }

    // --- Machine-readable CSV (19 fields, matches CodeGen format) ---
    double mibps = 0.0;
    if (dt.dataLoadMs > 0.0 && dt.loadBytes > 0) {
        double mib = (double)dt.loadBytes / (1024.0 * 1024.0);
        mibps = mib / (dt.dataLoadMs / 1000.0);
    }
    printf("TIMING_CSV,%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%zu,%.1f,%.3f,%.3f\n",
           dt.scaleFactor.c_str(), dt.queryName.c_str(),
           dt.analyzeMs, dt.planMs, dt.codegenMs, dt.compileMs, dt.psoMs,
           dt.dataLoadMs, dt.bufferAllocMs, dt.gpuTotalMs, dt.postMs,
           queryCompute, endToEnd,
           dt.loadSource.c_str(), dt.loadBytes, mibps,
           dt.ingestMs, queryCompute);
}

// Legacy entrypoint: all LC query files call this 3-arg form. Populates the
// global DetailedTiming from (parseMs, gpuMs, postMs) + current LoadStats,
// then delegates to the detailed printer above.
inline void printTimingSummary(double parseMs, double gpuMs, double postMs) {
    g_detailed_timing.queryName   = g_current_query;
    g_detailed_timing.scaleFactor = g_current_sf;
    g_detailed_timing.dataLoadMs  = parseMs;
    g_detailed_timing.gpuTotalMs  = gpuMs;
    g_detailed_timing.postMs      = postMs;
    g_detailed_timing.loadSource  = loadStats().source();
    g_detailed_timing.loadBytes   = loadStats().bytes;
    g_detailed_timing.ingestMs    = loadStats().excludedMs;

    // Headline summary preserved for existing log-grep-based tooling.
    const DetailedTiming& dt = g_detailed_timing;
    const double compileOverhead = dt.analyzeMs + dt.planMs + dt.codegenMs +
                                    dt.compileMs + dt.psoMs;
    const double queryCompute    = dt.bufferAllocMs + dt.gpuTotalMs + dt.postMs;
    const double endToEnd        = compileOverhead + dt.dataLoadMs + queryCompute;
    printf("  Data Load (%s):    %10.2f ms\n", dt.loadSource.c_str(), parseMs);
    printf("  GPU Execution:      %10.2f ms\n", gpuMs);
    printf("  CPU Post Process:   %10.2f ms\n", postMs);
    printf("  Query Compute:      %10.2f ms  (buffers+GPU+post)  <<\n", queryCompute);
    printf("  End-to-End:         %10.2f ms  <<\n", endToEnd);

    printDetailedTimingSummary(g_detailed_timing);
}

// --- Bitmap Buffer Creation ---
inline MTL::Buffer* createBitmapBuffer(MTL::Device* device, int maxKey) {
    const uint ints = (maxKey + 31) / 32 + 1;
    auto buf = device->newBuffer(ints * sizeof(uint), MTL::ResourceStorageModeShared);
    memset(buf->contents(), 0, ints * sizeof(uint));
    return buf;
}

// --- Filled Buffer Creation (allocate + memset in one call) ---
inline MTL::Buffer* createFilledBuffer(MTL::Device* device, size_t bytes, int fillByte = 0) {
    auto buf = device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    memset(buf->contents(), fillByte, bytes);
    return buf;
}

// ===================================================================
// POST-PROCESSING STRUCTS AND FUNCTIONS
// ===================================================================

// --- Q3 Sort and Print ---
struct Q3Aggregates_CPU {
    int key;
    float revenue;
    unsigned int orderdate;
    unsigned int shippriority;
};
inline double sortAndPrintQ3(Q3Aggregates_CPU* dense, uint resultCount) {
    auto t0 = std::chrono::high_resolution_clock::now();
    size_t topK = std::min((size_t)10, (size_t)resultCount);
    std::partial_sort(dense, dense + topK, dense + resultCount,
        [](const Q3Aggregates_CPU& a, const Q3Aggregates_CPU& b) {
            if (a.revenue != b.revenue) return a.revenue > b.revenue;
            return a.orderdate < b.orderdate;
        });
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("\nTPC-H Q3 Results (Top 10):\n");
    printf("+----------+------------+------------+--------------+\n");
    printf("| orderkey |   revenue  | orderdate  | shippriority |\n");
    printf("+----------+------------+------------+--------------+\n");
    for (size_t i = 0; i < topK; i++) {
        printf("| %8d | $%10.2f | %10u | %12u |\n",
               dense[i].key, dense[i].revenue, dense[i].orderdate, dense[i].shippriority);
    }
    printf("+----------+------------+------------+--------------+\n");
    printf("Total results found: %u\n", resultCount);
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// --- Q3 Chunk Slot (used by Q3 SF100 and Q5 SF100) ---
struct Q3ChunkSlot {
    MTL::Buffer* orderkey; MTL::Buffer* shipdate; MTL::Buffer* extprice; MTL::Buffer* discount;
};

// --- Q5 Post-Processing ---
struct Q5Result { std::string name; float revenue; };
inline void postProcessQ5(const float* nation_revenue, std::map<int, std::string>& nation_names) {
    std::vector<Q5Result> final_results;
    for (int i = 0; i < 25; i++) {
        if (nation_revenue[i] > 0.0f)
            final_results.push_back({nation_names[i], nation_revenue[i]});
    }
    std::sort(final_results.begin(), final_results.end(),
              [](const Q5Result& a, const Q5Result& b) { return a.revenue > b.revenue; });
    printf("\nTPC-H Q5 Results:\n");
    printf("+------------------+-----------------+\n");
    printf("| n_name           |         revenue |\n");
    printf("+------------------+-----------------+\n");
    for (const auto& r : final_results)
        printf("| %-16s | $%14.2f |\n", r.name.c_str(), r.revenue);
    printf("+------------------+-----------------+\n");
}

// --- Q2 Post-Processing ---
struct Q2MatchResult_CPU {
    int partkey;
    int suppkey;
    unsigned int supplycost_cents;
};
struct Q2Result {
    float s_acctbal;
    std::string s_name, n_name, p_mfgr, s_address, s_phone, s_comment;
    int p_partkey;
};
struct SuppBitmapResult {
    std::vector<uint> bitmap;
    uint bitmap_ints;
    std::vector<size_t> index;  // direct-index by suppkey, SIZE_MAX = absent
};
inline SuppBitmapResult buildSuppBitmapAndIndex(const int* suppkey, const int* nationkey,
                                                 size_t count, const std::vector<int>& target_keys) {
    int max_key = 0;
    for (size_t i = 0; i < count; i++) max_key = std::max(max_key, suppkey[i]);
    SuppBitmapResult r;
    r.bitmap_ints = (max_key + 31) / 32 + 1;
    r.bitmap.resize(r.bitmap_ints, 0);
    r.index.assign(max_key + 1, SIZE_MAX);
    for (size_t i = 0; i < count; i++) {
        r.index[suppkey[i]] = i;
        bool match = false;
        for (int nk : target_keys) { if (nationkey[i] == nk) { match = true; break; } }
        if (match) r.bitmap[suppkey[i] / 32] |= (1u << (suppkey[i] % 32));
    }
    return r;
}
inline void postProcessQ2(Q2MatchResult_CPU* gpu_results, uint result_count,
                           const std::vector<size_t>& supp_index,
                           const float* s_acctbal, const int* s_nationkey,
                           const char* s_name, const char* s_address,
                           const char* s_phone, const char* s_comment,
                           std::map<int, std::string>& nation_names,
                           const int* p_partkey, size_t part_size,
                           const char* p_mfgr) {
    // Direct-index by partkey
    int max_pk = 0;
    for (size_t i = 0; i < part_size; i++) max_pk = std::max(max_pk, p_partkey[i]);
    std::vector<size_t> part_index(max_pk + 1, SIZE_MAX);
    for (size_t i = 0; i < part_size; i++) part_index[p_partkey[i]] = i;

    // Build nation alphabetical sort order (25 nations, cheap)
    int max_nk = 0;
    for (auto& [k, _] : nation_names) max_nk = std::max(max_nk, k);
    std::vector<int> nation_order(max_nk + 1, 0);
    {
        std::vector<std::pair<std::string, int>> ns;
        for (auto& [k, v] : nation_names) ns.push_back({v, k});
        std::sort(ns.begin(), ns.end());
        for (int i = 0; i < (int)ns.size(); i++) nation_order[ns[i].second] = i;
    }

    // Phase 1: Lightweight numeric pre-keys (no string allocation)
    struct PreKey { uint idx; float acctbal; int nation_ord; int suppkey; int partkey; };
    std::vector<PreKey> pre_keys;
    pre_keys.reserve(result_count);
    for (uint i = 0; i < result_count; i++) {
        int sk = gpu_results[i].suppkey;
        if (sk < 0 || (size_t)sk >= supp_index.size() || supp_index[sk] == SIZE_MAX) continue;
        size_t si = supp_index[sk];
        int nk = s_nationkey[si];
        pre_keys.push_back({i, s_acctbal[si],
                            (nk >= 0 && nk <= max_nk) ? nation_order[nk] : nk,
                            sk, gpu_results[i].partkey});
    }

    // Phase 2: partial_sort top LIMIT by exact sort order (numeric proxies)
    // s_name is "Supplier#XXXXXXXXX" (zero-padded), so suppkey order = s_name order
    constexpr size_t LIMIT = 100;
    const size_t K = std::min(pre_keys.size(), LIMIT);
    std::partial_sort(pre_keys.begin(), pre_keys.begin() + K, pre_keys.end(),
        [](const PreKey& a, const PreKey& b) {
            if (a.acctbal != b.acctbal) return a.acctbal > b.acctbal;
            if (a.nation_ord != b.nation_ord) return a.nation_ord < b.nation_ord;
            if (a.suppkey != b.suppkey) return a.suppkey < b.suppkey;
            return a.partkey < b.partkey;
        });

    // Phase 3: Materialize strings only for top K (100 vs ~100K)
    std::vector<Q2Result> final_results;
    final_results.reserve(K);
    for (size_t i = 0; i < K; i++) {
        auto& pk = pre_keys[i];
        size_t si = supp_index[pk.suppkey];
        Q2Result r;
        r.s_acctbal = pk.acctbal;
        r.s_name = trimFixed(s_name, si, 25);
        r.n_name = nation_names[s_nationkey[si]];
        r.p_partkey = pk.partkey;
        if (pk.partkey >= 0 && pk.partkey <= max_pk && part_index[pk.partkey] != SIZE_MAX)
            r.p_mfgr = trimFixed(p_mfgr, part_index[pk.partkey], 25);
        r.s_address = trimFixed(s_address, si, 40);
        r.s_phone = trimFixed(s_phone, si, 15);
        r.s_comment = trimFixed(s_comment, si, 101);
        final_results.push_back(std::move(r));
    }

    // Phase 4: Final exact sort (100 elements, virtually free)
    std::sort(final_results.begin(), final_results.end(), [](const Q2Result& a, const Q2Result& b) {
        if (a.s_acctbal != b.s_acctbal) return a.s_acctbal > b.s_acctbal;
        if (a.n_name != b.n_name) return a.n_name < b.n_name;
        if (a.s_name != b.s_name) return a.s_name < b.s_name;
        return a.p_partkey < b.p_partkey;
    });
    printf("\nTPC-H Q2 Results (Top 10 of LIMIT 100):\n");
    printf("+----------+------------------+----------+--------+------------------+\n");
    printf("| s_acctbal|          s_name  | n_name   | p_key  | p_mfgr           |\n");
    printf("+----------+------------------+----------+--------+------------------+\n");
    size_t show = std::min((size_t)10, final_results.size());
    for (size_t i = 0; i < show; i++) {
        printf("| %8.2f | %-16s | %-8s | %6d | %-16s |\n",
               final_results[i].s_acctbal, final_results[i].s_name.c_str(),
               final_results[i].n_name.c_str(), final_results[i].p_partkey,
               final_results[i].p_mfgr.c_str());
    }
    printf("+----------+------------------+----------+--------+------------------+\n");
}

// --- Nation/Region SF100 Loader ---
inline void parseNationRegionSF100(const MappedFile& natFile, const std::vector<size_t>& natIdx,
                                    std::vector<int>& nationkey, std::vector<int>& regionkey,
                                    std::vector<char>& name_chars,
                                    const MappedFile* regFile = nullptr, const std::vector<size_t>* regIdx = nullptr,
                                    std::vector<int>* r_regionkey_out = nullptr, std::vector<char>* r_name_chars_out = nullptr) {
    nationkey.resize(natIdx.size());
    name_chars.resize(natIdx.size() * 25);
    parseIntColumnChunk(natFile, natIdx, 0, natIdx.size(), 0, nationkey.data());
    parseCharColumnChunkFixed(natFile, natIdx, 0, natIdx.size(), 1, 25, name_chars.data());
    if (regFile && regIdx && r_regionkey_out && r_name_chars_out) {
        regionkey.resize(natIdx.size());
        parseIntColumnChunk(natFile, natIdx, 0, natIdx.size(), 2, regionkey.data());
        r_regionkey_out->resize(regIdx->size());
        r_name_chars_out->resize(regIdx->size() * 25);
        parseIntColumnChunk(*regFile, *regIdx, 0, regIdx->size(), 0, r_regionkey_out->data());
        parseCharColumnChunkFixed(*regFile, *regIdx, 0, regIdx->size(), 1, 25, r_name_chars_out->data());
    }
}

// --- Q9 Post-Processing ---
struct Q9Result {
    int nationkey;
    int year;
    float profit;
};
struct Q9Aggregates_CPU {
    uint key;
    float profit;
};
inline void postProcessQ9(const void* finalHTContents, uint htSize,
                           std::map<int, std::string>& nation_names) {
    auto* results = (const Q9Aggregates_CPU*)finalHTContents;
    std::vector<Q9Result> final_results;
    for (uint i = 0; i < htSize; ++i) {
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
    std::map<int, double> year_totals;
    for (const auto& r : final_results) year_totals[r.year] += (double)r.profit;
    printf("\nComparable TPC-H Q9 (yearly sum_profit):\n");
    printf("+--------+---------------+\n");
    printf("| o_year |   sum_profit  |\n");
    printf("+--------+---------------+\n");
    for (const auto& kv : year_totals) printf("| %6d | %13.4f |\n", kv.first, kv.second);
    printf("+--------+---------------+\n");
}

// --- Q13 Post-Processing ---
struct Q13Result {
    uint c_count;
    uint custdist;
};
inline void postProcessQ13(const void* histContents, uint histMaxBins) {
    auto* hist = (const uint*)histContents;
    std::vector<Q13Result> final_results;
    for (uint i = 0; i < histMaxBins; i++) {
        if (hist[i] > 0) final_results.push_back({i, hist[i]});
    }
    std::sort(final_results.begin(), final_results.end(), [](const Q13Result& a, const Q13Result& b) {
        if (a.custdist != b.custdist) return a.custdist > b.custdist;
        return a.c_count > b.c_count;
    });
    printf("\nTPC-H Query 13 Results (Comparable histogram):\n");
    printf("+---------+----------+\n");
    printf("| c_count | custdist |\n");
    printf("+---------+----------+\n");
    for (const auto& res : final_results) printf("| %7u | %8u |\n", res.c_count, res.custdist);
    printf("+---------+----------+\n");
}

// ===================================================================
// CHUNKED STREAM LOOP — Generic double-buffered streaming framework
// ===================================================================
// Timing result from chunked streaming
struct ChunkedStreamTiming {
    double parseMs = 0.0;
    double gpuMs   = 0.0;
    double postMs  = 0.0;
    size_t chunkCount = 0;
};

// ParseFn:    (SlotT& slot, size_t startRow, size_t rowCount)
// DispatchFn: (SlotT& slot, uint chunkSize, MTL::CommandBuffer* cmdBuf)
//             — must create encoder, encode, endEncoding, and commit
// AccumFn:    (uint chunkSize, size_t chunkNum) — per-chunk post-GPU work
template<typename SlotT, typename ParseFn, typename DispatchFn, typename AccumFn>
ChunkedStreamTiming chunkedStreamLoop(
    MTL::CommandQueue* commandQueue,
    SlotT* slots, int numSlots,
    size_t totalRows, size_t chunkRows,
    ParseFn parseChunk,
    DispatchFn dispatchGPU,
    AccumFn onChunkDone)
{
    ChunkedStreamTiming t;

    // Pre-parse first chunk into slot 0
    size_t firstChunk = std::min(chunkRows, totalRows);
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        parseChunk(slots[0], 0, firstChunk);
        auto t1 = std::chrono::high_resolution_clock::now();
        t.parseMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    size_t offset = 0;
    while (offset < totalRows) {
        size_t rowsThisChunk = std::min(chunkRows, totalRows - offset);
        SlotT& slot = slots[t.chunkCount % numSlots];

        // Dispatch GPU (caller encodes + commits)
        MTL::CommandBuffer* cmdBuf = commandQueue->commandBuffer();
        dispatchGPU(slot, (uint)rowsThisChunk, cmdBuf);

        // Double-buffer: parse next chunk while GPU runs
        size_t nextOffset = offset + rowsThisChunk;
        if (nextOffset < totalRows) {
            size_t nextRows = std::min(chunkRows, totalRows - nextOffset);
            SlotT& nextSlot = slots[(t.chunkCount + 1) % numSlots];
            auto t0 = std::chrono::high_resolution_clock::now();
            parseChunk(nextSlot, nextOffset, nextRows);
            auto t1 = std::chrono::high_resolution_clock::now();
            t.parseMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        cmdBuf->waitUntilCompleted();
        t.gpuMs += (cmdBuf->GPUEndTime() - cmdBuf->GPUStartTime()) * 1000.0;

        auto p0 = std::chrono::high_resolution_clock::now();
        onChunkDone((uint)rowsThisChunk, t.chunkCount);
        auto p1 = std::chrono::high_resolution_clock::now();
        t.postMs += std::chrono::duration<double, std::milli>(p1 - p0).count();

        t.chunkCount++;
        offset += rowsThisChunk;
    }

    return t;
}

// ===================================================================
// FORWARD DECLARATIONS — Query benchmark functions
// ===================================================================
void runQ1Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ2Benchmark(MTL::Device* pDevice, MTL::CommandQueue* pCommandQueue, MTL::Library* pLibrary);
void runQ3Benchmark(MTL::Device* pDevice, MTL::CommandQueue* pCommandQueue, MTL::Library* pLibrary);
void runQ5Benchmark(MTL::Device* pDevice, MTL::CommandQueue* pCommandQueue, MTL::Library* pLibrary);
void runQ6Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ9Benchmark(MTL::Device* pDevice, MTL::CommandQueue* pCommandQueue, MTL::Library* pLibrary);
void runQ12Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ13Benchmark(MTL::Device* pDevice, MTL::CommandQueue* pCommandQueue, MTL::Library* pLibrary);
void runQ14Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ19Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ4Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ11Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ7Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ8Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ10Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ15Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ16Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ17Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ18Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ20Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ21Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ22Benchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);

void runQ1BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ2BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ3BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ4BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ5BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ6BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ9BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ12BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ13BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ14BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ19BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ11BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ7BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ8BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ10BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ15BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ16BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ17BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ18BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ20BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ21BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runQ22BenchmarkSF100(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);

void runSelectionBenchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runAggregationBenchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runJoinBenchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
