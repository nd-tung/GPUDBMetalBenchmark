#pragma once

// Metal-cpp headers (no PRIVATE_IMPLEMENTATION here — that lives in infra.cpp)
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

// Round up to next power of 2 (host-side, must match GPU-side next_pow2)
inline uint nextPow2(uint v) {
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}

// Global dataset configuration
extern std::string g_dataset_path;
extern bool g_sf100_mode;

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
inline MTL::ComputePipelineState* createPipeline(MTL::Device* device, MTL::Library* library, const char* name) {
    NS::Error* error = nullptr;
    auto fn = library->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
    if (!fn) { std::cerr << "Kernel not found: " << name << std::endl; return nullptr; }
    auto pso = device->newComputePipelineState(fn, &error);
    fn->release();
    if (!pso) { std::cerr << "Failed to create pipeline: " << name << std::endl; }
    return pso;
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

// --- Timing Summary ---
inline void printTimingSummary(double parseMs, double gpuMs, double postMs) {
    printf("  CPU Parsing (.tbl): %10.2f ms\n", parseMs);
    printf("  GPU Execution:      %10.2f ms\n", gpuMs);
    printf("  CPU Post Process:   %10.2f ms\n", postMs);
    printf("  Total Execution:    %10.2f ms  (GPU + CPU post)\n", gpuMs + postMs);
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

void runSelectionBenchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runAggregationBenchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
void runJoinBenchmark(MTL::Device* device, MTL::CommandQueue* commandQueue, MTL::Library* library);
