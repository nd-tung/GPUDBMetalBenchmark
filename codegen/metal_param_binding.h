#pragma once
// ===================================================================
// Metal Param Binding — metadata-driven buffer management
// ===================================================================
//
// Adapts the CUDA ParamBinding concept to Metal's buffer/constant model.
// Each binding describes one kernel parameter: its Metal type, memory
// management, symbolic size, and auto-assigned [[buffer(N)]] index.
// ===================================================================

#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <cstddef>

namespace codegen {

// What kind of parameter is it?
enum class MetalParamKind {
    TableData,        // device const T* — read-only table column / row array
    TableSize,        // constant uint&  — row count for a table
    DeviceBuffer,     // device T*       — read-write scratch / output buffer
    ConstantScalar,   // constant T&     — small value passed via setBytes()
    ConstantData,     // constant T*     — host data copied via setBytes()
};

struct MetalParamBinding {
    std::string name;            // e.g. "d_lineitem_shipdate", "data_size"
    std::string metalDecl;       // e.g. "device const int*", "constant uint&"
    MetalParamKind kind;
    std::string tableName;       // for TableData/TableSize: which table
    std::string elementType;     // e.g. "float", "uint", "atomic_uint"
    std::string sizeExpr;        // symbolic: "numLineItems", "maxCustkey + 1"
    bool zeroInit = false;       // zero-fill on allocation
    int fillByte = 0;            // byte value for memset (0 for zero, 0xFF for -1 sentinel)
    bool readOnly = false;       // hint for storage mode
    size_t hostCopyBytes = 0;    // for ConstantData: how many bytes
    int bufferIndex = -1;        // auto-assigned [[buffer(N)]]
};

// ===================================================================
// Size Resolver — symbolic name → actual byte count
// ===================================================================

class MetalSizeResolver {
public:
    void registerSymbol(const std::string& name, size_t value) {
        symbols_[name] = value;
    }

    bool hasSymbol(const std::string& name) const {
        return symbols_.count(name) > 0;
    }

    size_t getSymbol(const std::string& name) const {
        auto it = symbols_.find(name);
        if (it == symbols_.end())
            throw std::runtime_error("MetalSizeResolver: unknown symbol '" + name + "'");
        return it->second;
    }

    // Resolve a simple expression like "maxCustkey + 1" or "numLineItems"
    // Supports: constant, symbol, symbol + constant, symbol * constant
    size_t resolve(const std::string& expr) const;

private:
    std::unordered_map<std::string, size_t> symbols_;
};

// ===================================================================
// Result Schema — describes what a query produces
// ===================================================================

struct MetalResultSchema {
    enum Kind { NONE, MATERIALIZE, SCALAR_AGG, KEYED_AGG };
    Kind kind = NONE;

    struct ColumnDesc {
        std::string displayName;    // "l_returnflag"
        std::string bufferName;     // "d_returnflag"
        std::string elementType;    // "uint", "float", "char"
        int stringLen = 0;          // for fixed-width strings
        bool isLongPair = false;    // needs lo/hi reconstruction
    };
    std::vector<ColumnDesc> columns;

    // MATERIALIZE: output rows scattered to arrays
    std::string counterBuffer;      // "d_resultCount"

    // SCALAR_AGG: single-row output
    struct ScalarAggEntry {
        std::string displayName;
        std::string loBuffer;       // lo part (or main buffer if not long pair)
        std::string hiBuffer;       // hi part (empty if not long pair)
        std::string elementType;    // "uint" for long pair, "float" for direct
        bool isLongPair = false;
        int scaleDown = 0;          // divide by 10^scaleDown (e.g. 2 for cents→dollars)
    };
    std::vector<ScalarAggEntry> scalarAggs;

    // KEYED_AGG: N-bucket output
    struct KeyedAggSlot {
        std::string name;           // display name
        int offset;                 // slot offset within bucket
        bool isLongPair = false;    // true → lo/hi at offset/offset+1
        int scaleDown = 0;          // divisor for fixed-point
    };
    struct KeyedAggInfo {
        int numBuckets = 0;
        int valuesPerBucket = 0;
        std::string bufferName;
        std::vector<KeyedAggSlot> slots; // describes each logical value
    };
    KeyedAggInfo keyedAgg;
};

} // namespace codegen
