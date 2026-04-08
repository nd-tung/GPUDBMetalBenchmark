#pragma once
// ===================================================================
// Metal Result Collector — generic GPU→CPU result collection
// ===================================================================

#include "metal_param_binding.h"
#include <Metal/Metal.hpp>
#include <string>
#include <vector>
#include <variant>
#include <unordered_map>

namespace codegen {

// ===================================================================
// Generic result container (variant-typed heterogeneous rows)
// ===================================================================

struct GenericResult {
    struct Column {
        std::string name;
        std::string type;  // "int", "float", "long", "string"
    };
    std::vector<Column> columns;

    using Value = std::variant<int64_t, double, std::string>;
    using Row = std::vector<Value>;
    std::vector<Row> rows;

    void print(int limit = -1) const;
    bool empty() const { return rows.empty(); }
    size_t numRows() const { return rows.size(); }
};

// ===================================================================
// Collector
// ===================================================================

using BufferMap = std::unordered_map<std::string, MTL::Buffer*>;

class MetalResultCollector {
public:
    static GenericResult collect(const MetalResultSchema& schema,
                                 const BufferMap& buffers);

private:
    static GenericResult collectScalarAgg(const MetalResultSchema& schema,
                                          const BufferMap& buffers);
    static GenericResult collectKeyedAgg(const MetalResultSchema& schema,
                                         const BufferMap& buffers);
    static GenericResult collectMaterialize(const MetalResultSchema& schema,
                                            const BufferMap& buffers);

    static int64_t reconstructLong(uint32_t lo, uint32_t hi);
};

} // namespace codegen
