#pragma once
// ===================================================================
// Metal Generic Executor — data-driven GPU dispatch
// ===================================================================
//
// Reads MetalCodegen's bindings and result schema to generically:
// 1. Allocate GPU buffers based on symbolic size expressions
// 2. Bind buffers at auto-assigned [[buffer(N)]] indices
// 3. Dispatch each phase with correct threadgroup config
// 4. Collect results via MetalResultCollector
// ===================================================================

#include "metal_codegen_base.h"
#include "metal_result_collector.h"
#include "runtime_compiler.h"
#include <Metal/Metal.hpp>
#include <string>
#include <vector>
#include <unordered_map>

namespace codegen {

struct MetalExecutionResult {
    GenericResult result;
    float totalKernelTimeMs = 0.0f;
    std::vector<float> phaseTimesMs;
    float parseTimeMs = 0.0f;
    float postTimeMs = 0.0f;
};

class MetalGenericExecutor {
public:
    MetalGenericExecutor(MTL::Device* device, MTL::CommandQueue* cmdQueue);

    // Register table data from host memory
    void registerTable(const std::string& name, const void* data,
                       size_t rowCount, size_t bytesPerRow);

    // Register a single column buffer from host memory
    void registerColumn(const std::string& paramName, const void* data,
                        size_t rowCount, size_t bytesPerElem);

    // Register row count for a table (so n_tableName can be resolved)
    void registerTableRowCount(const std::string& tableName, size_t rowCount);

    // Register a pre-allocated Metal buffer
    void registerTableBuffer(const std::string& name, MTL::Buffer* buffer,
                             size_t rowCount);

    // Register a pre-allocated buffer into the cross-phase buffer map
    // (used for pre-computed data like bitmaps or lookup arrays)
    void registerAllocatedBuffer(const std::string& name, MTL::Buffer* buffer);

    // Register a symbolic size (e.g. "maxCustkey" → 150000)
    void registerSymbol(const std::string& name, size_t value);

    // Register a scalar constant (for constant T& params set via setBytes)
    void registerScalarInt(const std::string& name, int value);
    void registerScalarFloat(const std::string& name, float value);

    // Execute a compiled query through new codegen pipeline
    MetalExecutionResult execute(
        const RuntimeCompiler::CompiledQuery& compiled,
        const MetalCodegen& codegen,
        int warmupRuns = 2,
        int measuredRuns = 1
    );

    // Clean up allocated buffers (call after execute, results consumed)
    void releaseAllocatedBuffers();

    // Debug: access an allocated buffer by name
    MTL::Buffer* getAllocatedBuffer(const std::string& name) const {
        auto it = allocatedBuffers_.find(name);
        return it != allocatedBuffers_.end() ? it->second : nullptr;
    }

private:
    MTL::Device* device_;
    MTL::CommandQueue* cmdQueue_;
    MetalSizeResolver sizeResolver_;

    // Registered tables: name → {buffer, rowCount}
    struct TableInfo {
        MTL::Buffer* buffer = nullptr;
        size_t rowCount = 0;
        bool ownsBuffer = false;  // if we allocated it
    };
    std::unordered_map<std::string, TableInfo> tables_;

    // Buffers allocated for scratch/output (we own these)
    std::unordered_map<std::string, MTL::Buffer*> allocatedBuffers_;

    // Scalar constant values (set via setBytes during binding)
    std::unordered_map<std::string, int> scalarInts_;
    std::unordered_map<std::string, float> scalarFloats_;

    // Allocate all buffers needed by a phase
    BufferMap allocatePhaseBuffers(const MetalCodegen::PhaseInfo& phase);

    // Bind all parameters for a phase
    void bindPhaseBuffers(MTL::ComputeCommandEncoder* encoder,
                          const MetalCodegen::PhaseInfo& phase,
                          const BufferMap& buffers);

    // Zero-init buffers that require it
    void zeroInitBuffers(const MetalCodegen::PhaseInfo& phase,
                         const BufferMap& buffers);

    // Find PSO by kernel name
    MTL::ComputePipelineState* findPSO(const RuntimeCompiler::CompiledQuery& cq,
                                        const std::string& name);
};

} // namespace codegen
