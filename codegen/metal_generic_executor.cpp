#include "metal_generic_executor.h"
#include <iostream>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace codegen {

// ===================================================================
// Construction
// ===================================================================

MetalGenericExecutor::MetalGenericExecutor(MTL::Device* device, MTL::CommandQueue* cmdQueue)
    : device_(device), cmdQueue_(cmdQueue) {}

// ===================================================================
// Table registration
// ===================================================================

void MetalGenericExecutor::registerTable(const std::string& name, const void* data,
                                          size_t rowCount, size_t bytesPerRow) {
    size_t totalBytes = rowCount * bytesPerRow;
    auto* buf = device_->newBuffer(data, totalBytes, MTL::ResourceStorageModeShared);
    tables_[name] = {buf, rowCount, true};

    // Register size symbol: n_{table} and num{Table}
    sizeResolver_.registerSymbol("n_" + name, rowCount);
    sizeResolver_.registerSymbol("num" + name, rowCount);
}

void MetalGenericExecutor::registerColumn(const std::string& paramName, const void* data,
                                           size_t rowCount, size_t bytesPerElem) {
    size_t totalBytes = rowCount * bytesPerElem;
    auto* buf = device_->newBuffer(data, totalBytes, MTL::ResourceStorageModeShared);
    tables_[paramName] = {buf, rowCount, true};
}

void MetalGenericExecutor::registerTableRowCount(const std::string& tableName, size_t rowCount) {
    sizeResolver_.registerSymbol("n_" + tableName, rowCount);
    sizeResolver_.registerSymbol("num" + tableName, rowCount);
}

void MetalGenericExecutor::registerTableBuffer(const std::string& name,
                                                MTL::Buffer* buffer,
                                                size_t rowCount) {
    tables_[name] = {buffer, rowCount, false};
    sizeResolver_.registerSymbol("n_" + name, rowCount);
    sizeResolver_.registerSymbol("num" + name, rowCount);
}

void MetalGenericExecutor::registerAllocatedBuffer(const std::string& name, MTL::Buffer* buffer) {
    allocatedBuffers_[name] = buffer;
}

void MetalGenericExecutor::registerSymbol(const std::string& name, size_t value) {
    sizeResolver_.registerSymbol(name, value);
}
void MetalGenericExecutor::registerScalarInt(const std::string& name, int value) {
    scalarInts_[name] = value;
}

void MetalGenericExecutor::registerScalarFloat(const std::string& name, float value) {
    scalarFloats_[name] = value;
}
// ===================================================================
// Find PSO by name
// ===================================================================

MTL::ComputePipelineState* MetalGenericExecutor::findPSO(
    const RuntimeCompiler::CompiledQuery& cq, const std::string& name) {
    for (size_t i = 0; i < cq.kernelNames.size(); i++)
        if (cq.kernelNames[i] == name) return cq.pipelines[i];
    return nullptr;
}

// ===================================================================
// Buffer allocation
// ===================================================================

BufferMap MetalGenericExecutor::allocatePhaseBuffers(
    const MetalCodegen::PhaseInfo& phase) {

    BufferMap buffers;

    for (const auto& b : phase.bindings) {
        switch (b.kind) {
            case MetalParamKind::TableData: {
                // Try looking up by binding name first (columnar: "l_shipdate"),
                // then by table name (AoS: "lineitem")
                auto tIt = tables_.find(b.name);
                if (tIt == tables_.end()) tIt = tables_.find(b.tableName);
                if (tIt != tables_.end()) {
                    buffers[b.name] = tIt->second.buffer;
                } else {
                    std::cerr << "Warning: table/column '" << b.name
                              << "' not registered\n";
                }
                break;
            }

            case MetalParamKind::TableSize: {
                // Table sizes are passed via setBytes, not as buffers.
                // We still track them in the buffer map for binding.
                // Look up row count via registered columns or tables.
                size_t rowCount = 0;
                auto tIt = tables_.find(b.tableName);
                if (tIt != tables_.end()) {
                    rowCount = tIt->second.rowCount;
                } else {
                    // Try size resolver
                    std::string symName = "n_" + b.tableName;
                    if (sizeResolver_.hasSymbol(symName)) {
                        rowCount = sizeResolver_.getSymbol(symName);
                    }
                }

                if (rowCount > 0) {
                    uint32_t sz = static_cast<uint32_t>(rowCount);
                    std::string key = b.name;
                    if (allocatedBuffers_.count(key)) {
                        memcpy(allocatedBuffers_[key]->contents(), &sz, sizeof(uint32_t));
                        buffers[key] = allocatedBuffers_[key];
                    } else {
                        auto* buf = device_->newBuffer(sizeof(uint32_t),
                                                       MTL::ResourceStorageModeShared);
                        memcpy(buf->contents(), &sz, sizeof(uint32_t));
                        allocatedBuffers_[key] = buf;
                        buffers[key] = buf;
                    }
                }
                break;
            }

            case MetalParamKind::DeviceBuffer: {
                std::string key = b.name;
                if (allocatedBuffers_.count(key)) {
                    // Already allocated (shared across phases)
                    buffers[key] = allocatedBuffers_[key];
                } else if (!b.sizeExpr.empty()) {
                    size_t count = sizeResolver_.resolve(b.sizeExpr);
                    // Determine element size from type
                    size_t elemSize = 4; // default: uint/int/float
                    if (b.elementType == "long" || b.elementType == "double" ||
                        b.elementType == "ulong")
                        elemSize = 8;
                    else if (b.elementType == "char" || b.elementType == "uchar")
                        elemSize = 1;
                    else if (b.elementType == "short" || b.elementType == "ushort")
                        elemSize = 2;

                    size_t totalBytes = count * elemSize;
                    if (totalBytes == 0) totalBytes = elemSize; // minimum 1 element
                    auto* buf = device_->newBuffer(totalBytes,
                                                   MTL::ResourceStorageModeShared);
                    allocatedBuffers_[key] = buf;
                    buffers[key] = buf;
                }
                break;
            }

            case MetalParamKind::ConstantScalar:
            case MetalParamKind::ConstantData:
                // These are set via setBytes, handled during binding
                break;
        }
    }

    return buffers;
}

// ===================================================================
// Zero-init buffers
// ===================================================================

void MetalGenericExecutor::zeroInitBuffers(const MetalCodegen::PhaseInfo& phase,
                                            const BufferMap& buffers) {
    for (const auto& b : phase.bindings) {
        if (b.zeroInit && b.kind == MetalParamKind::DeviceBuffer) {
            auto it = buffers.find(b.name);
            if (it != buffers.end() && it->second) {
                memset(it->second->contents(), b.fillByte, it->second->length());
            }
        }
    }
}

// ===================================================================
// Bind buffers to encoder
// ===================================================================

void MetalGenericExecutor::bindPhaseBuffers(MTL::ComputeCommandEncoder* encoder,
                                             const MetalCodegen::PhaseInfo& phase,
                                             const BufferMap& buffers) {
    for (const auto& b : phase.bindings) {
        if (b.bufferIndex < 0) continue;

        switch (b.kind) {
            case MetalParamKind::TableData:
            case MetalParamKind::DeviceBuffer: {
                auto it = buffers.find(b.name);
                if (it != buffers.end() && it->second) {
                    encoder->setBuffer(it->second, 0, b.bufferIndex);
                }
                break;
            }

            case MetalParamKind::TableSize: {
                auto it = buffers.find(b.name);
                if (it != buffers.end() && it->second) {
                    encoder->setBuffer(it->second, 0, b.bufferIndex);
                }
                break;
            }

            case MetalParamKind::ConstantScalar: {
                // Look up registered scalar value and set via setBytes
                auto ii = scalarInts_.find(b.name);
                if (ii != scalarInts_.end()) {
                    encoder->setBytes(&ii->second, sizeof(int), b.bufferIndex);
                } else {
                    auto fi = scalarFloats_.find(b.name);
                    if (fi != scalarFloats_.end()) {
                        encoder->setBytes(&fi->second, sizeof(float), b.bufferIndex);
                    }
                }
                break;
            }

            case MetalParamKind::ConstantData:
                // Caller must set these manually via setBytes
                break;
        }
    }
}

// ===================================================================
// Execute
// ===================================================================

MetalExecutionResult MetalGenericExecutor::execute(
    const RuntimeCompiler::CompiledQuery& compiled,
    const MetalCodegen& codegen,
    int warmupRuns,
    int measuredRuns) {

    MetalExecutionResult execResult;
    const auto& phases = codegen.getPhases();

    if (phases.empty()) {
        std::cerr << "MetalGenericExecutor: no phases to execute\n";
        return execResult;
    }

    // Pre-allocate all buffers across all phases
    BufferMap allBuffers;
    for (const auto& phase : phases) {
        auto phaseBuffers = allocatePhaseBuffers(phase);
        for (auto& [k, v] : phaseBuffers) {
            if (!allBuffers.count(k))
                allBuffers[k] = v;
        }
    }

    int totalRuns = warmupRuns + measuredRuns;

    for (int iter = 0; iter < totalRuns; iter++) {
        // Zero-init output buffers each iteration
        for (const auto& phase : phases) {
            zeroInitBuffers(phase, allBuffers);
        }

        auto* cmdBuf = cmdQueue_->commandBuffer();
        auto* encoder = cmdBuf->computeCommandEncoder();

        std::vector<float> phaseTimes;

        for (size_t pi = 0; pi < phases.size(); pi++) {
            const auto& phase = phases[pi];
            auto* pso = findPSO(compiled, phase.name);
            if (!pso) {
                std::cerr << "MetalGenericExecutor: PSO not found for '"
                          << phase.name << "'\n";
                continue;
            }

            encoder->setComputePipelineState(pso);
            bindPhaseBuffers(encoder, phase, allBuffers);

            // Dispatch configuration
            NS::UInteger tgSize = pso->maxTotalThreadsPerThreadgroup();
            if (tgSize > (NS::UInteger)phase.threadgroupSize)
                tgSize = phase.threadgroupSize;

            if (phase.isSingleThread) {
                encoder->dispatchThreadgroups(MTL::Size::Make(1, 1, 1),
                                              MTL::Size::Make(1, 1, 1));
            } else {
                // Default: 1024 threadgroups for grid-stride kernels
                NS::UInteger numTG = 1024;
                encoder->dispatchThreadgroups(MTL::Size::Make(numTG, 1, 1),
                                              MTL::Size::Make(tgSize, 1, 1));
            }

            // Add memory barrier between phases
            if (pi + 1 < phases.size()) {
                encoder->memoryBarrier(MTL::BarrierScopeBuffers);
            }
        }

        encoder->endEncoding();
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();

        // Measure last run
        if (iter == totalRuns - 1) {
            execResult.totalKernelTimeMs =
                static_cast<float>((cmdBuf->GPUEndTime() - cmdBuf->GPUStartTime()) * 1000.0);
        }
    }

    // Collect results
    execResult.result = MetalResultCollector::collect(codegen.getResultSchema(), allBuffers);

    return execResult;
}

// ===================================================================
// Cleanup
// ===================================================================

void MetalGenericExecutor::releaseAllocatedBuffers() {
    for (auto& [_, buf] : allocatedBuffers_) {
        if (buf) buf->release();
    }
    allocatedBuffers_.clear();

    for (auto& [_, info] : tables_) {
        if (info.ownsBuffer && info.buffer)
            info.buffer->release();
    }
    tables_.clear();
}

} // namespace codegen
