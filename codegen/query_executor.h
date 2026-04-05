#pragma once
#include "query_plan.h"
#include "metal_codegen.h"
#include "runtime_compiler.h"
#include <Metal/Metal.hpp>
#include <string>

namespace codegen {

// Execute a compiled query plan on the GPU.
// Loads data from the given data directory, allocates GPU buffers,
// dispatches kernels, reads results, and prints output.
void executeQuery(MTL::Device* device, MTL::CommandQueue* cmdQueue,
                  const QueryPlan& plan,
                  const RuntimeCompiler::CompiledQuery& compiled,
                  const GeneratedKernels& gen,
                  const std::string& dataDir);

} // namespace codegen
