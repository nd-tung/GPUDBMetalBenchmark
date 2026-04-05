#pragma once
#include "query_plan.h"
#include <string>

namespace codegen {

// Given a QueryPlan, generate Metal shader source code.
// Returns a Metal source string that can be compiled at runtime via device->newLibrary().
// Also populates kernel names for the executor to create pipeline states.
struct GeneratedKernels {
    std::string metalSource;

    struct KernelInfo {
        std::string name;
        int threadgroupSize = 1024; // suggested
        bool isSingleThread = false; // stage2 reduction kernels
    };
    std::vector<KernelInfo> kernels;
};

GeneratedKernels generateMetal(const QueryPlan& plan);

} // namespace codegen
