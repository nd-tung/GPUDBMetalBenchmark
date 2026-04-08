#include "runtime_compiler.h"
#include <Foundation/Foundation.hpp>
#include <iostream>
#include <sstream>

namespace codegen {

RuntimeCompiler::~RuntimeCompiler() {
    for (auto& [name, pso] : pipelineCache_)
        pso->release();
}

MTL::Library* RuntimeCompiler::compile(const std::string& source) {
    NS::Error* error = nullptr;
    auto* sourceStr = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    auto* opts = MTL::CompileOptions::alloc()->init();
    auto* library = device_->newLibrary(sourceStr, opts, &error);
    opts->release();

    if (!library) {
        std::cerr << "Metal compilation failed:" << std::endl;
        if (error)
            std::cerr << error->localizedDescription()->utf8String() << std::endl;
        // Print source with line numbers for debugging
        std::istringstream ss(source);
        std::string line;
        int lineNo = 1;
        while (std::getline(ss, line)) {
            std::cerr << lineNo++ << ": " << line << "\n";
        }
        return nullptr;
    }
    return library;
}

MTL::ComputePipelineState* RuntimeCompiler::getPipeline(MTL::Library* lib, const std::string& kernelName) {
    auto it = pipelineCache_.find(kernelName);
    if (it != pipelineCache_.end()) return it->second;

    auto* funcName = NS::String::string(kernelName.c_str(), NS::UTF8StringEncoding);
    auto* func = lib->newFunction(funcName);
    if (!func) {
        std::cerr << "Kernel function not found: " << kernelName << std::endl;
        return nullptr;
    }

    NS::Error* error = nullptr;
    auto* pso = device_->newComputePipelineState(func, &error);
    func->release();

    if (!pso) {
        std::cerr << "Failed to create pipeline for " << kernelName << ": ";
        if (error) std::cerr << error->localizedDescription()->utf8String();
        std::cerr << std::endl;
        return nullptr;
    }

    pipelineCache_[kernelName] = pso;
    return pso;
}

} // namespace codegen
