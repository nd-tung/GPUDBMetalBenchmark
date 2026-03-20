#include "infra.h"

static void showHelp() {
    std::cout << "GPU Database Metal Benchmark" << std::endl;
    std::cout << "Usage: GPUDBMetalBenchmark [sf1|sf10|sf100] [query]" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "Available queries:" << std::endl;
    std::cout << "  all           - Run all benchmarks (default)" << std::endl;
    std::cout << "  selection     - Run selection benchmark" << std::endl;
    std::cout << "  aggregation   - Run aggregation benchmark" << std::endl;
    std::cout << "  join          - Run join benchmark" << std::endl;
    std::cout << "  q1            - Run TPC-H Query 1 (Pricing Summary Report)" << std::endl;
    std::cout << "  q2            - Run TPC-H Query 2 (Minimum Cost Supplier)" << std::endl;
    std::cout << "  q3            - Run TPC-H Query 3 (Shipping Priority)" << std::endl;
    std::cout << "  q5            - Run TPC-H Query 5 (Local Supplier Volume)" << std::endl;
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

int main(int argc, const char * argv[]) {
    std::string query = "all";
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "help" || arg == "--help" || arg == "-h") {
            showHelp();
            return 0;
        }
        if (arg == "sf1") {
            g_dataset_path = "data/SF-1/";
            g_sf100_mode = false;
            continue;
        }
        if (arg == "sf10") {
            g_dataset_path = "data/SF-10/";
            g_sf100_mode = false;
            continue;
        }
        if (arg == "sf100") {
            g_dataset_path = "data/SF-100/";
            g_sf100_mode = true;
            continue;
        }
        query = arg;
    }

    NS::AutoreleasePool* pAutoreleasePool = NS::AutoreleasePool::alloc()->init();
    
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (device) {
        device->setShouldMaximizeConcurrentCompilation(true);
    }

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
        NS::String* libraryPath = NS::String::string("build/kernels.metallib", NS::UTF8StringEncoding);
        library = device->newLibrary(libraryPath, &error);
        libraryPath->release();
    }
    if (!library) {
        // Compile from source at runtime by concatenating all kernel files.
        const char* kernelFiles[] = {
            "kernels/common.h",
            "kernels/microbenchmarks.metal",
            "kernels/q1.metal",
            "kernels/q2.metal",
            "kernels/q3.metal",
            "kernels/q5.metal",
            "kernels/q6.metal",
            "kernels/q9.metal",
            "kernels/q13.metal"
        };
        std::string metalSource;
        bool allFilesRead = true;
        for (const char* path : kernelFiles) {
            std::ifstream f(path);
            if (!f.is_open()) { allFilesRead = false; break; }
            std::string line;
            while (std::getline(f, line)) {
                if (line.find("#include \"") != std::string::npos) continue;
                metalSource += line;
                metalSource += '\n';
            }
        }
        if (allFilesRead && !metalSource.empty()) {
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

    using BenchFn = void(*)(MTL::Device*, MTL::CommandQueue*, MTL::Library*);
    using BenchEntry = std::pair<std::string, BenchFn>;

    std::vector<BenchEntry> sf100_benchmarks = {
        {"q1", runQ1BenchmarkSF100}, {"q2", runQ2BenchmarkSF100},
        {"q3", runQ3BenchmarkSF100}, {"q5", runQ5BenchmarkSF100},
        {"q6", runQ6BenchmarkSF100}, {"q9", runQ9BenchmarkSF100},
        {"q13", runQ13BenchmarkSF100}
    };
    std::vector<BenchEntry> std_benchmarks = {
        {"selection", runSelectionBenchmark}, {"aggregation", runAggregationBenchmark},
        {"join", runJoinBenchmark},
        {"q1", runQ1Benchmark}, {"q2", runQ2Benchmark}, {"q3", runQ3Benchmark},
        {"q5", runQ5Benchmark}, {"q6", runQ6Benchmark}, {"q9", runQ9Benchmark},
        {"q13", runQ13Benchmark}
    };

    auto& benchmarks = g_sf100_mode ? sf100_benchmarks : std_benchmarks;
    if (query == "all") {
        for (auto& [name, fn] : benchmarks) fn(device, commandQueue, library);
    } else {
        auto it = std::find_if(benchmarks.begin(), benchmarks.end(),
                               [&](const BenchEntry& e) { return e.first == query; });
        if (it != benchmarks.end()) {
            it->second(device, commandQueue, library);
        } else {
            std::cerr << "Unknown query: " << query << std::endl;
            std::cerr << "Use 'help' to see available options." << std::endl;
            return 1;
        }
    }
    
    return 0;
}
