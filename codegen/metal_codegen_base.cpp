#include "metal_codegen_base.h"
#include <sstream>
#include <algorithm>
#include <stdexcept>

namespace codegen {

// ===================================================================
// Common Metal header — SIMD reductions, bitmap ops, atomics
// ===================================================================

std::string MetalCodegen::commonHeader() {
    return R"METAL(#include <metal_stdlib>
using namespace metal;

// --- SIMD reduction for long (int64) via 2×uint shuffle ---
inline long simd_reduce_add_long(long v) {
    for (uint d = 16; d >= 1; d >>= 1) {
        uint lo = simd_shuffle_down((uint)(v), d);
        uint hi = simd_shuffle_down((uint)((ulong)v >> 32), d);
        v += (long)(((ulong)hi << 32) | (ulong)lo);
    }
    return v;
}

inline float tg_reduce_float(float val, uint tid, uint tg_size,
                             threadgroup float* shared) {
    float sv = simd_sum(val);
    uint lane = tid & 31u;
    uint gid  = tid >> 5u;
    uint ng   = (tg_size + 31u) >> 5u;
    if (lane == 0u) shared[gid] = sv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = 0.0f;
    if (gid == 0u) {
        float v2 = (lane < ng) ? shared[lane] : 0.0f;
        r = simd_sum(v2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

inline uint tg_reduce_uint(uint val, uint tid, uint tg_size,
                           threadgroup uint* shared) {
    uint sv = simd_sum(val);
    uint lane = tid & 31u;
    uint gid  = tid >> 5u;
    uint ng   = (tg_size + 31u) >> 5u;
    if (lane == 0u) shared[gid] = sv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint r = 0u;
    if (gid == 0u) {
        uint v2 = (lane < ng) ? shared[lane] : 0u;
        r = simd_sum(v2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

inline long tg_reduce_long(long val, uint tid, uint tg_size,
                           threadgroup long* shared) {
    long sv = simd_reduce_add_long(val);
    uint lane = tid & 31u;
    uint gid  = tid >> 5u;
    uint ng   = (tg_size + 31u) >> 5u;
    if (lane == 0u) shared[gid] = sv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    long r = 0;
    if (gid == 0u) {
        long v2 = (lane < ng) ? shared[lane] : 0;
        r = simd_reduce_add_long(v2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

inline bool bitmap_test(const device uint* bitmap, int key) {
    return (bitmap[(uint)key >> 5] >> ((uint)key & 31u)) & 1u;
}

inline void bitmap_set(device atomic_uint* bitmap, int key) {
    atomic_fetch_or_explicit(&bitmap[(uint)key >> 5],
                             1u << ((uint)key & 31u),
                             memory_order_relaxed);
}

inline void atomic_add_long_pair(device atomic_uint* lo,
                                 device atomic_uint* hi,
                                 long val) {
    ulong uval = as_type<ulong>(val);
    uint add_lo = (uint)(uval);
    uint add_hi = (uint)(uval >> 32);
    uint old_lo = atomic_fetch_add_explicit(lo, add_lo, memory_order_relaxed);
    uint new_lo = old_lo + add_lo;
    uint carry = (new_lo < old_lo) ? 1u : 0u;
    if (add_hi != 0 || carry != 0)
        atomic_fetch_add_explicit(hi, add_hi + carry, memory_order_relaxed);
}

inline long load_long_pair(const device uint* lo, const device uint* hi) {
    ulong v = ((ulong)(*hi) << 32) | (ulong)(*lo);
    return as_type<long>(v);
}

inline uint next_pow2(uint v) {
    v--; v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}

)METAL";
}

// ===================================================================
// Indentation
// ===================================================================

std::string MetalCodegen::indent() const {
    return std::string(indentLevel_ * INDENT_SIZE, ' ');
}

// ===================================================================
// Code emission
// ===================================================================

void MetalCodegen::addLine(const std::string& line) {
    if (!currentPhase_)
        throw std::runtime_error("MetalCodegen::addLine: no active phase");
    currentPhase_->code += indent() + line + "\n";
}

void MetalCodegen::addBlock(const std::string& header, std::function<void()> body,
                             const std::string& trailing) {
    if (!currentPhase_)
        throw std::runtime_error("MetalCodegen::addBlock: no active phase");
    currentPhase_->code += indent() + header + " {\n";
    indentLevel_++;
    body();
    indentLevel_--;
    currentPhase_->code += indent() + "}" + trailing + "\n";
}

void MetalCodegen::addIf(const std::string& condition, std::function<void()> body) {
    addBlock("if (" + condition + ")", body);
}

void MetalCodegen::addComment(const std::string& comment) {
    if (!currentPhase_)
        throw std::runtime_error("MetalCodegen::addComment: no active phase");
    currentPhase_->code += indent() + "// " + comment + "\n";
}

void MetalCodegen::increaseIndent() { indentLevel_++; }
void MetalCodegen::decreaseIndent() { if (indentLevel_ > 0) indentLevel_--; }

void MetalCodegen::addRawCode(const std::string& code) {
    if (!currentPhase_)
        throw std::runtime_error("MetalCodegen::addRawCode: no active phase");
    currentPhase_->code += code;
}

void MetalCodegen::addHelper(const std::string& code) {
    helperCode_ += code + "\n";
}

// ===================================================================
// Phase management
// ===================================================================

void MetalCodegen::beginPhase(const std::string& phaseName) {
    if (currentPhase_)
        throw std::runtime_error("MetalCodegen::beginPhase: previous phase '" +
                                 currentPhase_->name + "' not ended");
    phases_.push_back({});
    currentPhase_ = &phases_.back();
    currentPhase_->name = phaseName;
    indentLevel_ = 1;
}

void MetalCodegen::endPhase() {
    if (!currentPhase_)
        throw std::runtime_error("MetalCodegen::endPhase: no active phase");
    currentPhase_ = nullptr;
    indentLevel_ = 1;
}

int MetalCodegen::phaseCount() const {
    return static_cast<int>(phases_.size());
}

void MetalCodegen::setPhaseScannedTable(const std::string& tableName) {
    if (!currentPhase_)
        throw std::runtime_error("setPhaseScannedTable: no active phase");
    currentPhase_->scannedTable = tableName;
}

void MetalCodegen::setPhaseThreadgroupSize(int size) {
    if (!currentPhase_)
        throw std::runtime_error("setPhaseThreadgroupSize: no active phase");
    currentPhase_->threadgroupSize = size;
}

void MetalCodegen::setPhaseSingleThread(bool single) {
    if (!currentPhase_)
        throw std::runtime_error("setPhaseSingleThread: no active phase");
    currentPhase_->isSingleThread = single;
}

// ===================================================================
// Parameter registration
// ===================================================================

void MetalCodegen::addTableParam(const std::string& table, const std::string& metalType) {
    if (!currentPhase_)
        throw std::runtime_error("addTableParam: no active phase");

    // Data pointer
    {
        MetalParamBinding b;
        b.name = "d_" + table;
        b.metalDecl = "device const " + metalType + "*";
        b.kind = MetalParamKind::TableData;
        b.tableName = table;
        b.elementType = metalType;
        b.readOnly = true;
        currentPhase_->bindings.push_back(b);
    }
    // Size
    {
        MetalParamBinding b;
        b.name = "n_" + table;
        b.metalDecl = "constant uint&";
        b.kind = MetalParamKind::TableSize;
        b.tableName = table;
        b.elementType = "uint";
        currentPhase_->bindings.push_back(b);
    }

    registeredTables_.insert(table);
}

void MetalCodegen::addColumnParam(const std::string& paramName, const std::string& metalType,
                                   const std::string& tableName) {
    if (!currentPhase_)
        throw std::runtime_error("addColumnParam: no active phase");
    MetalParamBinding b;
    b.name = paramName;
    b.metalDecl = "device const " + metalType + "*";
    b.kind = MetalParamKind::TableData;
    b.tableName = tableName.empty() ? paramName : tableName;
    b.elementType = metalType;
    b.readOnly = true;
    currentPhase_->bindings.push_back(b);
}

void MetalCodegen::addTableSizeParam(const std::string& table) {
    if (!currentPhase_)
        throw std::runtime_error("addTableSizeParam: no active phase");
    MetalParamBinding b;
    b.name = "n_" + table;
    b.metalDecl = "constant uint&";
    b.kind = MetalParamKind::TableSize;
    b.tableName = table;
    b.elementType = "uint";
    currentPhase_->bindings.push_back(b);
}

void MetalCodegen::addBufferParam(const std::string& name, const std::string& elemType,
                                   const std::string& sizeExpr, bool zeroInit, int fillByte) {
    if (!currentPhase_)
        throw std::runtime_error("addBufferParam: no active phase");
    // Skip if already registered in this phase
    for (const auto& existing : currentPhase_->bindings)
        if (existing.name == name) return;
    MetalParamBinding b;
    b.name = name;
    b.metalDecl = "device " + elemType + "*";
    b.kind = MetalParamKind::DeviceBuffer;
    b.elementType = elemType;
    b.sizeExpr = sizeExpr;
    b.zeroInit = zeroInit;
    b.fillByte = fillByte;
    currentPhase_->bindings.push_back(b);

    if (!sizeExpr.empty())
        globalBufferSizes_[name] = sizeExpr;
}

void MetalCodegen::addAtomicBufferParam(const std::string& name,
                                         const std::string& atomicType,
                                         const std::string& sizeExpr) {
    if (!currentPhase_)
        throw std::runtime_error("addAtomicBufferParam: no active phase");
    // Skip if already registered in this phase
    for (const auto& existing : currentPhase_->bindings)
        if (existing.name == name) return;
    MetalParamBinding b;
    b.name = name;
    b.metalDecl = "device " + atomicType + "*";
    b.kind = MetalParamKind::DeviceBuffer;
    b.elementType = atomicType;
    b.sizeExpr = sizeExpr;
    b.zeroInit = true;
    currentPhase_->bindings.push_back(b);

    if (!sizeExpr.empty())
        globalBufferSizes_[name] = sizeExpr;
}

void MetalCodegen::addScalarParam(const std::string& name, const std::string& type) {
    if (!currentPhase_)
        throw std::runtime_error("addScalarParam: no active phase");
    for (const auto& existing : currentPhase_->bindings)
        if (existing.name == name) return;
    MetalParamBinding b;
    b.name = name;
    b.metalDecl = "constant " + type + "&";
    b.kind = MetalParamKind::ConstantScalar;
    b.elementType = type;
    currentPhase_->bindings.push_back(b);
}

void MetalCodegen::addConstantDataParam(const std::string& name, const std::string& type,
                                         size_t bytes) {
    if (!currentPhase_)
        throw std::runtime_error("addConstantDataParam: no active phase");
    MetalParamBinding b;
    b.name = name;
    b.metalDecl = "constant " + type + "*";
    b.kind = MetalParamKind::ConstantData;
    b.elementType = type;
    b.hostCopyBytes = bytes;
    currentPhase_->bindings.push_back(b);
}

void MetalCodegen::addBitmapReadParam(const std::string& name, const std::string& sizeExpr) {
    if (!currentPhase_)
        throw std::runtime_error("addBitmapReadParam: no active phase");
    MetalParamBinding b;
    b.name = name;
    b.metalDecl = "device const uint*";
    b.kind = MetalParamKind::DeviceBuffer;
    b.elementType = "uint";
    b.sizeExpr = sizeExpr;
    b.readOnly = true;
    currentPhase_->bindings.push_back(b);
}

void MetalCodegen::addBitmapWriteParam(const std::string& name, const std::string& sizeExpr) {
    addAtomicBufferParam(name, "atomic_uint", sizeExpr);
}

// ===================================================================
// Global buffer sizes
// ===================================================================

void MetalCodegen::setBufferSize(const std::string& name, const std::string& sizeExpr) {
    globalBufferSizes_[name] = sizeExpr;
}

const std::unordered_map<std::string, std::string>& MetalCodegen::getGlobalBufferSizes() const {
    return globalBufferSizes_;
}

// ===================================================================
// Output schema registration
// ===================================================================

void MetalCodegen::registerScalarAggOutput(const std::string& loBuffer,
                                            const std::string& hiBuffer,
                                            const std::string& type) {
    resultSchema_.kind = MetalResultSchema::SCALAR_AGG;
    // Store the buffer names for the next scalarAgg entry
    // (will be associated with the column added by registerScalarAggColumn)
    (void)type;
    // We store lo/hi buffers temporarily — they'll be set on the scalarAggs entry
    // by registerScalarAggColumn which is called right after
    scalarAggPendingLo_ = loBuffer;
    scalarAggPendingHi_ = hiBuffer;
}

void MetalCodegen::registerScalarAggColumn(const std::string& displayName, int index,
                                            int scaleDown) {
    resultSchema_.kind = MetalResultSchema::SCALAR_AGG;
    MetalResultSchema::ScalarAggEntry entry;
    entry.displayName = displayName;
    entry.scaleDown = scaleDown;
    entry.loBuffer = scalarAggPendingLo_;
    entry.hiBuffer = scalarAggPendingHi_;
    entry.isLongPair = !scalarAggPendingHi_.empty();
    entry.elementType = entry.isLongPair ? "uint" : "float";
    (void)index;
    resultSchema_.scalarAggs.push_back(entry);
}

void MetalCodegen::registerMaterializeOutput(const std::string& counterBuffer) {
    resultSchema_.kind = MetalResultSchema::MATERIALIZE;
    resultSchema_.counterBuffer = counterBuffer;
}

void MetalCodegen::registerOutputColumn(const std::string& displayName,
                                         const std::string& bufferName,
                                         const std::string& elementType,
                                         int stringLen) {
    MetalResultSchema::ColumnDesc col;
    col.displayName = displayName;
    col.bufferName = bufferName;
    col.elementType = elementType;
    col.stringLen = stringLen;
    resultSchema_.columns.push_back(col);
}

void MetalCodegen::registerKeyedAggOutput(const std::string& bufferName,
                                           int numBuckets, int valuesPerBucket,
                                           const std::vector<MetalResultSchema::KeyedAggSlot>& slots) {
    resultSchema_.kind = MetalResultSchema::KEYED_AGG;
    resultSchema_.keyedAgg.bufferName = bufferName;
    resultSchema_.keyedAgg.numBuckets = numBuckets;
    resultSchema_.keyedAgg.valuesPerBucket = valuesPerBucket;
    resultSchema_.keyedAgg.slots = slots;
}

const MetalResultSchema& MetalCodegen::getResultSchema() const {
    return resultSchema_;
}

// ===================================================================
// Buffer index assignment
// ===================================================================

void MetalCodegen::assignBufferIndices(PhaseInfo& phase) const {
    int nextIndex = 0;
    for (auto& b : phase.bindings) {
        b.bufferIndex = nextIndex++;
    }
}

// ===================================================================
// Kernel signature generation
// ===================================================================

std::string MetalCodegen::generateSignature(const PhaseInfo& phase) const {
    std::ostringstream sig;
    sig << "kernel void " << phase.name << "(\n";

    bool needsThreadPos = !phase.isSingleThread;

    for (size_t i = 0; i < phase.bindings.size(); i++) {
        const auto& b = phase.bindings[i];
        sig << "    " << b.metalDecl << " " << b.name
            << " [[buffer(" << b.bufferIndex << ")]]";
        if (i + 1 < phase.bindings.size() || needsThreadPos)
            sig << ",";
        sig << "\n";
    }

    if (!phase.isSingleThread) {
        sig << "    uint tid [[thread_position_in_grid]],\n";
        sig << "    uint tpg [[threads_per_grid]],\n";
        sig << "    uint lid [[thread_position_in_threadgroup]],\n";
        sig << "    uint tg_size [[threads_per_threadgroup]],\n";
        sig << "    uint gid [[threadgroup_position_in_grid]],\n";
        sig << "    uint simd_lane [[thread_index_in_simdgroup]],\n";
        sig << "    uint simd_id [[simdgroup_index_in_threadgroup]]\n";
    } else {
        // Single-thread kernel still needs thread_position_in_grid
        sig << "    uint tid [[thread_position_in_grid]]\n";
    }

    sig << ")";
    return sig.str();
}

// ===================================================================
// Final output assembly
// ===================================================================

std::string MetalCodegen::print() const {
    std::ostringstream out;

    // 1. Common header (SIMD reductions, bitmap, atomics)
    out << commonHeader();

    // 2. Helper / device functions
    if (!helperCode_.empty()) {
        out << "// --- Helper functions ---\n";
        out << helperCode_ << "\n";
    }

    // 3. Each phase → one kernel function
    for (size_t pi = 0; pi < phases_.size(); pi++) {
        PhaseInfo phase = phases_[pi]; // copy so we can assign indices
        const_cast<MetalCodegen*>(this)->assignBufferIndices(
            const_cast<PhaseInfo&>(phases_[pi]));
        // Re-read after assignment
        phase = phases_[pi];

        out << "\n// === Phase " << pi << ": " << phase.name << " ===\n";
        out << generateSignature(phase) << " {\n";
        out << phase.code;
        out << "}\n";
    }

    return out.str();
}

// ===================================================================
// Phase info accessors
// ===================================================================

const std::vector<MetalCodegen::PhaseInfo>& MetalCodegen::getPhases() const {
    return phases_;
}

std::vector<MetalParamBinding> MetalCodegen::getAllBindings() const {
    std::vector<MetalParamBinding> all;
    std::unordered_set<std::string> seen;
    for (const auto& phase : phases_) {
        for (const auto& b : phase.bindings) {
            if (seen.insert(b.name).second) {
                all.push_back(b);
            }
        }
    }
    return all;
}

} // namespace codegen
