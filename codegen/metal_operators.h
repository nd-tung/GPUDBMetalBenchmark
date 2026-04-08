#pragma once
// ===================================================================
// Metal Operators — Composable producer-consumer code generation
// ===================================================================
//
// Each operator has a produce() method that emits Metal shader code
// into a MetalCodegen instance. Operators form trees where each
// calls its child's produce() and wraps the consumer callback.
//
// Modeled on the CUDACodeGeneral operator framework, but emitting
// Metal Shading Language with Apple GPU-specific optimizations
// (SIMD group reductions, Metal atomics, [[buffer(N)]] attributes).
// ===================================================================

#include "metal_codegen_base.h"
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace codegen {

// ===================================================================
// BASE CLASSES
// ===================================================================

class MetalOperator {
public:
    virtual ~MetalOperator() = default;
    virtual void produce(MetalCodegen& cg, ConsumerFn consume) = 0;
    virtual std::string describe() const = 0;
};

class MetalUnaryOperator : public MetalOperator {
protected:
    std::unique_ptr<MetalOperator> child_;
public:
    explicit MetalUnaryOperator(std::unique_ptr<MetalOperator> child)
        : child_(std::move(child)) {}
};

class MetalBinaryOperator : public MetalOperator {
protected:
    std::unique_ptr<MetalOperator> build_;
    std::unique_ptr<MetalOperator> probe_;
public:
    MetalBinaryOperator(std::unique_ptr<MetalOperator> build,
                        std::unique_ptr<MetalOperator> probe)
        : build_(std::move(build)), probe_(std::move(probe)) {}
};

// ===================================================================
// LEAF OPERATORS — Table Scans
// ===================================================================

// Grid-stride loop over a table. Supports columnar layout.
// Emits:
//   for (uint {idxVar} = tid; {idxVar} < n_{table}; {idxVar} += tpg) {
//       <consume()>
//   }
class MetalGridStrideScan : public MetalOperator {
public:
    // Column descriptor for columnar scan
    struct ColumnDesc {
        std::string paramName;   // buffer parameter name (e.g. "l_shipdate")
        std::string metalType;   // Metal type (e.g. "int", "float", "char")
    };

    MetalGridStrideScan(const std::string& table,
                        const std::string& rowVar = "row",
                        const std::string& idxVar = "i");
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

    // Add a column for columnar scan
    void addColumn(const std::string& paramName, const std::string& metalType);

    const std::string& tableName() const { return tableName_; }
    const std::string& rowVar() const { return rowVar_; }
    const std::string& idxVar() const { return idxVar_; }

private:
    std::string tableName_;
    std::string rowVar_;
    std::string idxVar_;
    std::vector<ColumnDesc> columns_;
};

// Simple scan without grid-stride (for single-threadgroup or index-based).
// Emits:
//   if (tid < n_{table}) { ... }
class MetalSimpleScan : public MetalOperator {
public:
    MetalSimpleScan(const std::string& table,
                    const std::string& rowVar = "row");
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string tableName_;
    std::string rowVar_;
};

// ===================================================================
// UNARY OPERATORS — Pipeline operators
// ===================================================================

// Selection (WHERE filter): if (predicate) { consume(); }
class MetalSelection : public MetalUnaryOperator {
public:
    MetalSelection(std::unique_ptr<MetalOperator> child,
                   const std::string& predicate);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string predicate_;
};

// Compute expression: type var = expr; consume();
class MetalComputeExpr : public MetalUnaryOperator {
public:
    MetalComputeExpr(std::unique_ptr<MetalOperator> child,
                     const std::string& varName,
                     const std::string& varType,
                     const std::string& expression);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string varName_;
    std::string varType_;
    std::string expression_;
};

// ===================================================================
// BITMAP OPERATORS
// ===================================================================

// Build a bitmap: atomic_fetch_or to set bit for key
class MetalBitmapBuild : public MetalUnaryOperator {
public:
    MetalBitmapBuild(std::unique_ptr<MetalOperator> child,
                     const std::string& bitmapName,
                     const std::string& keyExpr,
                     const std::string& sizeExpr);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string bitmapName_;
    std::string keyExpr_;
    std::string sizeExpr_;
};

// Probe a bitmap: if (bitmap_test(bitmap, key)) { consume(); }
class MetalBitmapProbe : public MetalUnaryOperator {
public:
    MetalBitmapProbe(std::unique_ptr<MetalOperator> child,
                     const std::string& bitmapName,
                     const std::string& keyExpr);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string bitmapName_;
    std::string keyExpr_;
};

// Anti-probe: if (!bitmap_test(bitmap, key)) { consume(); }
class MetalAntiBitmapProbe : public MetalUnaryOperator {
public:
    MetalAntiBitmapProbe(std::unique_ptr<MetalOperator> child,
                         const std::string& bitmapName,
                         const std::string& keyExpr);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string bitmapName_;
    std::string keyExpr_;
};

// Multi-bitmap probe: AND of multiple bitmap tests
class MetalMultiBitmapProbe : public MetalUnaryOperator {
public:
    MetalMultiBitmapProbe(std::unique_ptr<MetalOperator> child);
    void addBitmap(const std::string& bitmapName, const std::string& keyExpr);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::vector<std::pair<std::string, std::string>> bitmaps_;
};

// ===================================================================
// DIRECT-ADDRESS MAP OPERATORS
// ===================================================================

// Store: map[key] = value
class MetalArrayStore : public MetalUnaryOperator {
public:
    MetalArrayStore(std::unique_ptr<MetalOperator> child,
                    const std::string& arrayName,
                    const std::string& keyExpr,
                    const std::string& valueExpr,
                    const std::string& valueType = "int",
                    const std::string& sizeExpr = "");
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string arrayName_;
    std::string keyExpr_;
    std::string valueExpr_;
    std::string valueType_;
    std::string sizeExpr_;
};

// Lookup: type var = map[key]; (with optional guard for sentinel value)
class MetalArrayLookup : public MetalUnaryOperator {
public:
    MetalArrayLookup(std::unique_ptr<MetalOperator> child,
                     const std::string& arrayName,
                     const std::string& keyExpr,
                     const std::string& resultVar,
                     const std::string& resultType = "int",
                     int sentinel = -1);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string arrayName_;
    std::string keyExpr_;
    std::string resultVar_;
    std::string resultType_;
    int sentinel_;
};

// ===================================================================
// AGGREGATION OPERATORS
// ===================================================================

// Threadgroup reduce using SIMD group reductions.
// Emits per-thread local accumulation + tg_reduce_float/long + atomic to global.
class MetalTGReduce : public MetalUnaryOperator {
public:
    struct Accumulator {
        std::string name;        // local variable name
        std::string loBuffer;    // output buffer name (lo part or direct)
        std::string hiBuffer;    // output buffer name (hi part, empty if float)
        std::string valueExpr;   // expression to accumulate
        std::string type;        // "float" or "long"
        int binIndex = 0;        // for multi-bin, which index in the output
    };

    MetalTGReduce(std::unique_ptr<MetalOperator> child,
                  const std::string& outputPrefix);
    void addAccumulator(const std::string& name, const std::string& valueExpr,
                        const std::string& type = "float",
                        const std::string& loBuffer = "",
                        const std::string& hiBuffer = "");

    // Register result schema for this reduce's output
    // scaleDown: divisor for fixed-point (e.g. 100 means stored as val*100)
    void setResultAlias(const std::string& displayName, int scaleDown = 0);

    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string outputPrefix_;
    std::vector<Accumulator> accumulators_;

    // Result schema info
    struct ResultInfo {
        std::string displayName;
        int scaleDown = 0;
    };
    std::vector<ResultInfo> resultInfos_;
};

// Keyed aggregation using atomics.
// For "add" with isLongPair=true, uses atomic_add_long_pair for 64-bit correctness.
// offset/offset+1 form the lo/hi pair in the output buffer.
class MetalKeyedAgg : public MetalUnaryOperator {
public:
    struct Aggregate {
        std::string name;
        int offset;
        std::string valueExpr;
        std::string atomicOp;   // "add", "min", "max"
        bool isLongPair = false; // true → uses lo/hi atomic_uint pair at offset/offset+1
        int scaleDown = 0;      // result divisor (e.g. 100 for cents→dollars, 0=none)
    };

    MetalKeyedAgg(std::unique_ptr<MetalOperator> child,
                  const std::string& outputArrayName,
                  const std::string& bucketExpr,
                  int numBuckets,
                  int valuesPerBucket,
                  const std::string& sizeExpr = "");
    void addAggregate(const std::string& name, int offset,
                      const std::string& valueExpr,
                      const std::string& atomicOp = "add",
                      bool isLongPair = false,
                      int scaleDown = 0);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string outputArrayName_;
    std::string bucketExpr_;
    int numBuckets_;
    int valuesPerBucket_;
    std::string sizeExpr_;
    std::vector<Aggregate> aggregates_;
};

// Simple atomic add to array: atomic_fetch_add(&arr[bucket], value)
class MetalAtomicAgg : public MetalUnaryOperator {
public:
    MetalAtomicAgg(std::unique_ptr<MetalOperator> child,
                   const std::string& arrayName,
                   const std::string& bucketExpr,
                   const std::string& valueExpr,
                   const std::string& sizeExpr = "",
                   const std::string& atomicType = "atomic_uint",
                   const std::string& castType = "uint");
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string arrayName_;
    std::string bucketExpr_;
    std::string valueExpr_;
    std::string sizeExpr_;
    std::string atomicType_;
    std::string castType_;
};

// Atomic count: atomic_fetch_add(&arr[bucket], 1)
class MetalAtomicCount : public MetalUnaryOperator {
public:
    MetalAtomicCount(std::unique_ptr<MetalOperator> child,
                     const std::string& arrayName,
                     const std::string& bucketExpr,
                     const std::string& sizeExpr = "");
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string arrayName_;
    std::string bucketExpr_;
    std::string sizeExpr_;
};

// ===================================================================
// MATERIALIZATION
// ===================================================================

// Materialize rows to output arrays via atomic counter
class MetalMaterialize : public MetalUnaryOperator {
public:
    struct Column {
        std::string arrayName;
        std::string type;
        std::string valueExpr;
        std::string displayName;
        std::string sizeExpr;
    };

    MetalMaterialize(std::unique_ptr<MetalOperator> child,
                     const std::string& counterName,
                     const std::string& counterSizeExpr = "1");
    void addColumn(const std::string& arrayName, const std::string& type,
                   const std::string& valueExpr, const std::string& displayName = "",
                   const std::string& sizeExpr = "");
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string counterName_;
    std::string counterSizeExpr_;
    std::vector<Column> columns_;
};

// ===================================================================
// COMPACT — sparse → dense extraction
// ===================================================================

class MetalCompact : public MetalOperator {
public:
    MetalCompact(const std::string& inputArray,
                 const std::string& outputArray,
                 const std::string& counterName,
                 int maxSlots,
                 const std::string& elementType = "int");
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string inputArray_;
    std::string outputArray_;
    std::string counterName_;
    int maxSlots_;
    std::string elementType_;
};

// ===================================================================
// STRING MATCH — LIKE pattern
// ===================================================================

class MetalStringMatch : public MetalUnaryOperator {
public:
    MetalStringMatch(std::unique_ptr<MetalOperator> child,
                     const std::string& stringExpr,
                     const std::string& pattern,
                     bool negated = false,
                     int fixedWidth = 0);
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string stringExpr_;
    std::string pattern_;
    bool negated_;
    int fixedWidth_;
};

// ===================================================================
// HISTOGRAM — count distribution
// ===================================================================

class MetalHistogram : public MetalOperator {
public:
    MetalHistogram(const std::string& inputCountArray,
                   const std::string& outputHistArray,
                   const std::string& inputSize,
                   int maxBuckets,
                   const std::string& histSizeExpr = "");
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string inputCountArray_;
    std::string outputHistArray_;
    std::string inputSize_;
    int maxBuckets_;
    std::string histSizeExpr_;
};

// ===================================================================
// BINARY OPERATORS — Joins (multi-phase)
// ===================================================================

// Semi-join via bitmap: build phase sets bitmap, probe phase tests it.
// Produces TWO phases: build kernel + probe kernel.
class MetalSemiJoin : public MetalBinaryOperator {
public:
    MetalSemiJoin(std::unique_ptr<MetalOperator> buildSide,
                  std::unique_ptr<MetalOperator> probeSide,
                  const std::string& buildKeyExpr,
                  const std::string& probeKeyExpr,
                  const std::string& bitmapSizeExpr,
                  const std::string& bitmapName = "");
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string buildKeyExpr_;
    std::string probeKeyExpr_;
    std::string bitmapSizeExpr_;
    std::string bitmapName_;
};

// Index join via direct-address array: build stores key→value, probe looks up.
class MetalIndexJoin : public MetalBinaryOperator {
public:
    MetalIndexJoin(std::unique_ptr<MetalOperator> buildSide,
                   std::unique_ptr<MetalOperator> probeSide,
                   const std::string& buildKeyExpr,
                   const std::string& buildValueExpr,
                   const std::string& probeKeyExpr,
                   const std::string& resultVar,
                   const std::string& arraySizeExpr,
                   const std::string& valueType = "int",
                   const std::string& arrayName = "");
    void produce(MetalCodegen& cg, ConsumerFn consume) override;
    std::string describe() const override;

private:
    std::string buildKeyExpr_;
    std::string buildValueExpr_;
    std::string probeKeyExpr_;
    std::string resultVar_;
    std::string arraySizeExpr_;
    std::string valueType_;
    std::string arrayName_;
};

} // namespace codegen
