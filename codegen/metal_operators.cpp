#include "metal_operators.h"
#include <sstream>

namespace codegen {

// ===================================================================
// MetalGridStrideScan
// ===================================================================

MetalGridStrideScan::MetalGridStrideScan(const std::string& table,
                                         const std::string& rowVar,
                                         const std::string& idxVar)
    : tableName_(table), rowVar_(rowVar), idxVar_(idxVar) {}

void MetalGridStrideScan::addColumn(const std::string& paramName, const std::string& metalType) {
    columns_.push_back({paramName, metalType});
}

void MetalGridStrideScan::produce(MetalCodegen& cg, ConsumerFn consume) {
    cg.setPhaseScannedTable(tableName_);

    if (!columns_.empty()) {
        // Columnar mode: register each column as a separate buffer
        for (const auto& col : columns_) {
            cg.addColumnParam(col.paramName, col.metalType, tableName_);
        }
        cg.addTableSizeParam(tableName_);
    } else {
        // AoS mode: single struct pointer + size
        cg.addTableParam(tableName_, "/* struct set by planner */");
    }

    // Emit grid-stride loop
    cg.addBlock("for (uint " + idxVar_ + " = tid; " + idxVar_ + " < n_" + tableName_ +
                "; " + idxVar_ + " += tpg)", [&]() {
        consume();
    });
}

std::string MetalGridStrideScan::describe() const {
    return "GridStrideScan(" + tableName_ + ")";
}

// ===================================================================
// MetalSimpleScan
// ===================================================================

MetalSimpleScan::MetalSimpleScan(const std::string& table,
                                 const std::string& rowVar)
    : tableName_(table), rowVar_(rowVar) {}

void MetalSimpleScan::produce(MetalCodegen& cg, ConsumerFn consume) {
    cg.setPhaseScannedTable(tableName_);
    cg.addTableParam(tableName_, "/* struct set by planner */");
    cg.addIf("tid < n_" + tableName_, [&]() {
        consume();
    });
}

std::string MetalSimpleScan::describe() const {
    return "SimpleScan(" + tableName_ + ")";
}

// ===================================================================
// MetalSelection
// ===================================================================

MetalSelection::MetalSelection(std::unique_ptr<MetalOperator> child,
                               const std::string& predicate)
    : MetalUnaryOperator(std::move(child)), predicate_(predicate) {}

void MetalSelection::produce(MetalCodegen& cg, ConsumerFn consume) {
    child_->produce(cg, [&]() {
        cg.addIf(predicate_, [&]() {
            consume();
        });
    });
}

std::string MetalSelection::describe() const {
    return "Selection(" + predicate_ + ")";
}

// ===================================================================
// MetalComputeExpr
// ===================================================================

MetalComputeExpr::MetalComputeExpr(std::unique_ptr<MetalOperator> child,
                                   const std::string& varName,
                                   const std::string& varType,
                                   const std::string& expression)
    : MetalUnaryOperator(std::move(child)),
      varName_(varName), varType_(varType), expression_(expression) {}

void MetalComputeExpr::produce(MetalCodegen& cg, ConsumerFn consume) {
    child_->produce(cg, [&]() {
        cg.addLine(varType_ + " " + varName_ + " = " + expression_ + ";");
        consume();
    });
}

std::string MetalComputeExpr::describe() const {
    return "ComputeExpr(" + varName_ + " = " + expression_ + ")";
}

// ===================================================================
// MetalBitmapBuild
// ===================================================================

MetalBitmapBuild::MetalBitmapBuild(std::unique_ptr<MetalOperator> child,
                                   const std::string& bitmapName,
                                   const std::string& keyExpr,
                                   const std::string& sizeExpr)
    : MetalUnaryOperator(std::move(child)),
      bitmapName_(bitmapName), keyExpr_(keyExpr), sizeExpr_(sizeExpr) {}

void MetalBitmapBuild::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Register bitmap buffer as atomic for writes
    cg.addBitmapWriteParam(bitmapName_, sizeExpr_);

    child_->produce(cg, [&]() {
        cg.addLine("bitmap_set(" + bitmapName_ + ", " + keyExpr_ + ");");
        consume();  // allow chaining after bitmap set
    });
}

std::string MetalBitmapBuild::describe() const {
    return "BitmapBuild(" + bitmapName_ + ", key=" + keyExpr_ + ")";
}

// ===================================================================
// MetalBitmapProbe
// ===================================================================

MetalBitmapProbe::MetalBitmapProbe(std::unique_ptr<MetalOperator> child,
                                   const std::string& bitmapName,
                                   const std::string& keyExpr)
    : MetalUnaryOperator(std::move(child)),
      bitmapName_(bitmapName), keyExpr_(keyExpr) {}

void MetalBitmapProbe::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Register bitmap as read-only in probe phase
    cg.addBitmapReadParam(bitmapName_, "");  // size comes from build phase

    child_->produce(cg, [&]() {
        cg.addIf("bitmap_test(" + bitmapName_ + ", " + keyExpr_ + ")", [&]() {
            consume();
        });
    });
}

std::string MetalBitmapProbe::describe() const {
    return "BitmapProbe(" + bitmapName_ + ", key=" + keyExpr_ + ")";
}

// ===================================================================
// MetalAntiBitmapProbe
// ===================================================================

MetalAntiBitmapProbe::MetalAntiBitmapProbe(std::unique_ptr<MetalOperator> child,
                                           const std::string& bitmapName,
                                           const std::string& keyExpr)
    : MetalUnaryOperator(std::move(child)),
      bitmapName_(bitmapName), keyExpr_(keyExpr) {}

void MetalAntiBitmapProbe::produce(MetalCodegen& cg, ConsumerFn consume) {
    cg.addBitmapReadParam(bitmapName_, "");

    child_->produce(cg, [&]() {
        cg.addIf("!bitmap_test(" + bitmapName_ + ", " + keyExpr_ + ")", [&]() {
            consume();
        });
    });
}

std::string MetalAntiBitmapProbe::describe() const {
    return "AntiBitmapProbe(" + bitmapName_ + ", key=" + keyExpr_ + ")";
}

// ===================================================================
// MetalMultiBitmapProbe
// ===================================================================

MetalMultiBitmapProbe::MetalMultiBitmapProbe(std::unique_ptr<MetalOperator> child)
    : MetalUnaryOperator(std::move(child)) {}

void MetalMultiBitmapProbe::addBitmap(const std::string& bitmapName,
                                       const std::string& keyExpr) {
    bitmaps_.emplace_back(bitmapName, keyExpr);
}

void MetalMultiBitmapProbe::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Register all bitmaps as read
    for (const auto& [name, _] : bitmaps_)
        cg.addBitmapReadParam(name, "");

    child_->produce(cg, [&]() {
        // Build compound condition: bitmap_test(a, k1) && bitmap_test(b, k2) && ...
        std::string cond;
        for (size_t i = 0; i < bitmaps_.size(); i++) {
            if (i > 0) cond += " && ";
            cond += "bitmap_test(" + bitmaps_[i].first + ", " + bitmaps_[i].second + ")";
        }
        cg.addIf(cond, [&]() {
            consume();
        });
    });
}

std::string MetalMultiBitmapProbe::describe() const {
    return "MultiBitmapProbe(" + std::to_string(bitmaps_.size()) + " bitmaps)";
}

// ===================================================================
// MetalArrayStore
// ===================================================================

MetalArrayStore::MetalArrayStore(std::unique_ptr<MetalOperator> child,
                                 const std::string& arrayName,
                                 const std::string& keyExpr,
                                 const std::string& valueExpr,
                                 const std::string& valueType,
                                 const std::string& sizeExpr)
    : MetalUnaryOperator(std::move(child)),
      arrayName_(arrayName), keyExpr_(keyExpr), valueExpr_(valueExpr),
      valueType_(valueType), sizeExpr_(sizeExpr) {}

void MetalArrayStore::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Register array as device buffer, initialized to sentinel (-1 = 0xFF fill for int)
    cg.addBufferParam(arrayName_, valueType_, sizeExpr_, true, 0xFF);

    child_->produce(cg, [&]() {
        cg.addLine(arrayName_ + "[" + keyExpr_ + "] = " + valueExpr_ + ";");
        consume();
    });
}

std::string MetalArrayStore::describe() const {
    return "ArrayStore(" + arrayName_ + "[" + keyExpr_ + "] = " + valueExpr_ + ")";
}

// ===================================================================
// MetalArrayLookup
// ===================================================================

MetalArrayLookup::MetalArrayLookup(std::unique_ptr<MetalOperator> child,
                                   const std::string& arrayName,
                                   const std::string& keyExpr,
                                   const std::string& resultVar,
                                   const std::string& resultType,
                                   int sentinel)
    : MetalUnaryOperator(std::move(child)),
      arrayName_(arrayName), keyExpr_(keyExpr), resultVar_(resultVar),
      resultType_(resultType), sentinel_(sentinel) {}

void MetalArrayLookup::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Read-only access to array
    cg.addBufferParam(arrayName_, resultType_, "", false);

    child_->produce(cg, [&]() {
        cg.addLine(resultType_ + " " + resultVar_ + " = " + arrayName_ +
                   "[" + keyExpr_ + "];");
        // Guard: skip sentinel values (e.g., -1 means not found)
        cg.addIf(resultVar_ + " != " + std::to_string(sentinel_), [&]() {
            consume();
        });
    });
}

std::string MetalArrayLookup::describe() const {
    return "ArrayLookup(" + resultVar_ + " = " + arrayName_ + "[" + keyExpr_ + "])";
}

// ===================================================================
// MetalTGReduce
// ===================================================================

MetalTGReduce::MetalTGReduce(std::unique_ptr<MetalOperator> child,
                             const std::string& outputPrefix)
    : MetalUnaryOperator(std::move(child)), outputPrefix_(outputPrefix) {}

void MetalTGReduce::addAccumulator(const std::string& name,
                                    const std::string& valueExpr,
                                    const std::string& type,
                                    const std::string& loBuffer,
                                    const std::string& hiBuffer) {
    Accumulator acc;
    acc.name = name;
    acc.valueExpr = valueExpr;
    acc.type = type;
    acc.loBuffer = loBuffer.empty() ? (outputPrefix_ + "_" + name + "_lo") : loBuffer;
    acc.hiBuffer = hiBuffer.empty() ? (type == "long" ? (outputPrefix_ + "_" + name + "_hi") : "") : hiBuffer;
    // For float type, we also need hi buffer for the long pair approach
    if (acc.type == "float" && acc.hiBuffer.empty()) {
        acc.hiBuffer = outputPrefix_ + "_" + name + "_hi";
    }
    acc.binIndex = static_cast<int>(accumulators_.size());
    accumulators_.push_back(acc);
}

void MetalTGReduce::setResultAlias(const std::string& displayName, int scaleDown) {
    resultInfos_.push_back({displayName, scaleDown});
}

void MetalTGReduce::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Register output buffers — always use lo/hi atomic_uint pair
    // This avoids the broken float atomic approach and uses atomic_add_long_pair()
    for (const auto& acc : accumulators_) {
        cg.addAtomicBufferParam(acc.loBuffer, "atomic_uint", "1");
        cg.addAtomicBufferParam(acc.hiBuffer, "atomic_uint", "1");
    }

    // Declare local accumulators — always accumulate as long for correctness
    for (const auto& acc : accumulators_) {
        cg.addLine("long local_" + acc.name + " = 0;");
    }

    // Child produces rows; inside the loop we accumulate
    child_->produce(cg, [&]() {
        for (const auto& acc : accumulators_) {
            cg.addLine("local_" + acc.name + " += (long)(" + acc.valueExpr + ");");
        }
        consume();
    });

    // After the loop: SIMD + threadgroup reduction → atomic_add_long_pair
    cg.addComment("--- Threadgroup reduction ---");
    for (const auto& acc : accumulators_) {
        std::string localVar = "local_" + acc.name;
        std::string tgVar = "tg_" + acc.name;

        cg.addLine("threadgroup long tg_shared_" + acc.name + "[32];");
        cg.addLine("long " + tgVar + " = tg_reduce_long(" + localVar +
                   ", lid, tg_size, tg_shared_" + acc.name + ");");
        cg.addIf("lid == 0", [&]() {
            cg.addLine("atomic_add_long_pair(" + acc.loBuffer + ", " +
                       acc.hiBuffer + ", " + tgVar + ");");
        });
    }

    // Register result schema
    if (!resultInfos_.empty()) {
        for (size_t i = 0; i < accumulators_.size() && i < resultInfos_.size(); i++) {
            const auto& acc = accumulators_[i];
            const auto& info = resultInfos_[i];
            cg.registerScalarAggOutput(acc.loBuffer, acc.hiBuffer, acc.type);
            cg.registerScalarAggColumn(info.displayName, (int)i, info.scaleDown);
        }
    }
}

std::string MetalTGReduce::describe() const {
    return "TGReduce(" + std::to_string(accumulators_.size()) + " accumulators)";
}

// ===================================================================
// MetalKeyedAgg
// ===================================================================

MetalKeyedAgg::MetalKeyedAgg(std::unique_ptr<MetalOperator> child,
                             const std::string& outputArrayName,
                             const std::string& bucketExpr,
                             int numBuckets,
                             int valuesPerBucket,
                             const std::string& sizeExpr)
    : MetalUnaryOperator(std::move(child)),
      outputArrayName_(outputArrayName), bucketExpr_(bucketExpr),
      numBuckets_(numBuckets), valuesPerBucket_(valuesPerBucket),
      sizeExpr_(sizeExpr) {}

void MetalKeyedAgg::addAggregate(const std::string& name, int offset,
                                  const std::string& valueExpr,
                                  const std::string& atomicOp,
                                  bool isLongPair,
                                  int scaleDown) {
    aggregates_.push_back({name, offset, valueExpr, atomicOp, isLongPair, scaleDown});
}

void MetalKeyedAgg::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Register output buffer
    std::string sz = sizeExpr_.empty()
        ? std::to_string(numBuckets_ * valuesPerBucket_)
        : sizeExpr_;
    cg.addAtomicBufferParam(outputArrayName_, "atomic_uint", sz);

    child_->produce(cg, [&]() {
        cg.addLine("int _bucket = " + bucketExpr_ + ";");
        for (const auto& agg : aggregates_) {
            std::string base = "_bucket * " + std::to_string(valuesPerBucket_);
            if (agg.isLongPair && agg.atomicOp == "add") {
                // 64-bit accumulation via lo/hi atomic_uint pair
                std::string loIdx = base + " + " + std::to_string(agg.offset);
                std::string hiIdx = base + " + " + std::to_string(agg.offset + 1);
                cg.addLine("atomic_add_long_pair(&" + outputArrayName_ + "[" + loIdx +
                           "], &" + outputArrayName_ + "[" + hiIdx +
                           "], (long)(" + agg.valueExpr + "));");
            } else {
                std::string idx = base + " + " + std::to_string(agg.offset);
                if (agg.atomicOp == "add") {
                    cg.addLine("atomic_fetch_add_explicit(&" + outputArrayName_ + "[" + idx +
                               "], (uint)(" + agg.valueExpr + "), memory_order_relaxed);");
                } else if (agg.atomicOp == "min") {
                    cg.addLine("atomic_fetch_min_explicit(&" + outputArrayName_ + "[" + idx +
                               "], (uint)(" + agg.valueExpr + "), memory_order_relaxed);");
                } else if (agg.atomicOp == "max") {
                    cg.addLine("atomic_fetch_max_explicit(&" + outputArrayName_ + "[" + idx +
                               "], (uint)(" + agg.valueExpr + "), memory_order_relaxed);");
                }
            }
        }
        consume();
    });

    // Register result schema with slot layout
    std::vector<MetalResultSchema::KeyedAggSlot> slots;
    for (const auto& agg : aggregates_) {
        slots.push_back({agg.name, agg.offset, agg.isLongPair, agg.scaleDown});
    }
    cg.registerKeyedAggOutput(outputArrayName_, numBuckets_, valuesPerBucket_, slots);
}

std::string MetalKeyedAgg::describe() const {
    return "KeyedAgg(" + outputArrayName_ + ", " + std::to_string(numBuckets_) +
           " buckets, " + std::to_string(aggregates_.size()) + " aggs)";
}

// ===================================================================
// MetalAtomicAgg
// ===================================================================

MetalAtomicAgg::MetalAtomicAgg(std::unique_ptr<MetalOperator> child,
                               const std::string& arrayName,
                               const std::string& bucketExpr,
                               const std::string& valueExpr,
                               const std::string& sizeExpr,
                               const std::string& atomicType,
                               const std::string& castType)
    : MetalUnaryOperator(std::move(child)),
      arrayName_(arrayName), bucketExpr_(bucketExpr),
      valueExpr_(valueExpr), sizeExpr_(sizeExpr),
      atomicType_(atomicType), castType_(castType) {}

void MetalAtomicAgg::produce(MetalCodegen& cg, ConsumerFn consume) {
    cg.addAtomicBufferParam(arrayName_, atomicType_, sizeExpr_);

    child_->produce(cg, [&]() {
        cg.addLine("atomic_fetch_add_explicit(&" + arrayName_ + "[" + bucketExpr_ +
                   "], (" + castType_ + ")(" + valueExpr_ + "), memory_order_relaxed);");
        consume();
    });
}

std::string MetalAtomicAgg::describe() const {
    return "AtomicAgg(" + arrayName_ + "[" + bucketExpr_ + "])";
}

// ===================================================================
// MetalAtomicCount
// ===================================================================

MetalAtomicCount::MetalAtomicCount(std::unique_ptr<MetalOperator> child,
                                   const std::string& arrayName,
                                   const std::string& bucketExpr,
                                   const std::string& sizeExpr)
    : MetalUnaryOperator(std::move(child)),
      arrayName_(arrayName), bucketExpr_(bucketExpr), sizeExpr_(sizeExpr) {}

void MetalAtomicCount::produce(MetalCodegen& cg, ConsumerFn consume) {
    cg.addAtomicBufferParam(arrayName_, "atomic_uint", sizeExpr_);

    child_->produce(cg, [&]() {
        cg.addLine("atomic_fetch_add_explicit(&" + arrayName_ + "[" + bucketExpr_ +
                   "], 1u, memory_order_relaxed);");
        consume();
    });
}

std::string MetalAtomicCount::describe() const {
    return "AtomicCount(" + arrayName_ + "[" + bucketExpr_ + "])";
}

// ===================================================================
// MetalMaterialize
// ===================================================================

MetalMaterialize::MetalMaterialize(std::unique_ptr<MetalOperator> child,
                                   const std::string& counterName,
                                   const std::string& counterSizeExpr)
    : MetalUnaryOperator(std::move(child)),
      counterName_(counterName), counterSizeExpr_(counterSizeExpr) {}

void MetalMaterialize::addColumn(const std::string& arrayName, const std::string& type,
                                  const std::string& valueExpr,
                                  const std::string& displayName,
                                  const std::string& sizeExpr) {
    columns_.push_back({arrayName, type, valueExpr,
                        displayName.empty() ? arrayName : displayName, sizeExpr});
}

void MetalMaterialize::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Register counter
    cg.addAtomicBufferParam(counterName_, "atomic_uint", counterSizeExpr_);

    // Register output column buffers
    for (const auto& col : columns_) {
        cg.addBufferParam(col.arrayName, col.type, col.sizeExpr, false);
    }

    // Register result schema
    cg.registerMaterializeOutput(counterName_);
    for (const auto& col : columns_) {
        cg.registerOutputColumn(col.displayName, col.arrayName, col.type);
    }

    child_->produce(cg, [&]() {
        // Atomic increment counter to get output position
        cg.addLine("uint _pos = atomic_fetch_add_explicit(&" + counterName_ +
                   "[0], 1u, memory_order_relaxed);");
        // Scatter values to output arrays
        for (const auto& col : columns_) {
            cg.addLine(col.arrayName + "[_pos] = " + col.valueExpr + ";");
        }
        consume();
    });
}

std::string MetalMaterialize::describe() const {
    return "Materialize(" + std::to_string(columns_.size()) + " columns)";
}

// ===================================================================
// MetalCompact
// ===================================================================

MetalCompact::MetalCompact(const std::string& inputArray,
                           const std::string& outputArray,
                           const std::string& counterName,
                           int maxSlots,
                           const std::string& elementType)
    : inputArray_(inputArray), outputArray_(outputArray),
      counterName_(counterName), maxSlots_(maxSlots), elementType_(elementType) {}

void MetalCompact::produce(MetalCodegen& cg, ConsumerFn consume) {
    cg.setPhaseScannedTable("");  // no table scan
    cg.setPhaseSingleThread(false);

    // Input and output buffers
    cg.addBufferParam(inputArray_, elementType_, std::to_string(maxSlots_), false);
    cg.addBufferParam(outputArray_, elementType_, std::to_string(maxSlots_), false);
    cg.addAtomicBufferParam(counterName_, "atomic_uint", "1");

    cg.addBlock("for (uint i = tid; i < " + std::to_string(maxSlots_) + "; i += tpg)", [&]() {
        cg.addIf(inputArray_ + "[i] != 0", [&]() {
            cg.addLine("uint _pos = atomic_fetch_add_explicit(&" + counterName_ +
                       "[0], 1u, memory_order_relaxed);");
            cg.addLine(outputArray_ + "[_pos] = " + inputArray_ + "[i];");
        });
    });

    consume();
}

std::string MetalCompact::describe() const {
    return "Compact(" + inputArray_ + " → " + outputArray_ + ")";
}

// ===================================================================
// MetalStringMatch
// ===================================================================

MetalStringMatch::MetalStringMatch(std::unique_ptr<MetalOperator> child,
                                   const std::string& stringExpr,
                                   const std::string& pattern,
                                   bool negated,
                                   int fixedWidth)
    : MetalUnaryOperator(std::move(child)),
      stringExpr_(stringExpr), pattern_(pattern),
      negated_(negated), fixedWidth_(fixedWidth) {}

void MetalStringMatch::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Generate inline LIKE matching code
    // For simple patterns like '%word%', 'word%', '%word'
    // we generate direct comparison code

    child_->produce(cg, [&]() {
        // Emit a LIKE match helper inline
        // We handle common patterns: '%substr%', 'prefix%', '%suffix'
        bool startsWild = !pattern_.empty() && pattern_.front() == '%';
        bool endsWild = !pattern_.empty() && pattern_.back() == '%';

        std::string core = pattern_;
        if (startsWild) core = core.substr(1);
        if (endsWild && !core.empty()) core.pop_back();

        if (startsWild && endsWild) {
            // Contains match: strstr-like search
            cg.addLine("bool _match = false;");
            cg.addBlock("for (int _p = 0; _p < " + std::to_string(fixedWidth_) +
                        " - " + std::to_string((int)core.size()) + " + 1; _p++)", [&]() {
                cg.addLine("bool _eq = true;");
                for (size_t i = 0; i < core.size(); i++) {
                    cg.addLine("_eq = _eq && (" + stringExpr_ + "[_p + " +
                               std::to_string(i) + "] == '" + core[i] + "');");
                }
                cg.addIf("_eq", [&]() {
                    cg.addLine("_match = true;");
                });
            });
        } else if (endsWild) {
            // Prefix match
            cg.addLine("bool _match = true;");
            for (size_t i = 0; i < core.size(); i++) {
                cg.addLine("_match = _match && (" + stringExpr_ + "[" +
                           std::to_string(i) + "] == '" + core[i] + "');");
            }
        } else if (startsWild) {
            // Suffix match
            cg.addLine("bool _match = true;");
            // Find end of string or use fixedWidth
            for (size_t i = 0; i < core.size(); i++) {
                int pos = fixedWidth_ - (int)core.size() + (int)i;
                cg.addLine("_match = _match && (" + stringExpr_ + "[" +
                           std::to_string(pos) + "] == '" + core[i] + "');");
            }
        } else {
            // Exact match
            cg.addLine("bool _match = true;");
            for (size_t i = 0; i < core.size(); i++) {
                cg.addLine("_match = _match && (" + stringExpr_ + "[" +
                           std::to_string(i) + "] == '" + core[i] + "');");
            }
        }

        std::string cond = negated_ ? "!_match" : "_match";
        cg.addIf(cond, [&]() {
            consume();
        });
    });
}

std::string MetalStringMatch::describe() const {
    return "StringMatch(" + pattern_ + (negated_ ? ", negated" : "") + ")";
}

// ===================================================================
// MetalHistogram
// ===================================================================

MetalHistogram::MetalHistogram(const std::string& inputCountArray,
                               const std::string& outputHistArray,
                               const std::string& inputSize,
                               int maxBuckets,
                               const std::string& histSizeExpr)
    : inputCountArray_(inputCountArray), outputHistArray_(outputHistArray),
      inputSize_(inputSize), maxBuckets_(maxBuckets),
      histSizeExpr_(histSizeExpr) {}

void MetalHistogram::produce(MetalCodegen& cg, ConsumerFn consume) {
    cg.setPhaseScannedTable("");
    cg.setPhaseSingleThread(false);

    cg.addBufferParam(inputCountArray_, "uint", inputSize_, false);
    cg.addAtomicBufferParam(outputHistArray_, "atomic_uint",
                            histSizeExpr_.empty() ? std::to_string(maxBuckets_ + 1) : histSizeExpr_);
    cg.addScalarParam("n_" + inputCountArray_, "uint");

    cg.addBlock("for (uint i = tid; i < n_" + inputCountArray_ + "; i += tpg)", [&]() {
        cg.addLine("uint cnt = " + inputCountArray_ + "[i];");
        cg.addIf("cnt > 0 && cnt <= " + std::to_string(maxBuckets_), [&]() {
            cg.addLine("atomic_fetch_add_explicit(&" + outputHistArray_ +
                       "[cnt], 1u, memory_order_relaxed);");
        });
    });

    consume();
}

std::string MetalHistogram::describe() const {
    return "Histogram(" + inputCountArray_ + " → " + outputHistArray_ + ")";
}

// ===================================================================
// MetalSemiJoin
// ===================================================================

MetalSemiJoin::MetalSemiJoin(std::unique_ptr<MetalOperator> buildSide,
                             std::unique_ptr<MetalOperator> probeSide,
                             const std::string& buildKeyExpr,
                             const std::string& probeKeyExpr,
                             const std::string& bitmapSizeExpr,
                             const std::string& bitmapName)
    : MetalBinaryOperator(std::move(buildSide), std::move(probeSide)),
      buildKeyExpr_(buildKeyExpr), probeKeyExpr_(probeKeyExpr),
      bitmapSizeExpr_(bitmapSizeExpr),
      bitmapName_(bitmapName.empty() ? "d_semi_bitmap" : bitmapName) {}

void MetalSemiJoin::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Phase 1: Build bitmap
    auto buildOp = std::make_unique<MetalBitmapBuild>(
        std::move(build_), bitmapName_, buildKeyExpr_, bitmapSizeExpr_);
    buildOp->produce(cg, [](){});  // no consumer after bitmap build

    cg.endPhase();

    // Phase 2: Probe bitmap
    // The caller must call beginPhase() before and after
    // Actually, the probe side wraps with bitmap probe
    auto probeOp = std::make_unique<MetalBitmapProbe>(
        std::move(probe_), bitmapName_, probeKeyExpr_);
    probeOp->produce(cg, consume);
}

std::string MetalSemiJoin::describe() const {
    return "SemiJoin(bitmap=" + bitmapName_ + ")";
}

// ===================================================================
// MetalIndexJoin
// ===================================================================

MetalIndexJoin::MetalIndexJoin(std::unique_ptr<MetalOperator> buildSide,
                               std::unique_ptr<MetalOperator> probeSide,
                               const std::string& buildKeyExpr,
                               const std::string& buildValueExpr,
                               const std::string& probeKeyExpr,
                               const std::string& resultVar,
                               const std::string& arraySizeExpr,
                               const std::string& valueType,
                               const std::string& arrayName)
    : MetalBinaryOperator(std::move(buildSide), std::move(probeSide)),
      buildKeyExpr_(buildKeyExpr), buildValueExpr_(buildValueExpr),
      probeKeyExpr_(probeKeyExpr), resultVar_(resultVar),
      arraySizeExpr_(arraySizeExpr), valueType_(valueType),
      arrayName_(arrayName.empty() ? "d_idx_map" : arrayName) {}

void MetalIndexJoin::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Phase 1: Build direct-address array
    auto buildOp = std::make_unique<MetalArrayStore>(
        std::move(build_), arrayName_, buildKeyExpr_, buildValueExpr_,
        valueType_, arraySizeExpr_);
    buildOp->produce(cg, [](){});

    cg.endPhase();

    // Phase 2: Probe via lookup
    auto probeOp = std::make_unique<MetalArrayLookup>(
        std::move(probe_), arrayName_, probeKeyExpr_, resultVar_, valueType_);
    probeOp->produce(cg, consume);
}

std::string MetalIndexJoin::describe() const {
    return "IndexJoin(array=" + arrayName_ + ")";
}

} // namespace codegen
