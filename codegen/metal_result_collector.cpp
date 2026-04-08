#include "metal_result_collector.h"
#include <iostream>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <cstdint>

namespace codegen {

// ===================================================================
// GenericResult::print
// ===================================================================

void GenericResult::print(int limit) const {
    if (columns.empty()) return;

    // Compute column widths
    std::vector<size_t> widths(columns.size());
    for (size_t c = 0; c < columns.size(); c++)
        widths[c] = columns[c].name.size();

    size_t rowCount = (limit > 0 && (size_t)limit < rows.size()) ? (size_t)limit : rows.size();
    for (size_t r = 0; r < rowCount; r++) {
        for (size_t c = 0; c < columns.size() && c < rows[r].size(); c++) {
            size_t w = 0;
            std::visit([&w](auto&& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, std::string>)
                    w = v.size();
                else
                    w = std::to_string(v).size();
            }, rows[r][c]);
            widths[c] = std::max(widths[c], w + 2);
        }
    }
    // Minimum width
    for (auto& w : widths) w = std::max(w, (size_t)10);

    // Print header separator
    auto printSep = [&]() {
        std::cout << "+";
        for (size_t c = 0; c < columns.size(); c++) {
            std::cout << std::string(widths[c] + 2, '-') << "+";
        }
        std::cout << "\n";
    };

    printSep();
    std::cout << "|";
    for (size_t c = 0; c < columns.size(); c++) {
        std::cout << " " << std::setw((int)widths[c]) << std::right << columns[c].name << " |";
    }
    std::cout << "\n";
    printSep();

    // Print rows
    for (size_t r = 0; r < rowCount; r++) {
        std::cout << "|";
        for (size_t c = 0; c < columns.size() && c < rows[r].size(); c++) {
            std::cout << " ";
            std::visit([&](auto&& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, int64_t>)
                    std::cout << std::setw((int)widths[c]) << std::right << v;
                else if constexpr (std::is_same_v<T, double>)
                    std::cout << std::setw((int)widths[c]) << std::right << std::fixed << std::setprecision(2) << v;
                else
                    std::cout << std::setw((int)widths[c]) << std::right << v;
            }, rows[r][c]);
            std::cout << " |";
        }
        std::cout << "\n";
    }
    printSep();

    if (limit > 0 && (size_t)limit < rows.size())
        std::cout << "... (" << rows.size() - limit << " more rows)\n";
}

// ===================================================================
// Long reconstruction
// ===================================================================

int64_t MetalResultCollector::reconstructLong(uint32_t lo, uint32_t hi) {
    uint64_t uval = ((uint64_t)hi << 32) | (uint64_t)lo;
    return static_cast<int64_t>(uval);
}

// ===================================================================
// collect — dispatch based on schema kind
// ===================================================================

GenericResult MetalResultCollector::collect(const MetalResultSchema& schema,
                                            const BufferMap& buffers) {
    switch (schema.kind) {
        case MetalResultSchema::SCALAR_AGG:
            return collectScalarAgg(schema, buffers);
        case MetalResultSchema::KEYED_AGG:
            return collectKeyedAgg(schema, buffers);
        case MetalResultSchema::MATERIALIZE:
            return collectMaterialize(schema, buffers);
        case MetalResultSchema::NONE:
        default:
            return {};
    }
}

// ===================================================================
// collectScalarAgg
// ===================================================================

GenericResult MetalResultCollector::collectScalarAgg(const MetalResultSchema& schema,
                                                     const BufferMap& buffers) {
    GenericResult result;
    GenericResult::Row row;

    for (const auto& entry : schema.scalarAggs) {
        result.columns.push_back({entry.displayName, entry.isLongPair ? "long" : "float"});

        double value = 0.0;
        if (entry.isLongPair) {
            auto loIt = buffers.find(entry.loBuffer);
            auto hiIt = buffers.find(entry.hiBuffer);
            if (loIt != buffers.end() && hiIt != buffers.end()) {
                uint32_t lo = *static_cast<uint32_t*>(loIt->second->contents());
                uint32_t hi = *static_cast<uint32_t*>(hiIt->second->contents());
                int64_t raw = reconstructLong(lo, hi);
                value = static_cast<double>(raw);
            }
        } else {
            auto it = buffers.find(entry.loBuffer);
            if (it != buffers.end()) {
                // Could be float stored as uint via atomic
                uint32_t raw = *static_cast<uint32_t*>(it->second->contents());
                float fval;
                memcpy(&fval, &raw, sizeof(float));
                value = static_cast<double>(fval);
            }
        }

        // Apply scale-down (divide by the scaleDown factor, e.g. 100 → /100)
        if (entry.scaleDown > 0) {
            value /= static_cast<double>(entry.scaleDown);
        }

        row.push_back(value);
    }

    if (!row.empty())
        result.rows.push_back(std::move(row));

    return result;
}

// ===================================================================
// collectKeyedAgg
// ===================================================================

GenericResult MetalResultCollector::collectKeyedAgg(const MetalResultSchema& schema,
                                                    const BufferMap& buffers) {
    GenericResult result;

    auto it = buffers.find(schema.keyedAgg.bufferName);
    if (it == buffers.end()) return result;

    const auto* data = static_cast<const uint32_t*>(it->second->contents());
    int nb = schema.keyedAgg.numBuckets;
    int vpb = schema.keyedAgg.valuesPerBucket;
    const auto& slots = schema.keyedAgg.slots;

    // Build column headers
    result.columns.push_back({"bucket", "int"});
    if (!slots.empty()) {
        for (const auto& slot : slots) {
            result.columns.push_back({slot.name, slot.isLongPair ? "long" : "uint"});
        }
    } else {
        for (int v = 0; v < vpb; v++) {
            result.columns.push_back({"val_" + std::to_string(v), "uint"});
        }
    }

    for (int b = 0; b < nb; b++) {
        // Check if bucket has data
        bool hasData = false;
        for (int v = 0; v < vpb; v++) {
            if (data[b * vpb + v] != 0) { hasData = true; break; }
        }
        if (!hasData) continue;

        GenericResult::Row row;
        row.push_back((int64_t)b);  // bucket key

        if (!slots.empty()) {
            for (const auto& slot : slots) {
                if (slot.isLongPair) {
                    uint32_t lo = data[b * vpb + slot.offset];
                    uint32_t hi = data[b * vpb + slot.offset + 1];
                    int64_t val = ((int64_t)hi << 32) | (int64_t)lo;
                    if (slot.scaleDown > 0)
                        row.push_back((double)val / slot.scaleDown);
                    else
                        row.push_back(val);
                } else {
                    int64_t val = (int64_t)data[b * vpb + slot.offset];
                    if (slot.scaleDown > 0)
                        row.push_back((double)val / slot.scaleDown);
                    else
                        row.push_back(val);
                }
            }
        } else {
            for (int v = 0; v < vpb; v++) {
                row.push_back((int64_t)data[b * vpb + v]);
            }
        }
        result.rows.push_back(std::move(row));
    }

    return result;
}

// ===================================================================
// collectMaterialize
// ===================================================================

GenericResult MetalResultCollector::collectMaterialize(const MetalResultSchema& schema,
                                                       const BufferMap& buffers) {
    GenericResult result;

    // Get row count
    auto cntIt = buffers.find(schema.counterBuffer);
    if (cntIt == buffers.end()) return result;
    uint32_t rowCount = *static_cast<uint32_t*>(cntIt->second->contents());

    // Build columns
    for (const auto& col : schema.columns) {
        result.columns.push_back({col.displayName, col.elementType});
    }

    // Read rows
    for (uint32_t r = 0; r < rowCount; r++) {
        GenericResult::Row row;
        for (const auto& col : schema.columns) {
            auto bIt = buffers.find(col.bufferName);
            if (bIt == buffers.end()) {
                row.push_back((int64_t)0);
                continue;
            }

            if (col.elementType == "float") {
                const auto* arr = static_cast<const float*>(bIt->second->contents());
                row.push_back(static_cast<double>(arr[r]));
            } else if (col.elementType == "int" || col.elementType == "uint") {
                const auto* arr = static_cast<const uint32_t*>(bIt->second->contents());
                row.push_back(static_cast<int64_t>(arr[r]));
            } else if (col.stringLen > 0) {
                const auto* arr = static_cast<const char*>(bIt->second->contents());
                std::string s(arr + r * col.stringLen, col.stringLen);
                // Trim trailing spaces/nulls
                while (!s.empty() && (s.back() == ' ' || s.back() == '\0'))
                    s.pop_back();
                row.push_back(std::move(s));
            } else {
                const auto* arr = static_cast<const uint32_t*>(bIt->second->contents());
                row.push_back(static_cast<int64_t>(arr[r]));
            }
        }
        result.rows.push_back(std::move(row));
    }

    return result;
}

} // namespace codegen
