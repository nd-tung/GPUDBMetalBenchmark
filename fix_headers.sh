#!/bin/bash
set -e
cd /Users/tea/Documents/GPUDBMetalBenchmarkLC
f="codegen/metal_codegen.cpp"

sed -i '' 's|// Q1 KERNEL GENERATOR|// PATTERN: Two-Stage Reduce (Q1 — 6-bin fused reduce)|g' "$f"
sed -i '' 's|// Q6 KERNEL GENERATOR|// PATTERN: Two-Stage Reduce (Q6 — single-sum filter+reduce)|g' "$f"
sed -i '' 's|// Q3 KERNEL GENERATOR|// PATTERN: Bitmap + Map Build + Probe-Agg + Compact (Q3)|g' "$f"
sed -i '' 's|// Q14 KERNEL GENERATOR|// PATTERN: Two-Stage Reduce with Bitmap Probe (Q14)|g' "$f"
sed -i '' 's|// Q13 KERNEL GENERATOR|// PATTERN: String Match + Histogram (Q13)|g' "$f"
sed -i '' 's|// Q4 KERNEL GENERATOR|// PATTERN: Bitmap Build + N-bin Reduce (Q4)|g' "$f"
sed -i '' 's|// Q12 KERNEL GENERATOR|// PATTERN: Bitmap Build + N-bin Reduce (Q12)|g' "$f"
sed -i '' 's|// Q19 KERNEL GENERATOR|// PATTERN: Map Classify + String Filter + Reduce (Q19)|g' "$f"
sed -i '' 's|// Q15 KERNEL GENERATOR|// PATTERN: Atomic Map Aggregate (Q15)|g' "$f"
sed -i '' 's|// Q11 KERNEL GENERATOR|// PATTERN: Atomic Aggregate (Q11)|g' "$f"
sed -i '' 's|// Q10 KERNEL GENERATOR|// PATTERN: Map Build + Probe-Agg (Q10)|g' "$f"
sed -i '' 's|// Q5 KERNEL GENERATOR|// PATTERN: Multi-Map Build + Probe-Agg (Q5)|g' "$f"
sed -i '' 's|// Q7 KERNEL GENERATOR|// PATTERN: Multi-Map Build + Probe-Agg (Q7)|g' "$f"
sed -i '' 's|// Q8 KERNEL GENERATOR|// PATTERN: Map Build + Probe-Agg (Q8)|g' "$f"

# These don't have "KERNEL GENERATOR" headers - update their existing comment headers
sed -i '' 's|// Q9: Product Type Profit Measure|// PATTERN: Bitmap + Map + HT Build + Probe-Agg (Q9 — Product Type Profit)|g' "$f"
sed -i '' 's|// Q17: Small-Quantity-Order Revenue|// PATTERN: Atomic Agg + Probe-Reduce (Q17 — Small-Qty-Order Revenue)|g' "$f"
sed -i '' 's|// Q22: Global Sales Opportunity|// PATTERN: Scalar Agg + Bitmap Build + Bin Agg (Q22 — Global Sales)|g' "$f"
sed -i '' 's|// Q18: Large Volume Customer|// PATTERN: Atomic Agg + Compact (Q18 — Large Volume Customer)|g' "$f"
sed -i '' 's|// Q16: Parts/Supplier Relationship|// PATTERN: Group Bitmap + Popcount (Q16 — Parts/Supplier)|g' "$f"
sed -i '' 's|// Q20: Potential Part Promotion|// PATTERN: HT Agg + Probe Check (Q20 — Part Promotion)|g' "$f"
sed -i '' 's|// Q21: Suppliers Who Kept Orders Waiting|// PATTERN: Track Build + Count Qualify (Q21 — Supplier Wait)|g' "$f"

# Also update "Q2" section if it has a header
sed -i '' 's|// Q2: Minimum Cost Supplier|// PATTERN: Bitmap + Atomic Min + Compact (Q2 — Min Cost Supplier)|g' "$f"

echo "Section headers updated."
