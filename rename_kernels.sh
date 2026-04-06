#!/bin/bash
# Rename all kernel names from gen_qN_* to pattern-based naming
set -e

FILES="codegen/metal_codegen.cpp codegen/query_executor.cpp"

for f in $FILES; do
  # Q1
  sed -i '' 's/gen_q1_fused/Q1_reduce/g' "$f"
  sed -i '' 's/q1_rf_index/Q1_rf_index/g' "$f"
  sed -i '' 's/q1_ls_index/Q1_ls_index/g' "$f"
  # Q2
  sed -i '' 's/gen_q2_filter_part/Q2_bitmap_build/g' "$f"
  sed -i '' 's/gen_q2_find_min_cost/Q2_atomic_min/g' "$f"
  sed -i '' 's/gen_q2_match_suppliers/Q2_compact/g' "$f"
  # Q3
  sed -i '' 's/gen_q3_build_customer_bitmap/Q3_bitmap_build/g' "$f"
  sed -i '' 's/gen_q3_build_orders_map/Q3_map_build/g' "$f"
  sed -i '' 's/gen_q3_probe_agg/Q3_probe_agg/g' "$f"
  sed -i '' 's/gen_q3_compact/Q3_compact/g' "$f"
  # Q4
  sed -i '' 's/gen_q4_build_late_bitmap/Q4_bitmap_build/g' "$f"
  sed -i '' 's/gen_q4_count_stage1/Q4_reduce/g' "$f"
  sed -i '' 's/gen_q4_count_stage2/Q4_reduce_final/g' "$f"
  # Q5
  sed -i '' 's/gen_q5_build_customer_nation_map/Q5_map_build_cust/g' "$f"
  sed -i '' 's/gen_q5_build_supplier_nation_map/Q5_map_build_supp/g' "$f"
  sed -i '' 's/gen_q5_build_orders_map/Q5_map_build_orders/g' "$f"
  sed -i '' 's/gen_q5_probe_and_aggregate/Q5_probe_agg/g' "$f"
  # Q6
  sed -i '' 's/gen_q6_stage1/Q6_reduce/g' "$f"
  sed -i '' 's/gen_q6_stage2/Q6_reduce_final/g' "$f"
  # Q7
  sed -i '' 's/gen_q7_build_supplier_map/Q7_map_build_supp/g' "$f"
  sed -i '' 's/gen_q7_build_customer_map/Q7_map_build_cust/g' "$f"
  sed -i '' 's/gen_q7_build_orders_map/Q7_map_build_orders/g' "$f"
  sed -i '' 's/gen_q7_probe_and_aggregate/Q7_probe_agg/g' "$f"
  # Q8
  sed -i '' 's/gen_q8_build_orders_map/Q8_map_build/g' "$f"
  sed -i '' 's/gen_q8_probe_and_aggregate/Q8_probe_agg/g' "$f"
  # Q9
  sed -i '' 's/gen_q9_build_part_bitmap/Q9_bitmap_build/g' "$f"
  sed -i '' 's/gen_q9_build_supplier_map/Q9_map_build/g' "$f"
  sed -i '' 's/gen_q9_build_partsupp_ht/Q9_ht_build_partsupp/g' "$f"
  sed -i '' 's/gen_q9_build_orders_ht/Q9_ht_build_orders/g' "$f"
  sed -i '' 's/gen_q9_probe_and_aggregate/Q9_probe_agg/g' "$f"
  # Q10
  sed -i '' 's/gen_q10_build_orders_map/Q10_map_build/g' "$f"
  sed -i '' 's/gen_q10_probe_and_aggregate/Q10_probe_agg/g' "$f"
  # Q11
  sed -i '' 's/gen_q11_aggregate_value/Q11_atomic_agg/g' "$f"
  # Q12
  sed -i '' 's/gen_q12_build_priority_bitmap/Q12_bitmap_build/g' "$f"
  sed -i '' 's/gen_q12_filter_stage1/Q12_reduce/g' "$f"
  sed -i '' 's/gen_q12_filter_stage2/Q12_reduce_final/g' "$f"
  # Q13
  sed -i '' 's/gen_q13_count_orders/Q13_str_match_count/g' "$f"
  sed -i '' 's/gen_q13_histogram/Q13_histogram/g' "$f"
  # Q14
  sed -i '' 's/gen_q14_stage1/Q14_reduce/g' "$f"
  sed -i '' 's/gen_q14_stage2/Q14_reduce_final/g' "$f"
  # Q15
  sed -i '' 's/gen_q15_aggregate_revenue/Q15_atomic_agg/g' "$f"
  # Q16
  sed -i '' 's/gen_q16_scan_and_bitmap/Q16_group_bitmap/g' "$f"
  sed -i '' 's/gen_q16_popcount/Q16_popcount/g' "$f"
  # Q17
  sed -i '' 's/gen_q17_aggregate_qty_stats/Q17_atomic_agg/g' "$f"
  sed -i '' 's/gen_q17_sum_revenue/Q17_probe_reduce/g' "$f"
  # Q18
  sed -i '' 's/gen_q18_aggregate_quantity/Q18_atomic_agg/g' "$f"
  sed -i '' 's/gen_q18_filter_orders/Q18_compact/g' "$f"
  # Q19
  sed -i '' 's/gen_q19_build_part_group_map/Q19_map_classify/g' "$f"
  sed -i '' 's/gen_q19_shipmode_filter/Q19_str_filter/g' "$f"
  sed -i '' 's/gen_q19_sum_stage1/Q19_reduce/g' "$f"
  sed -i '' 's/gen_q19_sum_stage2/Q19_reduce_final/g' "$f"
  # Q20
  sed -i '' 's/gen_q20_aggregate_lineitem/Q20_ht_agg/g' "$f"
  sed -i '' 's/gen_q20_probe_partsupp/Q20_probe_check/g' "$f"
  # Q21
  sed -i '' 's/gen_q21_build_order_tracking/Q21_track_build/g' "$f"
  sed -i '' 's/gen_q21_count_qualifying/Q21_count_qualify/g' "$f"
  # Q22
  sed -i '' 's/gen_q22_avg_balance/Q22_scalar_agg/g' "$f"
  sed -i '' 's/gen_q22_build_orders_bitmap/Q22_bitmap_build/g' "$f"
  sed -i '' 's/gen_q22_final_aggregate/Q22_bin_agg/g' "$f"
  echo "Renamed kernels in $f"
done

echo "All kernel names renamed."
