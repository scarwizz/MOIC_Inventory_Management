"""scripts/05_evaluate.py
-----------------------
Fair evaluation of selected inventory policies under the corrected simulator.
Compares:
1. MOIC_LGBM_v2: LightGBM surrogate optimal policy
2. MOIC_CATB_v2: CatBoost surrogate optimal policy
3. MOIC_XGBM_v2: XGBoost surrogate optimal policy
4. TO_v2: True Optimal (exhaustive oracle search over 75 grid policies)
5. FSH: Fixed Standard Heuristic (R=7, L=3, TSL=0.95)

Hold-Out Protocol:
- Historical statistics: days d_1186 to d_1550 (365 days)
- Warm-up period: days d_1551 to d_1610 (60 days)
- Evaluation period: days d_1611 to d_1638 (28 days)

Cost Parameters:
- Primary: h=0.01/year, b=0.25, k=0.50
- Sensitivity: h=0.20/year, b=0.25, k=0.50

Outputs saved to:
- data/processed/evaluation_results_v2.csv
- data/processed/evaluation_summary_v2.json
"""

import sys
import time
import json
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.evaluation import evaluate_sku_policy, find_true_optimal_policy, summarize_evaluation_results
from src.optimization import compute_inventory_cost

# Days definition in sales_train_level_11.csv
# 1-indexed days: d_1 to d_1913
# Historical stats: d_1186 to d_1550 (365 days) -> index 1185 to 1550
# Warm-up: d_1551 to d_1610 (60 days) -> index 1550 to 1610
# Evaluation: d_1611 to d_1638 (28 days) -> index 1610 to 1638
HIST_START, HIST_END = 1185, 1550
WARMUP_START, WARMUP_END = 1550, 1610
EVAL_START, EVAL_END = 1610, 1638

SLICE_HIST = slice(HIST_START, HIST_END)
SLICE_WARMUP = slice(WARMUP_START, WARMUP_END)
SLICE_EVAL = slice(EVAL_START, EVAL_END)


def run_evaluation(
    sales_path: Path = None,
    policies_dir: Path = None,
    output_dir: Path = None,
    limit_skus: int = None,
    verbose: bool = True,
):
    if sales_path is None:
        sales_path = project_root / "data" / "processed" / "sales_train_level_11.csv"
        if not sales_path.exists():
            sales_path = project_root / "Dataset" / "sales_train_level_11.csv"
    if policies_dir is None:
        policies_dir = project_root / "data" / "processed"
    if output_dir is None:
        output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading sales data from: {sales_path}")
    sales_df = pd.read_csv(sales_path)
    if limit_skus is not None:
        sales_df = sales_df.iloc[:limit_skus]
        if verbose:
            print(f"Subsetting to {limit_skus} SKUs.")

    n_skus = len(sales_df)
    day_cols = [f"d_{i}" for i in range(1, 1914)]
    demand_matrix = sales_df[day_cols].values.astype(float)

    # Load optimal policy CSVs
    lgbm_pol = pd.read_csv(policies_dir / "optimal_policies_v2_LGBM.csv")
    catb_pol = pd.read_csv(policies_dir / "optimal_policies_v2_CATB.csv")
    xgbm_pol_path = policies_dir / "optimal_policies_v2_XGBM.csv"
    has_xgbm = xgbm_pol_path.exists()
    xgbm_pol = pd.read_csv(xgbm_pol_path) if has_xgbm else None

    if limit_skus is not None:
        lgbm_pol = lgbm_pol.iloc[:limit_skus]
        catb_pol = catb_pol.iloc[:limit_skus]
        if has_xgbm:
            xgbm_pol = xgbm_pol.iloc[:limit_skus]

    records = []
    records_retail = []

    eval_methods_str = "5 policies (LGBM, CATB, XGBM, TO, FSH)" if has_xgbm else "4 policies (LGBM, CATB, TO, FSH)"
    if verbose:
        print(f"Evaluating {n_skus:,} SKUs across {eval_methods_str}...")

    start_time = time.time()

    for i in tqdm(range(n_skus), desc="Evaluating SKUs", disable=not verbose):
        series = demand_matrix[i]
        item_id = sales_df.iloc[i]["item_id"]
        state_id = sales_df.iloc[i]["state_id"]

        row_res = {"item_id": item_id, "state_id": state_id}
        row_retail = {"item_id": item_id, "state_id": state_id}

        # 1. MOIC_LGBM_v2
        r_lgbm, l_lgbm, tsl_lgbm = int(lgbm_pol.iloc[i]["R"]), int(lgbm_pol.iloc[i]["L"]), float(lgbm_pol.iloc[i]["TSL"])
        m_lgbm = evaluate_sku_policy(series, SLICE_HIST, SLICE_WARMUP, SLICE_EVAL, r_lgbm, l_lgbm, tsl_lgbm, h_annual=0.01)
        m_lgbm_retail = evaluate_sku_policy(series, SLICE_HIST, SLICE_WARMUP, SLICE_EVAL, r_lgbm, l_lgbm, tsl_lgbm, h_annual=0.20)

        # 2. MOIC_CATB_v2
        r_catb, l_catb, tsl_catb = int(catb_pol.iloc[i]["R"]), int(catb_pol.iloc[i]["L"]), float(catb_pol.iloc[i]["TSL"])
        m_catb = evaluate_sku_policy(series, SLICE_HIST, SLICE_WARMUP, SLICE_EVAL, r_catb, l_catb, tsl_catb, h_annual=0.01)
        m_catb_retail = evaluate_sku_policy(series, SLICE_HIST, SLICE_WARMUP, SLICE_EVAL, r_catb, l_catb, tsl_catb, h_annual=0.20)

        # 3. MOIC_XGBM_v2 (if available)
        policy_tuples = [
            ("LGBM", m_lgbm, r_lgbm, l_lgbm, tsl_lgbm),
            ("CATB", m_catb, r_catb, l_catb, tsl_catb),
        ]
        retail_tuples = [("LGBM", m_lgbm_retail), ("CATB", m_catb_retail)]

        if has_xgbm:
            r_xgbm, l_xgbm, tsl_xgbm = int(xgbm_pol.iloc[i]["R"]), int(xgbm_pol.iloc[i]["L"]), float(xgbm_pol.iloc[i]["TSL"])
            m_xgbm = evaluate_sku_policy(series, SLICE_HIST, SLICE_WARMUP, SLICE_EVAL, r_xgbm, l_xgbm, tsl_xgbm, h_annual=0.01)
            m_xgbm_retail = evaluate_sku_policy(series, SLICE_HIST, SLICE_WARMUP, SLICE_EVAL, r_xgbm, l_xgbm, tsl_xgbm, h_annual=0.20)
            policy_tuples.append(("XGBM", m_xgbm, r_xgbm, l_xgbm, tsl_xgbm))
            retail_tuples.append(("XGBM", m_xgbm_retail))

        # 4. FSH (R=7, L=3, TSL=0.95)
        m_fsh = evaluate_sku_policy(series, SLICE_HIST, SLICE_WARMUP, SLICE_EVAL, 7, 3, 0.95, h_annual=0.01)
        m_fsh_retail = evaluate_sku_policy(series, SLICE_HIST, SLICE_WARMUP, SLICE_EVAL, 7, 3, 0.95, h_annual=0.20)

        # 5. TO_v2 (True Optimal Oracle)
        to_r, to_l, to_tsl, m_to = find_true_optimal_policy(series, SLICE_HIST, SLICE_WARMUP, SLICE_EVAL, h_annual=0.01)
        to_r_ret, to_l_ret, to_tsl_ret, m_to_retail = find_true_optimal_policy(series, SLICE_HIST, SLICE_WARMUP, SLICE_EVAL, h_annual=0.20)

        policy_tuples.extend([("TO", m_to, to_r, to_l, to_tsl), ("FSH", m_fsh, 7, 3, 0.95)])
        retail_tuples.extend([("TO", m_to_retail), ("FSH", m_fsh_retail)])

        # Populate primary records
        for prefix, m, r_val, l_val, tsl_val in policy_tuples:
            row_res[f"{prefix}_R"] = r_val
            row_res[f"{prefix}_L"] = l_val
            row_res[f"{prefix}_TSL"] = tsl_val
            row_res[f"{prefix}_avg_inventory"] = m["avg_inventory"]
            row_res[f"{prefix}_lost_sales"] = m["lost_sales"]
            row_res[f"{prefix}_orders"] = m["orders_placed"]
            row_res[f"{prefix}_fill_rate"] = m["fill_rate"]
            row_res[f"{prefix}_cost"] = m["cost"]

        # Populate retail records
        for prefix, m in retail_tuples:
            row_retail[f"{prefix}_cost"] = m["cost"]
            row_retail[f"{prefix}_fill_rate"] = m["fill_rate"]

        records.append(row_res)
        records_retail.append(row_retail)

    elapsed = time.time() - start_time
    if verbose:
        print(f"\nEvaluation finished in {elapsed:.1f}s ({n_skus/elapsed:.1f} SKUs/sec).")

    eval_df = pd.DataFrame(records)
    eval_csv = output_dir / "evaluation_results_v2.csv"
    eval_df.to_csv(eval_csv, index=False)
    if verbose:
        print(f"Saved detailed evaluation results to: {eval_csv}")

    # Compute summaries
    summary_primary = summarize_evaluation_results(eval_df, h_annual=0.01)
    eval_retail_df = pd.DataFrame(records_retail)
    summary_retail = summarize_evaluation_results(eval_retail_df, h_annual=0.20)

    consolidated_summary = {
        "primary_h001": summary_primary,
        "retail_h020": summary_retail,
        "elapsed_seconds": elapsed,
    }

    summary_json_path = output_dir / "evaluation_summary_v2.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(consolidated_summary, f, indent=2)
    if verbose:
        print(f"Saved evaluation summary JSON to: {summary_json_path}")

    # Display printed report
    if verbose:
        print("\n==========================================================================")
        print("MOIC v2 EVALUATION SUMMARY (Primary Cost: h=0.01/yr, b=0.25, k=0.50)")
        print("==========================================================================")
        print(f"{'Method':<10} {'Mean Cost':<12} {'Median Cost':<12} {'Std Cost':<12} {'Fill Rate':<12}")
        print("-" * 60)
        for m in [k for k in ["LGBM", "CATB", "XGBM", "TO", "FSH"] if k in summary_primary["methods"]]:
            stats = summary_primary["methods"][m]
            fr_str = f"{stats['mean_fill_rate']*100:.2f}%" if stats['mean_fill_rate'] is not None else "N/A"
            print(f"{m:<10} ${stats['mean_cost']:<11.4f} ${stats['median_cost']:<11.4f} ${stats['std_cost']:<11.4f} {fr_str:<12}")

        h2h = summary_primary.get("head_to_head_CATB_vs_LGBM", {})
        print("\nHead-to-Head: CatBoost vs LightGBM:")
        print(f"  Relative Cost Reduction: {h2h.get('relative_cost_reduction', 0)*100:.2f}%")
        print(f"  CatBoost strictly cheaper: {h2h.get('catb_strictly_cheaper_count', 0)} ({h2h.get('catb_strictly_cheaper_prop', 0)*100:.1f}%)")
        print(f"  LightGBM strictly cheaper: {h2h.get('lgbm_strictly_cheaper_count', 0)} ({h2h.get('lgbm_strictly_cheaper_prop', 0)*100:.1f}%)")
        print(f"  Identical policies: {h2h.get('identical_count', 0)} ({h2h.get('identical_prop', 0)*100:.1f}%)")

        print("\n==========================================================================")
        print("MOIC v2 RETAIL SENSITIVITY (Holding Cost: h=0.20/yr, b=0.25, k=0.50)")
        print("==========================================================================")
        print(f"{'Method':<10} {'Mean Cost':<12} {'Median Cost':<12} {'Std Cost':<12} {'Fill Rate':<12}")
        print("-" * 60)
        for m in [k for k in ["LGBM", "CATB", "XGBM", "TO", "FSH"] if k in summary_retail["methods"]]:
            stats = summary_retail["methods"][m]
            fr_str = f"{stats['mean_fill_rate']*100:.2f}%" if stats['mean_fill_rate'] is not None else "N/A"
            print(f"{m:<10} ${stats['mean_cost']:<11.4f} ${stats['median_cost']:<11.4f} ${stats['std_cost']:<11.4f} {fr_str:<12}")

    return consolidated_summary


if __name__ == "__main__":
    run_evaluation()
