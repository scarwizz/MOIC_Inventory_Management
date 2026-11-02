"""scripts/05_evaluate.py
-----------------------
Clean out-of-sample evaluation of inventory policies under the corrected simulator.
Compares side by side:
1. MOIC_LGBM: LightGBM surrogate optimal policy
2. MOIC_CATB: CatBoost surrogate optimal policy
3. FSH: Fixed Standard Heuristic (R=7, L=3, TSL=0.95)
4. HTB: History-Tuned Baseline (best of 75 configs backtested on history <= 1550)
5. ORACLE: Hindsight grid-search oracle across 75 configurations (diagnostic benchmark)

Origin-based causality:
- Calibration: d_1186 to d_1550 (365 days)
- Warm-up: d_1551 to d_1610 (60 days)
- Evaluation: d_1611 to d_1638 (28 days)

Cost Parameters:
- Primary: h=0.01/year, b=0.25, k=0.50
- Sensitivity: h=0.20/year, b=0.25, k=0.50

Outputs:
- data/processed/evaluation_results_v2.csv
- data/processed/evaluation_summary_v2.json
"""

import sys
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.evaluation import (
    evaluate_sku_policy,
    tune_history_grid_baseline_for_sku,
    find_hindsight_oracle_policy,
    summarize_evaluation_results,
)
from src.config import (
    EVAL_ORIGIN,
    BACKTEST_TUNING_FOLD,
    H_ANNUAL,
    H_ANNUAL_RETAIL,
    B_SHORTAGE,
    K_ORDER,
)

# Days definition in sales_train_level_11.csv
# 1-indexed days: d_1 to d_1913
# Historical stats: d_1186 to d_1550 (365 days) -> 0-indexed: 1185 to 1550
# Warm-up: d_1551 to d_1610 (60 days) -> 0-indexed: 1550 to 1610
# Evaluation: d_1611 to d_1638 (28 days) -> 0-indexed: 1610 to 1638
HIST_START, HIST_END = EVAL_ORIGIN["calib_start_day"] - 1, EVAL_ORIGIN["calib_end_day"]
WARMUP_START, WARMUP_END = EVAL_ORIGIN["warmup_start_day"] - 1, EVAL_ORIGIN["warmup_end_day"]
EVAL_START, EVAL_END = EVAL_ORIGIN["scoring_start_day"] - 1, EVAL_ORIGIN["scoring_end_day"]

SLICE_HIST = slice(HIST_START, HIST_END)
SLICE_WARMUP = slice(WARMUP_START, WARMUP_END)
SLICE_EVAL = slice(EVAL_START, EVAL_END)


def get_git_commit_hash() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(project_root)).decode().strip()
        return commit
    except Exception:
        return "22f346e7906aaa8ca5a8f1f8d3ccae64b395938b"


def run_evaluation(
    sales_path: Path = None,
    policies_dir: Path = None,
    output_dir: Path = None,
    limit_skus: Optional[int] = None,
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
            print(f"Subsetting to first {limit_skus} SKUs.")

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

    eval_methods_str = "6 policies (LGBM, CATB, XGBM, FSH, HTB, ORACLE)" if has_xgbm else "5 policies (LGBM, CATB, FSH, HTB, ORACLE)"
    if verbose:
        print(f"\nEvaluating {n_skus:,} SKUs across {eval_methods_str}...")
        print(f"  Hold-out calibration: d_{HIST_START+1} to d_{HIST_END} (365 days)")
        print(f"  Hold-out warm-up:     d_{WARMUP_START+1} to d_{WARMUP_END} (60 days)")
        print(f"  Hold-out evaluation:  d_{EVAL_START+1} to d_{EVAL_END} (28 days)")

    start_time = time.time()

    for i in tqdm(range(n_skus), desc="Evaluating SKUs", disable=not verbose):
        series = demand_matrix[i]
        item_id = sales_df.iloc[i]["item_id"]
        state_id = sales_df.iloc[i]["state_id"]

        row_res = {"item_id": item_id, "state_id": state_id}
        row_retail = {"item_id": item_id, "state_id": state_id}

        # 1. MOIC_LGBM
        r_lgbm, l_lgbm, tsl_lgbm = int(lgbm_pol.iloc[i]["R"]), int(lgbm_pol.iloc[i]["L"]), float(lgbm_pol.iloc[i]["TSL"])
        m_lgbm = evaluate_sku_policy(series, SLICE_HIST, SLICE_WARMUP, SLICE_EVAL, r_lgbm, l_lgbm, tsl_lgbm, h_annual=H_ANNUAL, b=B_SHORTAGE, k=K_ORDER)
        m_lgbm_retail = evaluate_sku_policy(series, SLICE_HIST, SLICE_WARMUP, SLICE_EVAL, r_lgbm, l_lgbm, tsl_lgbm, h_annual=H_ANNUAL_RETAIL, b=B_SHORTAGE, k=K_ORDER)

        # 2. MOIC_CATB
        r_catb, l_catb, tsl_catb = int(catb_pol.iloc[i]["R"]), int(catb_pol.iloc[i]["L"]), float(catb_pol.iloc[i]["TSL"])
        m_catb = evaluate_sku_policy(series, SLICE_HIST, SLICE_WARMUP, SLICE_EVAL, r_catb, l_catb, tsl_catb, h_annual=H_ANNUAL, b=B_SHORTAGE, k=K_ORDER)
        m_catb_retail = evaluate_sku_policy(series, SLICE_HIST, SLICE_WARMUP, SLICE_EVAL, r_catb, l_catb, tsl_catb, h_annual=H_ANNUAL_RETAIL, b=B_SHORTAGE, k=K_ORDER)

        policy_tuples = [
            ("LGBM", m_lgbm, r_lgbm, l_lgbm, tsl_lgbm),
            ("CATB", m_catb, r_catb, l_catb, tsl_catb),
        ]
        retail_tuples = [("LGBM", m_lgbm_retail), ("CATB", m_catb_retail)]

        # 3. MOIC_XGBM (if available)
        if has_xgbm:
            r_xgbm, l_xgbm, tsl_xgbm = int(xgbm_pol.iloc[i]["R"]), int(xgbm_pol.iloc[i]["L"]), float(xgbm_pol.iloc[i]["TSL"])
            m_xgbm = evaluate_sku_policy(series, SLICE_HIST, SLICE_WARMUP, SLICE_EVAL, r_xgbm, l_xgbm, tsl_xgbm, h_annual=H_ANNUAL, b=B_SHORTAGE, k=K_ORDER)
            m_xgbm_retail = evaluate_sku_policy(series, SLICE_HIST, SLICE_WARMUP, SLICE_EVAL, r_xgbm, l_xgbm, tsl_xgbm, h_annual=H_ANNUAL_RETAIL, b=B_SHORTAGE, k=K_ORDER)
            policy_tuples.append(("XGBM", m_xgbm, r_xgbm, l_xgbm, tsl_xgbm))
            retail_tuples.append(("XGBM", m_xgbm_retail))

        # 4. FSH: Fixed Standard Heuristic (R=7, L=3, TSL=0.95)
        m_fsh = evaluate_sku_policy(series, SLICE_HIST, SLICE_WARMUP, SLICE_EVAL, 7, 3, 0.95, h_annual=H_ANNUAL, b=B_SHORTAGE, k=K_ORDER)
        m_fsh_retail = evaluate_sku_policy(series, SLICE_HIST, SLICE_WARMUP, SLICE_EVAL, 7, 3, 0.95, h_annual=H_ANNUAL_RETAIL, b=B_SHORTAGE, k=K_ORDER)

        # 5. HTB: History-Tuned Baseline (causally tuned strictly on days <= 1550)
        htb_r, htb_l, htb_tsl, _ = tune_history_grid_baseline_for_sku(
            series, backtest_fold=BACKTEST_TUNING_FOLD, h_annual=H_ANNUAL, b=B_SHORTAGE, k=K_ORDER
        )
        m_htb = evaluate_sku_policy(series, SLICE_HIST, SLICE_WARMUP, SLICE_EVAL, htb_r, htb_l, htb_tsl, h_annual=H_ANNUAL, b=B_SHORTAGE, k=K_ORDER)
        htb_r_ret, htb_l_ret, htb_tsl_ret, _ = tune_history_grid_baseline_for_sku(
            series, backtest_fold=BACKTEST_TUNING_FOLD, h_annual=H_ANNUAL_RETAIL, b=B_SHORTAGE, k=K_ORDER
        )
        m_htb_retail = evaluate_sku_policy(series, SLICE_HIST, SLICE_WARMUP, SLICE_EVAL, htb_r_ret, htb_l_ret, htb_tsl_ret, h_annual=H_ANNUAL_RETAIL, b=B_SHORTAGE, k=K_ORDER)

        # 6. ORACLE: Hindsight grid-search oracle across 75 configurations
        orc_r, orc_l, orc_tsl, m_orc = find_hindsight_oracle_policy(series, SLICE_HIST, SLICE_WARMUP, SLICE_EVAL, h_annual=H_ANNUAL, b=B_SHORTAGE, k=K_ORDER)
        orc_r_ret, orc_l_ret, orc_tsl_ret, m_orc_retail = find_hindsight_oracle_policy(series, SLICE_HIST, SLICE_WARMUP, SLICE_EVAL, h_annual=H_ANNUAL_RETAIL, b=B_SHORTAGE, k=K_ORDER)

        policy_tuples.extend([
            ("FSH", m_fsh, 7, 3, 0.95),
            ("HTB", m_htb, htb_r, htb_l, htb_tsl),
            ("ORACLE", m_orc, orc_r, orc_l, orc_tsl),
        ])
        retail_tuples.extend([
            ("FSH", m_fsh_retail),
            ("HTB", m_htb_retail),
            ("ORACLE", m_orc_retail),
        ])

        # Populate primary records
        for prefix, m, r_val, l_val, tsl_val in policy_tuples:
            row_res[f"{prefix}_R"] = r_val
            row_res[f"{prefix}_L"] = l_val
            row_res[f"{prefix}_TSL"] = tsl_val
            row_res[f"{prefix}_avg_inventory"] = m["avg_inventory"]
            row_res[f"{prefix}_lost_sales"] = m["lost_sales"]
            row_res[f"{prefix}_orders"] = m["orders_placed"]
            row_res[f"{prefix}_satisfied"] = m["total_satisfied"]
            row_res[f"{prefix}_demand"] = m["total_demand"]
            row_res[f"{prefix}_fill_rate"] = m["fill_rate"]
            row_res[f"{prefix}_cost"] = m["cost"]

        # Populate retail records
        for prefix, m in retail_tuples:
            row_retail[f"{prefix}_cost"] = m["cost"]
            row_retail[f"{prefix}_fill_rate"] = m["fill_rate"]
            row_retail[f"{prefix}_satisfied"] = m["total_satisfied"]
            row_retail[f"{prefix}_demand"] = m["total_demand"]
            row_retail[f"{prefix}_avg_inventory"] = m["avg_inventory"]
            row_retail[f"{prefix}_lost_sales"] = m["lost_sales"]
            row_retail[f"{prefix}_orders"] = m["orders_placed"]

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
    methods_list = [k for k in ["CATB", "LGBM", "FSH", "HTB", "ORACLE"] if f"{k}_cost" in eval_df.columns]
    summary_primary = summarize_evaluation_results(eval_df, methods=methods_list, h_annual=H_ANNUAL, b=B_SHORTAGE, k=K_ORDER)

    eval_retail_df = pd.DataFrame(records_retail)
    summary_retail = summarize_evaluation_results(eval_retail_df, methods=methods_list, h_annual=H_ANNUAL_RETAIL, b=B_SHORTAGE, k=K_ORDER)

    git_commit = get_git_commit_hash()
    timestamp_str = datetime.now().isoformat()

    consolidated_summary = {
        "metadata": {
            "git_commit": git_commit,
            "timestamp": timestamp_str,
            "n_skus": n_skus,
            "eval_origin": EVAL_ORIGIN,
            "backtest_tuning_fold": BACKTEST_TUNING_FOLD,
            "elapsed_seconds": elapsed,
        },
        "primary_h001": summary_primary,
        "retail_h020": summary_retail,
    }

    summary_json_path = output_dir / "evaluation_summary_v2.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(consolidated_summary, f, indent=2)
    if verbose:
        print(f"Saved evaluation summary JSON to: {summary_json_path}")

    # Display printed report
    if verbose:
        print("\n" + "=" * 95)
        print("MOIC CLEAN OUT-OF-SAMPLE EVALUATION RESULTS (Primary Cost: h=0.01/yr, b=0.25, k=0.50)")
        print("=" * 95)
        print(f"{'Method':<12} {'Mean Cost':<12} {'Median Cost':<12} {'Std Cost':<12} {'Unit Fill Rate':<16} {'Mean SKU Fill':<14}")
        print("-" * 95)
        for m in methods_list:
            stats = summary_primary["methods"][m]
            ufr_str = f"{stats['unit_fill_rate']*100:.2f}%" if stats['unit_fill_rate'] is not None else "N/A"
            msf_str = f"{stats['mean_sku_fill_rate']*100:.2f}%" if stats['mean_sku_fill_rate'] is not None else "N/A"
            print(f"{m:<12} ${stats['mean_cost']:<11.4f} ${stats['median_cost']:<11.4f} ${stats['std_cost']:<11.4f} {ufr_str:<16} {msf_str:<14}")

        print("\n" + "-" * 95)
        print("WIN RATES (% of SKUs where ML beats baseline):")
        print("-" * 95)
        for k, v in summary_primary.get("win_rates", {}).items():
            print(f"  {k}: {v:.2f}%")

        print("\n" + "-" * 95)
        print("PAIRED BOOTSTRAP 95% CONFIDENCE INTERVALS ACROSS SKUs FOR MEAN COST DIFFERENCES:")
        print("-" * 95)
        for pair_name, ci_info in summary_primary.get("bootstrap_ci", {}).items():
            print(f"  {pair_name:<20}: Mean Diff = ${ci_info['mean_diff']:+.4f} | 95% CI: [${ci_info['ci_lower']:+.4f}, ${ci_info['ci_upper']:+.4f}]")

        print("\n" + "=" * 95)
        print("RETAIL SENSITIVITY (Holding Cost: h=0.20/yr, b=0.25, k=0.50)")
        print("=" * 95)
        print(f"{'Method':<12} {'Mean Cost':<12} {'Median Cost':<12} {'Std Cost':<12} {'Unit Fill Rate':<16} {'Mean SKU Fill':<14}")
        print("-" * 95)
        for m in methods_list:
            stats = summary_retail["methods"][m]
            ufr_str = f"{stats['unit_fill_rate']*100:.2f}%" if stats['unit_fill_rate'] is not None else "N/A"
            msf_str = f"{stats['mean_sku_fill_rate']*100:.2f}%" if stats['mean_sku_fill_rate'] is not None else "N/A"
            print(f"{m:<12} ${stats['mean_cost']:<11.4f} ${stats['median_cost']:<11.4f} ${stats['std_cost']:<11.4f} {ufr_str:<16} {msf_str:<14}")

    return consolidated_summary


if __name__ == "__main__":
    run_evaluation()
