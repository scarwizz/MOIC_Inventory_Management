"""evaluation.py
-------------
Fair evaluation pipeline for MOIC inventory policies under the corrected simulator.
Includes warm-up horizon, historical stats estimation, and side-by-side comparison
against the True Optimal (oracle) and Fixed Standard Heuristic (FSH) baselines.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.simulator import run_inventory_simulation, calculate_policy_parameters
from src.optimization import compute_inventory_cost, generate_candidate_policies


def evaluate_sku_policy(
    demand_series: np.ndarray,
    history_slice: slice,
    warmup_slice: slice,
    eval_slice: slice,
    r: int,
    l: int,
    tsl: float,
    h_annual: float = 0.01,
    b: float = 0.25,
    k: float = 0.50,
) -> Dict[str, float]:
    """Evaluate a single (R, L, TSL) policy on a time series with historical stats & warm-up."""
    # 1. Historical stats & policy params
    hist_demand = demand_series[history_slice]
    _, _, _, s, S = calculate_policy_parameters(historical_sales=hist_demand, R=r, L=l, TSL=tsl)

    # 2. Demand for warmup + eval
    sim_demand = np.concatenate([demand_series[warmup_slice], demand_series[eval_slice]])
    warmup_days = warmup_slice.stop - warmup_slice.start
    eval_days = eval_slice.stop - eval_slice.start

    # 3. Simulate with warm-up
    res = run_inventory_simulation(
        demand_series=sim_demand,
        R=r,
        L=l,
        s=s,
        S=S,
        warmup_days=warmup_days,
        h_daily=h_annual / 365.0,
        b_shortage=b,
        k_order=k,
    )

    return {
        "avg_inventory": res["avg_inventory"],
        "lost_sales": res["total_lost_sales"],
        "orders_placed": res["total_orders"],
        "fill_rate": res["fill_rate"],
        "cost": res["total_cost"],
    }


def find_true_optimal_policy(
    demand_series: np.ndarray,
    history_slice: slice,
    warmup_slice: slice,
    eval_slice: slice,
    h_annual: float = 0.01,
    b: float = 0.25,
    k: float = 0.50,
) -> Tuple[int, int, float, Dict[str, float]]:
    """Brute-force oracle search over all 75 grid policies using the actual simulation."""
    best_cost = float("inf")
    best_policy = (7, 3, 0.95)
    best_metrics: Dict[str, float] = {}

    for r, l, tsl in generate_candidate_policies():
        metrics = evaluate_sku_policy(
            demand_series=demand_series,
            history_slice=history_slice,
            warmup_slice=warmup_slice,
            eval_slice=eval_slice,
            r=r,
            l=l,
            tsl=tsl,
            h_annual=h_annual,
            b=b,
            k=k,
        )
        if metrics["cost"] < best_cost:
            best_cost = metrics["cost"]
            best_policy = (r, l, tsl)
            best_metrics = metrics

    return best_policy[0], best_policy[1], best_policy[2], best_metrics


def summarize_evaluation_results(
    eval_df: pd.DataFrame,
    methods: List[str] = ["LGBM", "CATB", "XGBM", "TO", "FSH"],
    h_annual: float = 0.01,
    b: float = 0.25,
    k: float = 0.50,
) -> Dict[str, Any]:
    """Compute aggregate evaluation metrics across all evaluated policies."""
    summary: Dict[str, Any] = {
        "n_skus": len(eval_df),
        "cost_params": {"h_annual": h_annual, "b": b, "k": k},
        "methods": {},
    }

    for m in methods:
        cost_col = f"{m}_cost"
        inv_col = f"{m}_avg_inventory"
        ls_col = f"{m}_lost_sales"
        ord_col = f"{m}_orders"
        fr_col = f"{m}_fill_rate"

        if cost_col in eval_df.columns:
            costs = eval_df[cost_col].dropna()
            summary["methods"][m] = {
                "mean_cost": float(costs.mean()),
                "median_cost": float(costs.median()),
                "std_cost": float(costs.std()),
                "p25_cost": float(costs.quantile(0.25)),
                "p75_cost": float(costs.quantile(0.75)),
                "mean_fill_rate": float(eval_df[fr_col].mean()) if fr_col in eval_df else None,
                "mean_avg_inventory": float(eval_df[inv_col].mean()) if inv_col in eval_df else None,
                "mean_lost_sales": float(eval_df[ls_col].mean()) if ls_col in eval_df else None,
                "mean_orders": float(eval_df[ord_col].mean()) if ord_col in eval_df else None,
            }

    # Head-to-head CATB vs LGBM
    if "CATB_cost" in eval_df.columns and "LGBM_cost" in eval_df.columns:
        catb_c = eval_df["CATB_cost"]
        lgbm_c = eval_df["LGBM_cost"]
        lgbm_mean = float(lgbm_c.mean())
        catb_mean = float(catb_c.mean())
        rel_diff = ((lgbm_mean - catb_mean) / lgbm_mean) if lgbm_mean > 0 else 0.0

        n_skus = len(eval_df)
        catb_cheaper = int((catb_c < lgbm_c - 1e-6).sum())
        lgbm_cheaper = int((lgbm_c < catb_c - 1e-6).sum())
        identical = int(n_skus - catb_cheaper - lgbm_cheaper)

        summary["head_to_head_CATB_vs_LGBM"] = {
            "relative_cost_reduction": rel_diff,
            "catb_strictly_cheaper_count": catb_cheaper,
            "catb_strictly_cheaper_prop": catb_cheaper / n_skus,
            "lgbm_strictly_cheaper_count": lgbm_cheaper,
            "lgbm_strictly_cheaper_prop": lgbm_cheaper / n_skus,
            "identical_count": identical,
            "identical_prop": identical / n_skus,
        }

    return summary
