"""evaluation.py
-------------
Fair evaluation pipeline for MOIC inventory policies under the corrected simulator.
Includes warm-up horizon, historical stats estimation, side-by-side comparison
against the Fixed Standard Heuristic (FSH), History-Tuned Baseline (HTB),
and hindsight grid-search oracle across 75 configurations.

Origin-based causality:
- Evaluation origin: history <= d_1550 (calibration d_1186 to d_1550)
- Warm-up period: d_1551 to d_1610 (60 days)
- Evaluation period: d_1611 to d_1638 (28 days)
- Tuning for History-Tuned Baseline strictly backtests on days <= 1550.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

from src.simulator import run_inventory_simulation, calculate_policy_parameters
from src.optimization import compute_inventory_cost, generate_candidate_policies
from src.config import BACKTEST_TUNING_FOLD


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
        "total_satisfied": res["total_satisfied"],
        "total_demand": res["total_demand"],
        "fill_rate": res["fill_rate"],
        "cost": res["total_cost"],
    }


def tune_history_grid_baseline_for_sku(
    demand_series: np.ndarray,
    backtest_fold: Dict[str, Any] = BACKTEST_TUNING_FOLD,
    h_annual: float = 0.01,
    b: float = 0.25,
    k: float = 0.50,
) -> Tuple[int, int, float, Dict[str, float]]:
    """Tune (R, L, TSL) by backtesting on historical data strictly <= history_end_day.

    Calibrates (s, S) strictly on days before that fold's warm-up.
    Scores on its own warm-up and scoring window within days <= 1550.
    Never reads evaluation warm-up (d_1551-d_1610) or evaluation window (d_1611-d_1638).
    """
    calib_start_idx = backtest_fold["calib_start_day"] - 1
    calib_end_idx = backtest_fold["calib_end_day"]
    warmup_start_idx = backtest_fold["warmup_start_day"] - 1
    scoring_end_idx = backtest_fold["scoring_end_day"]
    warmup_days = backtest_fold["warmup_days"]

    calib_demand = demand_series[calib_start_idx:calib_end_idx]
    sim_demand = demand_series[warmup_start_idx:scoring_end_idx]

    d_hat = float(np.mean(calib_demand)) if len(calib_demand) > 0 else 0.0
    sigma_d = float(np.std(calib_demand, ddof=1)) if len(calib_demand) > 1 else 1.0
    if sigma_d <= 0:
        sigma_d = 1.0

    best_cost = float("inf")
    best_policy = (7, 3, 0.95)
    best_metrics: Dict[str, float] = {}

    h_daily = h_annual / 365.0

    for r, l, tsl in generate_candidate_policies():
        clamped_tsl = float(np.clip(tsl, 1e-4, 1.0 - 1e-4))
        z = float(norm.ppf(clamped_tsl))
        ss = max(0.0, z * sigma_d * np.sqrt(r + l))
        s = d_hat * (r + l) + ss
        S = s

        res = run_inventory_simulation(
            demand_series=sim_demand,
            R=r,
            L=l,
            s=s,
            S=S,
            initial_on_hand=S,
            warmup_days=warmup_days,
            h_daily=h_daily,
            b_shortage=b,
            k_order=k,
        )

        if res["total_cost"] < best_cost:
            best_cost = res["total_cost"]
            best_policy = (r, l, tsl)
            best_metrics = res

    return best_policy[0], best_policy[1], best_policy[2], best_metrics


def find_hindsight_oracle_policy(
    demand_series: np.ndarray,
    history_slice: slice,
    warmup_slice: slice,
    eval_slice: slice,
    h_annual: float = 0.01,
    b: float = 0.25,
    k: float = 0.50,
) -> Tuple[int, int, float, Dict[str, float]]:
    """Hindsight grid-search oracle across 75 configurations.
    Evaluates all 75 candidates on the evaluation window with true demand.
    Note: This is a diagnostic benchmark, NOT a theoretical bound or optimum.
    """
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


# Backward compatibility alias
find_true_optimal_policy = find_hindsight_oracle_policy


def compute_paired_bootstrap_ci(
    diff_series: Union[pd.Series, np.ndarray],
    n_bootstraps: int = 10000,
    ci: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, float]:
    """Compute paired bootstrap confidence interval for mean cost differences."""
    diffs = np.asarray(diff_series, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n == 0:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "ci_level": ci}

    rng = np.random.RandomState(random_seed)
    boot_indices = rng.randint(0, n, size=(n_bootstraps, n))
    boot_means = diffs[boot_indices].mean(axis=1)

    alpha = (1.0 - ci) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_lower": float(np.percentile(boot_means, lower_pct)),
        "ci_upper": float(np.percentile(boot_means, upper_pct)),
        "ci_level": ci,
    }


def summarize_evaluation_results(
    eval_df: pd.DataFrame,
    methods: Optional[List[str]] = None,
    h_annual: float = 0.01,
    b: float = 0.25,
    k: float = 0.50,
) -> Dict[str, Any]:
    """Compute aggregate evaluation metrics across all evaluated policies,
    including mean/median costs, aggregate unit fill rate, win rates vs baselines,
    and paired bootstrap confidence intervals.
    """
    if methods is None:
        methods = ["LGBM", "CATB", "XGBM", "HTB", "FSH", "ORACLE", "TO"]

    present_methods = [m for m in methods if f"{m}_cost" in eval_df.columns]
    n_skus = len(eval_df)

    summary: Dict[str, Any] = {
        "n_skus": n_skus,
        "cost_params": {"h_annual": h_annual, "b": b, "k": k, "horizon_days": 28},
        "methods": {},
        "win_rates": {},
        "bootstrap_ci": {},
    }

    for m in present_methods:
        cost_col = f"{m}_cost"
        inv_col = f"{m}_avg_inventory"
        ls_col = f"{m}_lost_sales"
        ord_col = f"{m}_orders"
        fr_col = f"{m}_fill_rate"
        sat_col = f"{m}_satisfied"
        dem_col = f"{m}_demand"

        costs = eval_df[cost_col].dropna()

        # Compute exact aggregate unit fill rate: sum(satisfied) / sum(demand)
        if sat_col in eval_df.columns and dem_col in eval_df.columns:
            tot_sat = float(eval_df[sat_col].sum())
            tot_dem = float(eval_df[dem_col].sum())
            agg_unit_fill_rate = (tot_sat / tot_dem) if tot_dem > 0 else 1.0
        elif fr_col in eval_df.columns:
            agg_unit_fill_rate = float(eval_df[fr_col].mean())
        else:
            agg_unit_fill_rate = None

        summary["methods"][m] = {
            "mean_cost": float(costs.mean()),
            "median_cost": float(costs.median()),
            "std_cost": float(costs.std()),
            "p25_cost": float(costs.quantile(0.25)),
            "p75_cost": float(costs.quantile(0.75)),
            "unit_fill_rate": agg_unit_fill_rate,
            "mean_sku_fill_rate": float(eval_df[fr_col].mean()) if fr_col in eval_df else None,
            "mean_avg_inventory": float(eval_df[inv_col].mean()) if inv_col in eval_df else None,
            "mean_lost_sales": float(eval_df[ls_col].mean()) if ls_col in eval_df else None,
            "mean_orders": float(eval_df[ord_col].mean()) if ord_col in eval_df else None,
        }

    # Win Rates vs Baselines (FSH and HTB)
    for ml in ["CATB", "LGBM"]:
        if f"{ml}_cost" in eval_df.columns:
            ml_cost = eval_df[f"{ml}_cost"]
            if "FSH_cost" in eval_df.columns:
                fsh_cost = eval_df["FSH_cost"]
                beats_fsh = float((ml_cost < fsh_cost - 1e-6).mean() * 100.0)
                summary["win_rates"][f"{ml}_beats_FSH_pct"] = beats_fsh
            if "HTB_cost" in eval_df.columns:
                htb_cost = eval_df["HTB_cost"]
                beats_htb = float((ml_cost < htb_cost - 1e-6).mean() * 100.0)
                summary["win_rates"][f"{ml}_beats_HTB_pct"] = beats_htb

    # Paired Bootstrap Confidence Intervals for Mean Cost Differences
    pairs_to_test = [
        ("CATB", "LGBM"),
        ("CATB", "FSH"),
        ("CATB", "HTB"),
        ("LGBM", "FSH"),
        ("LGBM", "HTB"),
    ]
    for m1, m2 in pairs_to_test:
        c1_col = f"{m1}_cost"
        c2_col = f"{m2}_cost"
        if c1_col in eval_df.columns and c2_col in eval_df.columns:
            diffs = eval_df[c1_col] - eval_df[c2_col]
            ci_res = compute_paired_bootstrap_ci(diffs)
            summary["bootstrap_ci"][f"{m1}_minus_{m2}"] = ci_res

    return summary
