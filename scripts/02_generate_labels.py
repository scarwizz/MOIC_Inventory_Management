"""scripts/02_generate_labels.py
------------------------------
Generate training labels for MOIC surrogate models using the corrected simulator
under strict origin-based causality.

Rebuilds labels across causal decision origins per SKU:
- Training origins: train_921 (history <= 921), train_1009 (history <= 1009)
- Chronological validation origin: val_1097 (history <= 1097, scoring ends at d_1185)
All training and validation scoring windows end by d_1185, guaranteeing ZERO leakage
into the evaluation period (calibration d_1186-d_1550, warm-up d_1551-d_1610, scoring d_1611-d_1638).

For each origin:
- Calibration slice (365 days ending at history_end_day) fits (s, S) parameters before warm-up.
- Warm-up slice (60 days) eliminates initial condition bias.
- Scoring slice (28 days) matches the exact evaluation horizon.
- Features are computed strictly via demand_features(series, history_end_day).

Output: data/processed/ml_training_data_v2_corrected.csv
"""

import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

from src.simulator import run_inventory_simulation
from src.features import demand_features
from src.config import (
    ALL_EXPERIMENT_ORIGINS,
    R_GRID,
    L_GRID,
    TSL_GRID,
    H_DAILY,
    B_SHORTAGE,
    K_ORDER,
    WARMUP_DAYS,
    SCORING_HORIZON_DAYS,
)


def precalculate_policy_z_scores(tsl_grid: List[float]) -> Dict[float, float]:
    """Pre-calculate clamped Z-scores for each TSL level."""
    z_map = {}
    for tsl in tsl_grid:
        clamped = float(np.clip(tsl, 1e-4, 1.0 - 1e-4))
        z_map[tsl] = float(norm.ppf(clamped))
    return z_map


def generate_labels(
    limit_skus: Optional[int] = None,
    output_path: Optional[Path] = None,
    origins: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    if output_path is None:
        output_path = project_root / "data" / "processed" / "ml_training_data_v2_corrected.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if origins is None:
        origins = ALL_EXPERIMENT_ORIGINS

    # 1. Load Sales Data
    level_11_path = project_root / "data" / "processed" / "sales_train_level_11.csv"
    if not level_11_path.exists():
        level_11_path = project_root / "Dataset" / "sales_train_level_11.csv"

    if verbose:
        print(f"Loading Level 11 sales from: {level_11_path}")
    sales_df = pd.read_csv(level_11_path)

    if limit_skus is not None:
        sales_df = sales_df.iloc[:limit_skus]
        if verbose:
            print(f"Subsetting to first {limit_skus} SKUs for benchmarking/testing.")

    n_series = len(sales_df)
    day_cols = [f"d_{i}" for i in range(1, 1914)]
    demand_matrix = sales_df[day_cols].values.astype(float)
    item_ids = sales_df["item_id"].values
    state_ids = sales_df["state_id"].values

    # Pre-generate policy combinations
    policy_list = []
    for r in R_GRID:
        for l in L_GRID:
            for tsl in TSL_GRID:
                policy_list.append((int(r), int(l), float(tsl)))
    n_policies = len(policy_list)

    z_scores = precalculate_policy_z_scores(TSL_GRID)

    total_sims = n_series * n_policies * len(origins)
    if verbose:
        print(f"Starting simulation across {len(origins)} causal origins:")
        for orig in origins:
            print(f"  Origin '{orig['origin_id']}': calib d_{orig['calib_start_day']}-d_{orig['calib_end_day']} | "
                  f"warmup d_{orig['warmup_start_day']}-d_{orig['warmup_end_day']} | "
                  f"scoring d_{orig['scoring_start_day']}-d_{orig['scoring_end_day']}")
        print(f"Total simulations: {n_series} SKUs x {n_policies} policies x {len(origins)} origins = {total_sims:,} runs.\n")

    start_time = time.time()
    records = []
    conservation_failures = 0

    for orig in origins:
        orig_id = orig["origin_id"]
        h_end = orig["history_end_day"]
        calib_start_idx = orig["calib_start_day"] - 1
        calib_end_idx = orig["calib_end_day"]  # slice end is exclusive
        warmup_start_idx = orig["warmup_start_day"] - 1
        scoring_end_idx = orig["scoring_end_day"]
        warmup_len = orig["warmup_days"]

        desc = f"Simulating Origin {orig_id} (h_end={h_end})"
        for i in tqdm(range(n_series), desc=desc, disable=not verbose):
            full_series = demand_matrix[i]
            item_id = item_ids[i]
            state_id = state_ids[i]

            # 1. Causal demand features strictly as of history_end_day
            feat = demand_features(full_series, history_end_day=h_end)
            adi = feat["ADI"]
            cv2 = feat["CV2"]
            demand_profile = (adi / cv2) if cv2 > 0 else 0.0

            # 2. Calibration history strictly before warm-up
            calib_demand = full_series[calib_start_idx:calib_end_idx]
            d_hat = float(np.mean(calib_demand)) if len(calib_demand) > 0 else 0.0
            sigma_d = float(np.std(calib_demand, ddof=1)) if len(calib_demand) > 1 else 1.0
            if sigma_d <= 0:
                sigma_d = 1.0

            # 3. Target demand: warm-up + scoring
            sim_demand = full_series[warmup_start_idx:scoring_end_idx]

            # 4. Evaluate each policy
            for r, l, tsl in policy_list:
                z = z_scores[tsl]
                ss = max(0.0, z * sigma_d * np.sqrt(r + l))
                lead_demand = d_hat * (r + l)
                s = lead_demand + ss
                S = lead_demand + ss

                sim_res = run_inventory_simulation(
                    demand_series=sim_demand,
                    R=r,
                    L=l,
                    s=s,
                    S=S,
                    initial_on_hand=S,
                    warmup_days=warmup_len,
                    h_daily=H_DAILY,
                    b_shortage=B_SHORTAGE,
                    k_order=K_ORDER,
                )

                if not sim_res["conservation_passed"]:
                    conservation_failures += 1

                records.append({
                    "origin_id": orig_id,
                    "history_end_day": h_end,
                    "scoring_end_day": orig["scoring_end_day"],
                    "item_id": item_id,
                    "state_id": state_id,
                    "ADI": adi,
                    "CV2": cv2,
                    "R": r,
                    "L": l,
                    "TSL": tsl,
                    "Risk_Period": r + l,
                    "Policy_Interaction": r * l,
                    "Demand_Profile": demand_profile,
                    "AvgInventory": sim_res["avg_inventory"],
                    "LostSales": sim_res["total_lost_sales"],
                    "NumOrders": sim_res["total_orders"],
                    "FillRate": sim_res["fill_rate"],
                    "Cost": sim_res["total_cost"],
                })

    elapsed = time.time() - start_time
    if verbose:
        print(f"\nCompleted {total_sims:,} simulations in {elapsed:.2f}s ({total_sims/elapsed:.1f} sims/sec).")
        print(f"Conservation failures: {conservation_failures} / {total_sims}")

    df_out = pd.DataFrame(records)
    df_out.to_csv(output_path, index=False)
    if verbose:
        print(f"Saved training dataset to: {output_path} (shape: {df_out.shape})")
        print("\nDataset Summary by Origin:")
        print(df_out.groupby("origin_id")[["AvgInventory", "LostSales", "NumOrders", "Cost"]].mean())

    return df_out


if __name__ == "__main__":
    generate_labels()
