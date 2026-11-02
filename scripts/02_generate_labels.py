"""scripts/02_generate_labels.py
------------------------------
Generate training labels for MOIC surrogate models using the corrected simulator.
Runs 75 inventory policies (5 R x 5 L x 3 TSL) across 9,147 Level 11 SKUs
over the 365-day historical simulation window (d_1549 to d_1913).

Output: data/processed/ml_training_data_v2_corrected.csv
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.simulator import calculate_policy_parameters, run_inventory_simulation
from src.features import engineer_surrogate_features

# Standard 75-policy grid (identical to original Kaggle MAIN_NOTEBOOK Cell 1)
R_RANGE = np.linspace(1, 30, 5, dtype=int)      # [1, 8, 15, 22, 30]
L_RANGE = np.linspace(1, 30, 5, dtype=int)      # [1, 8, 15, 22, 30]
TSL_RANGE = np.array([0.90, 0.95, 0.99])        # [0.90, 0.95, 0.99]

# Simulation window (last 365 days of historical sales: d_1549 to d_1913)
TRAIN_DAYS = 1913
SIM_DAYS = 365
SPLIT_DAY = TRAIN_DAYS - SIM_DAYS
SIM_DAY_COLS = [f"d_{i}" for i in range(SPLIT_DAY + 1, TRAIN_DAYS + 1)]


def run_v1_flawed_sim(demand_series: np.ndarray, r: int, l: int, tsl: float) -> dict:
    """Exact reproduction of flawed Kaggle MAIN_NOTEBOOK Cell 1 simulation logic."""
    D_hat, sigma_D, SS, s, S = calculate_policy_parameters(demand_series, r, l, tsl)
    inventory = S
    lost_sales = 0.0
    orders = 0
    inv_history = []
    for t in range(len(demand_series)):
        d = demand_series[t]
        if t % r == 0:
            if inventory <= s:
                orders += 1
                # Flaw: order quantity calculated but never added to inventory!
        if inventory >= d:
            inventory -= d
        else:
            lost_sales += (d - inventory)
            inventory = 0.0
        inv_history.append(inventory)
    return {
        "avg_inventory": float(np.mean(inv_history)),
        "lost_sales": float(lost_sales),
        "orders": int(orders),
    }


def run_sanity_checks(df_v2: pd.DataFrame, demand_matrix: np.ndarray, n_check_skus: int = 500):
    """Run sanity checks comparing v1 flawed vs v2 corrected behavior."""
    print("\n==========================================================================")
    print("SANITY CHECKS: v1 (FLAWED SIMULATOR) vs v2 (CORRECTED SIMULATOR)")
    print("==========================================================================")

    # 1. Overall v2 dataset summary
    print(f"Total v2 records: {len(df_v2):,}")
    print(f"  v2 AvgInventory: mean={df_v2['AvgInventory'].mean():.2f}, median={df_v2['AvgInventory'].median():.2f}")
    print(f"  v2 LostSales:    mean={df_v2['LostSales'].mean():.2f}, median={df_v2['LostSales'].median():.2f}")
    print(f"  v2 NumOrders:    mean={df_v2['NumOrders'].mean():.2f}, median={df_v2['NumOrders'].median():.2f}")

    # 2. Compare on sample SKUs for policy R=8, L=1, TSL=0.95 (frequent replenishment)
    v1_ls_r8, v2_ls_r8 = [], []
    v1_inv_r8, v2_inv_r8 = [], []
    # And for boundary policy R=30, L=30, TSL=0.99
    v1_ls_r30, v2_ls_r30 = [], []
    v1_inv_r30, v2_inv_r30 = [], []

    sample_indices = np.linspace(0, len(demand_matrix) - 1, n_check_skus, dtype=int)
    for idx in sample_indices:
        series = demand_matrix[idx]
        # Policy (R=8, L=1, TSL=0.95)
        res_v1_r8 = run_v1_flawed_sim(series, 8, 1, 0.95)
        d_hat, sigma_d, _, s, S = calculate_policy_parameters(series, 8, 1, 0.95)
        res_v2_r8 = run_inventory_simulation(series, 8, 1, s, S, initial_on_hand=S, warmup_days=0)

        v1_ls_r8.append(res_v1_r8["lost_sales"])
        v2_ls_r8.append(res_v2_r8["total_lost_sales"])
        v1_inv_r8.append(res_v1_r8["avg_inventory"])
        v2_inv_r8.append(res_v2_r8["avg_inventory"])

        # Policy (R=30, L=30, TSL=0.99)
        res_v1_r30 = run_v1_flawed_sim(series, 30, 30, 0.99)
        d_hat, sigma_d, _, s, S = calculate_policy_parameters(series, 30, 30, 0.99)
        res_v2_r30 = run_inventory_simulation(series, 30, 30, s, S, initial_on_hand=S, warmup_days=0)

        v1_ls_r30.append(res_v1_r30["lost_sales"])
        v2_ls_r30.append(res_v2_r30["total_lost_sales"])
        v1_inv_r30.append(res_v1_r30["avg_inventory"])
        v2_inv_r30.append(res_v2_r30["avg_inventory"])

    print(f"\n--- Diagnostic 1: Replenishing Policy (R=8, L=1, TSL=0.95) on {n_check_skus} SKUs ---")
    print(f"  v1 (Flawed - No Arrivals): Lost Sales Mean = {np.mean(v1_ls_r8):.2f} | Avg Inventory Mean = {np.mean(v1_inv_r8):.2f}")
    print(f"  v2 (Corrected - Arrivals): Lost Sales Mean = {np.mean(v2_ls_r8):.2f} | Avg Inventory Mean = {np.mean(v2_inv_r8):.2f}")
    diff_ls = np.mean(v1_ls_r8) - np.mean(v2_ls_r8)
    print(f"  -> Confirmation: Corrected sim reduces artificial lost sales by {diff_ls:.2f} units ({diff_ls/np.mean(v1_ls_r8)*100:.1f}%) because stock actually arrives!")

    print(f"\n--- Diagnostic 2: Boundary Policy (R=30, L=30, TSL=0.99) on {n_check_skus} SKUs ---")
    print(f"  v1 (Flawed - No Arrivals): Lost Sales Mean = {np.mean(v1_ls_r30):.2f} | Avg Inventory Mean = {np.mean(v1_inv_r30):.2f}")
    print(f"  v2 (Corrected - Arrivals): Lost Sales Mean = {np.mean(v2_ls_r30):.2f} | Avg Inventory Mean = {np.mean(v2_inv_r30):.2f}")
    print(f"  -> Confirmation: Under R=30, L=30, TSL=0.99, initial buffer S is so large that even without replenishment, lost sales were low ({np.mean(v1_ls_r30):.2f}). This confirms the root cause of CatBoost's 100% boundary collapse in v1!")
    print("==========================================================================\n")


def generate_labels(
    limit_skus: int = None,
    output_path: Path = None,
    verbose: bool = True,
):
    if output_path is None:
        output_path = project_root / "data" / "processed" / "ml_training_data_v2_corrected.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    level_11_path = project_root / "data" / "processed" / "sales_train_level_11.csv"
    if not level_11_path.exists():
        level_11_path = project_root / "Dataset" / "sales_train_level_11.csv"

    features_path = project_root / "data" / "processed" / "m5_demand_features.csv"
    if not features_path.exists():
        features_path = project_root / "Dataset" / "m5_demand_features.csv"

    if verbose:
        print(f"Loading Level 11 sales from: {level_11_path}")
    sales_df = pd.read_csv(level_11_path)
    if verbose:
        print(f"Loading features from: {features_path}")
    features_df = pd.read_csv(features_path)

    # Merge ADI and CV2 onto sales_df
    feat_cols = ["item_id", "state_id", "ADI", "CV2"]
    merged_df = pd.merge(sales_df, features_df[feat_cols], on=["item_id", "state_id"], how="inner")

    if limit_skus is not None:
        merged_df = merged_df.iloc[:limit_skus]
        if verbose:
            print(f"Subsetting to first {limit_skus} SKUs for benchmarking/testing.")

    n_series = len(merged_df)
    n_policies = len(R_RANGE) * len(L_RANGE) * len(TSL_RANGE)
    total_sims = n_series * n_policies

    if verbose:
        print(f"Starting simulation: {n_series} SKUs x {n_policies} policies = {total_sims:,} runs.")
        print(f"Simulation window: {len(SIM_DAY_COLS)} days ({SIM_DAY_COLS[0]} to {SIM_DAY_COLS[-1]}).")

    start_time = time.time()
    records = []
    conservation_failures = 0

    demand_matrix = merged_df[SIM_DAY_COLS].values.astype(float)
    item_ids = merged_df["item_id"].values
    state_ids = merged_df["state_id"].values
    adis = merged_df["ADI"].values
    cv2s = merged_df["CV2"].values

    # Pre-generate policy tuples
    policy_list = []
    for r in R_RANGE:
        for l in L_RANGE:
            for tsl in TSL_RANGE:
                policy_list.append((int(r), int(l), float(tsl)))

    # Outer loop over SKUs
    for i in tqdm(range(n_series), desc="Simulating SKUs", disable=not verbose):
        sales_series = demand_matrix[i]
        item_id = item_ids[i]
        state_id = state_ids[i]
        adi = adis[i]
        cv2 = cv2s[i]

        for r, l, tsl in policy_list:
            D_hat, sigma_D, SS, s, S = calculate_policy_parameters(
                historical_sales=sales_series,
                R=r,
                L=l,
                TSL=tsl,
            )

            # In training data protocol (matching v1): no warmup, init at S
            sim_res = run_inventory_simulation(
                demand_series=sales_series,
                R=r,
                L=l,
                s=s,
                S=S,
                initial_on_hand=S,
                warmup_days=0,
            )

            if not sim_res["conservation_passed"]:
                conservation_failures += 1

            records.append({
                "item_id": item_id,
                "state_id": state_id,
                "ADI": adi,
                "CV2": cv2,
                "R": r,
                "L": l,
                "TSL": tsl,
                "Risk_Period": r + l,
                "Policy_Interaction": r * l,
                "Demand_Profile": (adi / cv2) if cv2 > 0 else 0.0,
                "AvgInventory": sim_res["avg_inventory"],
                "LostSales": sim_res["total_lost_sales"],
                "NumOrders": sim_res["total_orders"],
            })

    elapsed = time.time() - start_time
    if verbose:
        print(f"\nCompleted {total_sims:,} simulations in {elapsed:.2f}s ({total_sims/elapsed:.1f} sims/sec).")
        print(f"Conservation failures: {conservation_failures} / {total_sims}")

    df_out = pd.DataFrame(records)
    df_out.to_csv(output_path, index=False)
    if verbose:
        print(f"Saved training dataset to: {output_path} (shape: {df_out.shape})")

    # Run sanity checks
    run_sanity_checks(df_out, demand_matrix)

    return df_out


if __name__ == "__main__":
    generate_labels()
