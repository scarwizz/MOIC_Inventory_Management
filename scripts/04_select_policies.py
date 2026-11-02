"""scripts/04_select_policies.py
------------------------------
Select optimal inventory policies (R*, L*, TSL*) for all 9,147 SKUs
using trained v2 surrogate models.
Evaluates the 75-policy grid and minimizes predicted total cost:
C = (h/365) * AvgInv * 365 + b * LostSales + k * NumOrders (h=0.01, b=0.25, k=0.50)

Outputs saved to: data/processed/optimal_policies_v2_{MODEL}.csv
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import pandas as pd
import numpy as np

from src.optimization import optimize_all_skus, generate_candidate_policies

MODEL_TYPES = ["LGBM", "CATB", "XGBM"]


def load_surrogate_models_for_type(model_type: str, models_dir: Path) -> dict:
    """Load the 3 surrogate models for a given model type."""
    target_map_lgbm = {
        "avg_inventory": "lgbm_avg_inventory.joblib",
        "lost_sales": "lgbm_lost_sales.joblib",
        "num_orders": "lgbm_num_orders.joblib",
    }
    target_map_catb = {
        "avg_inventory": "catb_AvgInventory.joblib",
        "lost_sales": "catb_LostSales.joblib",
        "num_orders": "catb_NumOrders.joblib",
    }
    target_map_xgbm = {
        "avg_inventory": "xgbm_AvgInventory.joblib",
        "lost_sales": "xgbm_LostSales.joblib",
        "num_orders": "xgbm_NumOrders.joblib",
    }

    m_upper = model_type.upper()
    if m_upper == "LGBM":
        m_map = target_map_lgbm
    elif m_upper == "CATB":
        m_map = target_map_catb
    elif m_upper == "XGBM":
        m_map = target_map_xgbm
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    models = {}
    for key, fname in m_map.items():
        fpath = models_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Model file not found: {fpath}")
        models[key] = joblib.load(fpath)

    return models


def select_policies(
    features_path: Path = None,
    models_dir: Path = None,
    output_dir: Path = None,
    model_types: list = MODEL_TYPES,
    verbose: bool = True,
):
    if features_path is None:
        features_path = project_root / "data" / "processed" / "m5_demand_features.csv"
        if not features_path.exists():
            features_path = project_root / "Dataset" / "m5_demand_features.csv"
    if models_dir is None:
        models_dir = project_root / "models" / "v2"
    if output_dir is None:
        output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading SKU demand features from: {features_path}")
    features_df = pd.read_csv(features_path)
    n_skus = len(features_df)
    if verbose:
        print(f"Loaded {n_skus:,} SKUs.")

    all_optimal = {}

    for m_type in model_types:
        try:
            models = load_surrogate_models_for_type(m_type, models_dir)
        except FileNotFoundError as e:
            print(f"\n[INFO] Skipping {m_type}: model files not found ({e}).")
            continue

        if verbose:
            print(f"\n==================================================")
            print(f"Optimizing policies using {m_type} surrogates...")
            print(f"==================================================")

        t0 = time.time()
        opt_df = optimize_all_skus(
            models=models,
            sku_features_df=features_df,
            h_annual=0.01,
            b=0.25,
            k=0.50,
            horizon_days=365,
        )
        elapsed = time.time() - t0

        out_csv = output_dir / f"optimal_policies_v2_{m_type}.csv"
        opt_df.to_csv(out_csv, index=False)
        all_optimal[m_type] = opt_df

        if verbose:
            print(f"Optimization completed in {elapsed:.1f}s.")
            print(f"Saved optimal policies to: {out_csv}")
            print("\nParameter Distribution Summary:")
            print(f"  R distribution:\n{opt_df['R'].value_counts(normalize=True).sort_index().to_dict()}")
            print(f"  L distribution:\n{opt_df['L'].value_counts(normalize=True).sort_index().to_dict()}")
            print(f"  TSL distribution:\n{opt_df['TSL'].value_counts(normalize=True).sort_index().to_dict()}")
            boundary_collapse_count = ((opt_df["R"] == 30) & (opt_df["L"] == 30) & (opt_df["TSL"] == 0.99)).sum()
            print(f"  Boundary collapse count (R=30, L=30, TSL=0.99): {boundary_collapse_count} / {n_skus} ({boundary_collapse_count/n_skus*100:.1f}%)")

    return all_optimal


if __name__ == "__main__":
    select_policies()
