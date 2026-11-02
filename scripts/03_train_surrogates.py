"""scripts/03_train_surrogates.py
------------------------------
Train surrogate ML models (LightGBM, CatBoost, XGBoost) on corrected simulation labels.
Trains 3 targets each: AvgInventory, LostSales, NumOrders (9 models total).
Evaluates MAE, RMSE, and R2 on an 80/20 validation split (random_state=42).

Outputs saved to: models/v2/
"""

import sys
import time
import json
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.features import FEATURE_COLUMNS
from src.training import (
    TARGETS,
    train_single_surrogate,
    save_surrogate_model,
)

# Reference v1 metrics for comparison
V1_METRICS = {
    "LGBM": {
        "AvgInventory": {"mae": 7.67, "rmse": 29.19},
        "LostSales": {"mae": 659.7, "rmse": 2037.3},
        "NumOrders": {"mae": 0.0007, "rmse": 0.0065},
    },
    "CATB": {
        "AvgInventory": {"mae": 9.52, "rmse": 33.15},
        "LostSales": {"mae": 1091.7, "rmse": 2845.2},
        "NumOrders": {"mae": 66.9, "rmse": 98.4},
    },
}


def train_surrogates(
    training_data_path: Path = None,
    models_dir: Path = None,
    model_types: list = ["LGBM", "CATB", "XGBM"],
    verbose: bool = True,
):
    if training_data_path is None:
        training_data_path = project_root / "data" / "processed" / "ml_training_data_v2_corrected.csv"
    if models_dir is None:
        models_dir = project_root / "models" / "v2"
    models_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading training data from: {training_data_path}")
    df = pd.read_csv(training_data_path)
    if verbose:
        print(f"Loaded {len(df):,} observations.")

    # 80/20 train/validation split
    train_df, val_df = train_test_split(df, test_size=0.20, random_state=42)
    if verbose:
        print(f"Train set: {len(train_df):,} | Validation set: {len(val_df):,}")

    results = {}

    # Check which model types can actually run in the environment
    executable_model_types = []
    for m in model_types:
        if m.upper() == "XGBM":
            try:
                import xgboost
                executable_model_types.append(m)
            except ImportError:
                print(f"\n[INFO] xgboost is not installed in the local venv; skipping XGBoost training (per safety rule: no new packages).")
        else:
            executable_model_types.append(m)

    for m_type in executable_model_types:
        if verbose:
            print(f"\n==================================================")
            print(f"Training {m_type} Surrogate Models (3 Targets)")
            print(f"==================================================")
        results[m_type] = {}

        for target in TARGETS:
            if verbose:
                print(f"  Training {m_type} for target: {target}...")
            t0 = time.time()
            model, metrics = train_single_surrogate(
                model_type=m_type,
                target=target,
                X_train=train_df,
                y_train=train_df[target],
                X_val=val_df,
                y_val=val_df[target],
            )
            elapsed = time.time() - t0
            saved_path = save_surrogate_model(model, m_type, target, models_dir)
            metrics["elapsed_seconds"] = elapsed
            metrics["saved_path"] = str(saved_path)
            results[m_type][target] = metrics

            if verbose:
                print(f"    Saved: {saved_path.name}")
                print(f"    Val MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} | R2: {metrics['r2']:.4f} ({elapsed:.1f}s)")

    # Save summary metrics
    metrics_path = models_dir / "v2_validation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"\nSaved validation metrics summary to: {metrics_path}")

    # Comparison Table
    if verbose:
        print("\n==========================================================================")
        print("SURROGATE VALIDATION COMPARISON (v1 vs v2)")
        print("==========================================================================")
        print(f"{'Model':<8} {'Target':<15} {'v1 MAE':<12} {'v2 MAE':<12} {'v2 RMSE':<12} {'v2 R2':<8}")
        print("-" * 74)
        for m_type in results.keys():
            for target in TARGETS:
                v1_mae = V1_METRICS.get(m_type, {}).get(target, {}).get("mae", "N/A")
                v1_str = f"{v1_mae:.4f}" if isinstance(v1_mae, (int, float)) else str(v1_mae)
                v2_m = results[m_type][target]
                print(f"{m_type:<8} {target:<15} {v1_str:<12} {v2_m['mae']:<12.4f} {v2_m['rmse']:<12.4f} {v2_m['r2']:<8.4f}")

    return results


if __name__ == "__main__":
    train_surrogates()
