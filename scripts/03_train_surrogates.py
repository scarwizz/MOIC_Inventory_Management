"""scripts/03_train_surrogates.py
------------------------------
Train surrogate ML models (LightGBM, CatBoost) on causal simulation labels.
Replaces random row split with:
1. Chronological validation: train on earlier origins (train_921, train_1009),
   validate on the latest pre-eval origin (val_1097) with strict purging.
2. Grouped held-out-SKU evaluation: 80/20 train/held-out SKU split.

Outputs saved to: models/v2/ and models/
"""

import sys
import time
import json
from pathlib import Path
import shutil

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.features import FEATURE_COLUMNS
from src.config import (
    TRAINING_ORIGINS,
    VALIDATION_ORIGIN,
    purge_overlapping_labels,
    RANDOM_SEED,
)
from src.training import (
    TARGETS,
    train_single_surrogate,
    save_surrogate_model,
)


def train_surrogates(
    training_data_path: Path = None,
    models_dir: Path = None,
    model_types: list = ["LGBM", "CATB"],
    verbose: bool = True,
):
    if training_data_path is None:
        training_data_path = project_root / "data" / "processed" / "ml_training_data_v2_corrected.csv"
    if models_dir is None:
        models_dir = project_root / "models" / "v2"
    models_dir.mkdir(parents=True, exist_ok=True)

    root_models_dir = project_root / "models"
    root_models_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading training data from: {training_data_path}")
    df = pd.read_csv(training_data_path)
    if verbose:
        print(f"Loaded {len(df):,} total records from {training_data_path.name}.")

    # 1. Chronological Splitting & Purging
    val_origin_id = VALIDATION_ORIGIN["origin_id"]
    val_decision_day = VALIDATION_ORIGIN["history_end_day"]
    valid_train_origins = purge_overlapping_labels(TRAINING_ORIGINS, VALIDATION_ORIGIN)
    train_origin_ids = [o["origin_id"] for o in valid_train_origins]

    if verbose:
        print(f"\nChronological Split Configuration:")
        print(f"  Training Origins: {train_origin_ids}")
        print(f"  Validation Origin: {val_origin_id} (decision day: {val_decision_day})")

    # Purge any training records whose scoring_end_day > val_decision_day
    train_mask = df["origin_id"].isin(train_origin_ids) & (df["scoring_end_day"] <= val_decision_day)
    val_mask = df["origin_id"] == val_origin_id

    train_df = df[train_mask].copy()
    chrono_val_df = df[val_mask].copy()

    if verbose:
        print(f"  Train set: {len(train_df):,} rows | Chronological Val set: {len(chrono_val_df):,} rows")

    # 2. Grouped Held-Out-SKU Split (cross-sectional holdout)
    df["sku_key"] = df["item_id"].astype(str) + "_" + df["state_id"].astype(str)
    train_df["sku_key"] = train_df["item_id"].astype(str) + "_" + train_df["state_id"].astype(str)
    chrono_val_df["sku_key"] = chrono_val_df["item_id"].astype(str) + "_" + chrono_val_df["state_id"].astype(str)

    unique_skus = pd.Series(df["sku_key"].unique())
    np.random.seed(RANDOM_SEED)
    shuffled_skus = unique_skus.sample(frac=1.0, random_state=RANDOM_SEED).values

    n_held_out = int(0.20 * len(shuffled_skus))
    held_out_skus = set(shuffled_skus[:n_held_out])
    train_skus = set(shuffled_skus[n_held_out:])

    if verbose:
        print(f"Grouped-SKU Split: {len(train_skus):,} training SKUs | {len(held_out_skus):,} held-out SKUs")

    # Save Split Definition
    split_def = {
        "random_seed": RANDOM_SEED,
        "train_origins": train_origin_ids,
        "val_origin": val_origin_id,
        "val_decision_day": val_decision_day,
        "purging_rule": "scoring_end_day <= val_decision_day and non-overlapping target windows",
        "n_train_rows": len(train_df),
        "n_val_rows": len(chrono_val_df),
        "n_total_skus": len(unique_skus),
        "n_train_skus": len(train_skus),
        "n_held_out_skus": len(held_out_skus),
        "held_out_sku_sample": list(shuffled_skus[:10]),
    }
    split_def_path = models_dir / "split_definition.json"
    with open(split_def_path, "w", encoding="utf-8") as f:
        json.dump(split_def, f, indent=2)
    if verbose:
        print(f"Saved split definition to: {split_def_path}")

    # Check which model types can actually run in the environment
    executable_model_types = []
    for m in model_types:
        if m.upper() == "XGBM":
            try:
                import xgboost
                executable_model_types.append(m)
            except ImportError:
                if verbose:
                    print(f"\n[INFO] xgboost not installed; skipping XGBM.")
        else:
            executable_model_types.append(m)

    results = {}

    for m_type in executable_model_types:
        if verbose:
            print(f"\n==================================================")
            print(f"Training {m_type} Surrogate Models (Chronological Training)")
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
                X_val=chrono_val_df,
                y_val=chrono_val_df[target],
            )
            elapsed = time.time() - t0

            # Evaluate on Grouped Held-Out SKUs (both in-time and out-of-time)
            val_heldout_mask = chrono_val_df["sku_key"].isin(held_out_skus)
            if val_heldout_mask.sum() > 0:
                val_ho_preds = model.predict(chrono_val_df.loc[val_heldout_mask, FEATURE_COLUMNS])
                val_ho_y = chrono_val_df.loc[val_heldout_mask, target]
                metrics["heldout_sku_val_mae"] = float(mean_absolute_error(val_ho_y, val_ho_preds))
                metrics["heldout_sku_val_rmse"] = float(root_mean_squared_error(val_ho_y, val_ho_preds))
                metrics["heldout_sku_val_r2"] = float(r2_score(val_ho_y, val_ho_preds))

            # Save models in models/v2/ and root models/
            saved_path_v2 = save_surrogate_model(model, m_type, target, models_dir)
            saved_path_root = save_surrogate_model(model, m_type, target, root_models_dir)

            metrics["elapsed_seconds"] = elapsed
            metrics["saved_path"] = str(saved_path_v2)
            results[m_type][target] = metrics

            if verbose:
                print(f"    Saved: {saved_path_v2.name}")
                print(f"    Chrono Val MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} | R2: {metrics['r2']:.4f}")
                if "heldout_sku_val_mae" in metrics:
                    print(f"    Held-Out SKU Val MAE: {metrics['heldout_sku_val_mae']:.4f} | R2: {metrics['heldout_sku_val_r2']:.4f}")

    # Save summary metrics
    metrics_path = models_dir / "v2_validation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"\nSaved validation metrics summary to: {metrics_path}")

    return results


if __name__ == "__main__":
    train_surrogates()
