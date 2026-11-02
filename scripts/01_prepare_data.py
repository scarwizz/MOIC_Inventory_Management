"""scripts/01_prepare_data.py
--------------------------
Load M5 dataset, aggregate daily sales to Product-State level (Level 11: 9,147 series),
and compute time-series demand characteristics (ADI, CV2) under origin-based causality.

Eval features strictly use history_end_day=1550 (never reading warm-up or eval days).
"""

import os
import sys
import shutil
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.features import calculate_ts_features, demand_features
from src.config import EVAL_ORIGIN


def find_data_file(filename: str, project_root: Path) -> Path:
    """Check data/raw, data/processed, and Dataset/ for the file."""
    search_paths = [
        project_root / "data" / "processed" / filename,
        project_root / "data" / "raw" / filename,
        project_root / "Dataset" / filename,
    ]
    for p in search_paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not locate {filename} in {[str(p) for p in search_paths]}")


def prepare_data(
    project_root: Path = None,
    force_recompute: bool = False,
    history_end_day: int = EVAL_ORIGIN["history_end_day"],
):
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    level_11_out = processed_dir / "sales_train_level_11.csv"
    features_out = processed_dir / "m5_demand_features.csv"

    # 1. Level 11 Aggregation
    if not level_11_out.exists():
        try:
            existing_level_11 = find_data_file("sales_train_level_11.csv", project_root)
            print(f"Loading existing Level 11 data from: {existing_level_11}")
            df_level_11 = pd.read_csv(existing_level_11)
            df_level_11.to_csv(level_11_out, index=False)
            print(f"Saved Level 11 data to: {level_11_out}")
        except FileNotFoundError:
            raw_sales_file = find_data_file("sales_train_validation.csv", project_root)
            print(f"Aggregating from raw sales data: {raw_sales_file}...")
            cols_to_read = ["item_id", "state_id"] + [f"d_{i}" for i in range(1, 1914)]
            sales_df = pd.read_csv(raw_sales_file, usecols=lambda x: x in cols_to_read or x.startswith("d_"))
            day_cols = [c for c in sales_df.columns if c.startswith("d_")]
            df_level_11 = sales_df.groupby(["item_id", "state_id"])[day_cols].sum().reset_index()
            df_level_11["id"] = df_level_11["item_id"].astype(str) + "_" + df_level_11["state_id"].astype(str)
            df_level_11.to_csv(level_11_out, index=False)
            print(f"Aggregation complete. Saved {len(df_level_11)} series to: {level_11_out}")
    else:
        print(f"Level 11 data already exists at: {level_11_out}")
        df_level_11 = pd.read_csv(level_11_out)

    # 2. Demand Features under origin-based causality (history_end_day=1550 for eval)
    recompute = force_recompute or not features_out.exists()
    if not recompute:
        try:
            existing_df = pd.read_csv(features_out)
            if "history_end_day" not in existing_df.columns or (existing_df["history_end_day"] != history_end_day).any():
                print(f"Existing features file lacks causal history_end_day={history_end_day}. Recomputing...")
                recompute = True
        except Exception:
            recompute = True

    if recompute:
        print(f"Computing ADI and CV2 features across {len(df_level_11)} series strictly as of history_end_day={history_end_day}...")
        day_cols = [c for c in df_level_11.columns if c.startswith("d_")]
        records = []
        for idx, row in df_level_11.iterrows():
            series_vals = row[day_cols].values.astype(float)
            feat = demand_features(series_vals, history_end_day=history_end_day)
            feat["id"] = row.get("id", f"{row['item_id']}_{row['state_id']}")
            feat["item_id"] = row["item_id"]
            feat["state_id"] = row["state_id"]
            feat["history_end_day"] = history_end_day
            records.append(feat)
        features_df = pd.DataFrame(records)
        cols_order = ["id", "item_id", "state_id", "history_end_day", "ADI", "CV2"]
        features_df = features_df[cols_order]
        features_df.to_csv(features_out, index=False)
        print(f"Causal feature calculation complete. Saved to: {features_out}")

        dataset_features_out = project_root / "Dataset" / "m5_demand_features.csv"
        if dataset_features_out.parent.exists():
            features_df.to_csv(dataset_features_out, index=False)
            print(f"Synced causal features to: {dataset_features_out}")
    else:
        print(f"Causal demand features already exist at: {features_out}")
        dataset_features_out = project_root / "Dataset" / "m5_demand_features.csv"
        if dataset_features_out.parent.exists() and not dataset_features_out.exists():
            shutil.copy(features_out, dataset_features_out)
            print(f"Synced causal features to: {dataset_features_out}")


if __name__ == "__main__":
    prepare_data()
