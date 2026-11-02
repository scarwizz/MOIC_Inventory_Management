"""scripts/01_prepare_data.py
--------------------------
Load M5 dataset, aggregate daily sales to Product-State level (Level 11: 9,147 series),
and compute time-series demand characteristics (ADI, CV2).
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.features import calculate_ts_features


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


def prepare_data(project_root: Path = None):
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    level_11_out = processed_dir / "sales_train_level_11.csv"
    features_out = processed_dir / "m5_demand_features.csv"

    # If processed files already exist in Dataset/, we can ensure they are also in data/processed/
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

    if not features_out.exists():
        try:
            existing_features = find_data_file("m5_demand_features.csv", project_root)
            print(f"Loading existing features from: {existing_features}")
            features_df = pd.read_csv(existing_features)
            features_df.to_csv(features_out, index=False)
            print(f"Saved features to: {features_out}")
        except FileNotFoundError:
            print("Computing ADI and CV2 features across 9,147 series...")
            day_cols = [c for c in df_level_11.columns if c.startswith("d_")]
            records = []
            for idx, row in df_level_11.iterrows():
                series_vals = row[day_cols].values.astype(float)
                feat = calculate_ts_features(series_vals)
                feat["id"] = row.get("id", f"{row['item_id']}_{row['state_id']}")
                feat["item_id"] = row["item_id"]
                feat["state_id"] = row["state_id"]
                records.append(feat)
            features_df = pd.DataFrame(records)
            cols_order = ["id", "item_id", "state_id", "ADI", "CV2"]
            features_df = features_df[cols_order]
            features_df.to_csv(features_out, index=False)
            print(f"Feature calculation complete. Saved to: {features_out}")
    else:
        print(f"Demand features already exist at: {features_out}")


if __name__ == "__main__":
    prepare_data()
