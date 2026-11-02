"""scripts/download_data.py
-------------------------
Download or generate sample dataset for MOIC Inventory Management.
Allows reproduction with either the committed 100-SKU sample dataset
or the full Kaggle M5 competition dataset.
"""

import sys
import shutil
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent


def setup_sample_data():
    """Copy sample sales dataset to data/processed/sales_train_level_11.csv if full dataset absent."""
    sample_path = project_root / "data" / "sample_sales_level_11.csv"
    target_path = project_root / "data" / "processed" / "sales_train_level_11.csv"

    if not target_path.exists():
        if sample_path.exists():
            print(f"Copying sample dataset from {sample_path} to {target_path}...")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(sample_path, target_path)
            print("Sample dataset ready for running quick benchmarks.")
        else:
            print("Sample dataset not found. Download the M5 dataset from Kaggle:")
            print("  kaggle competitions download -c m5-forecasting-accuracy")
    else:
        print(f"Dataset already exists at: {target_path}")


if __name__ == "__main__":
    setup_sample_data()
