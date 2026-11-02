"""scripts/06_export_for_agent.py
------------------------------
Verify and export models and feature pipelines for the interactive Streamlit agent.
Checks that all v2 surrogate model artifacts exist and can be loaded by inventory_tools.py.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import pandas as pd


def verify_agent_artifacts():
    print("Verifying agent artifacts and surrogate models...")
    models_v2_dir = project_root / "models" / "v2"

    expected_models = [
        "lgbm_avg_inventory.joblib",
        "lgbm_lost_sales.joblib",
        "lgbm_num_orders.joblib",
        "catb_AvgInventory.joblib",
        "catb_LostSales.joblib",
        "catb_NumOrders.joblib",
    ]

    missing = []
    for m in expected_models:
        p = models_v2_dir / m
        if not p.exists():
            missing.append(m)
        else:
            try:
                loaded = joblib.load(p)
                print(f"  [OK] {m} ({type(loaded).__name__})")
            except Exception as e:
                print(f"  [ERROR] {m} failed to load: {e}")
                missing.append(m)

    features_csv = project_root / "data" / "processed" / "m5_demand_features.csv"
    if not features_csv.exists():
        features_csv = project_root / "Dataset" / "m5_demand_features.csv"

    if features_csv.exists():
        df_feat = pd.read_csv(features_csv)
        print(f"  [OK] Demand features verified: {len(df_feat):,} SKUs at {features_csv}")
    else:
        print(f"  [ERROR] Demand features file not found!")

    if missing:
        print(f"\nVerification incomplete: {len(missing)} model artifacts missing: {missing}")
        return False
    else:
        print("\nAll agent artifacts verified successfully.")
        return True


if __name__ == "__main__":
    verify_agent_artifacts()
