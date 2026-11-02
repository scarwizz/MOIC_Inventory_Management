"""training.py
-----------
Model training module for MOIC surrogate models (LightGBM, CatBoost, XGBoost).
Surrogate models predict simulation metrics (AvgInventory, LostSales, NumOrders)
from SKU demand characteristics and inventory policy parameters.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.features import FEATURE_COLUMNS, engineer_surrogate_features

TARGETS = ["AvgInventory", "LostSales", "NumOrders"]

# Default hyperparameter configurations aligned with project specifications
MODEL_CONFIGS = {
    "LGBM": {
        "params": {
            "objective": "regression_l1",
            "metric": "mae",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbose": -1,
            "n_jobs": -1,
            "random_state": 42,
        }
    },
    "XGBM": {
        "params": {
            "objective": "reg:absoluteerror",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "random_state": 42,
        }
    },
    "CATB": {
        "params": {
            "loss_function": "MAE",
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "subsample": 0.8,
            "bootstrap_type": "Bernoulli",
            "verbose": 0,
            "thread_count": -1,
            "random_seed": 42,
        }
    },
}


def get_model_instance(model_type: str, target: str):
    """Instantiate a regressor for the given model_type and target."""
    model_type = model_type.upper()
    if model_type == "LGBM":
        import lightgbm as lgb
        return lgb.LGBMRegressor(**MODEL_CONFIGS["LGBM"]["params"])
    elif model_type == "XGBM":
        import xgboost as xgb
        return xgb.XGBRegressor(**MODEL_CONFIGS["XGBM"]["params"])
    elif model_type == "CATB":
        import catboost as catb
        return catb.CatBoostRegressor(**MODEL_CONFIGS["CATB"]["params"])
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Expected LGBM, XGBM, or CATB.")


def train_single_surrogate(
    model_type: str,
    target: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[Any, Dict[str, float]]:
    """Train a single surrogate model and return the model + validation metrics."""
    model = get_model_instance(model_type, target)
    model.fit(X_train[FEATURE_COLUMNS], y_train)

    val_preds = model.predict(X_val[FEATURE_COLUMNS])
    mae = float(mean_absolute_error(y_val, val_preds))
    rmse = float(root_mean_squared_error(y_val, val_preds))
    r2 = float(r2_score(y_val, val_preds))

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }
    return model, metrics


def save_surrogate_model(model: Any, model_type: str, target: str, save_dir: Path) -> Path:
    """Save surrogate model to disk following project naming conventions."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    target_map_lgbm = {
        "AvgInventory": "lgbm_avg_inventory.joblib",
        "LostSales": "lgbm_lost_sales.joblib",
        "NumOrders": "lgbm_num_orders.joblib",
    }
    target_map_catb = {
        "AvgInventory": "catb_AvgInventory.joblib",
        "LostSales": "catb_LostSales.joblib",
        "NumOrders": "catb_NumOrders.joblib",
    }
    target_map_xgbm = {
        "AvgInventory": "xgbm_AvgInventory.joblib",
        "LostSales": "xgbm_LostSales.joblib",
        "NumOrders": "xgbm_NumOrders.joblib",
    }

    m_upper = model_type.upper()
    if m_upper == "LGBM":
        filename = target_map_lgbm.get(target, f"lgbm_{target}.joblib")
    elif m_upper == "CATB":
        filename = target_map_catb.get(target, f"catb_{target}.joblib")
    elif m_upper == "XGBM":
        filename = target_map_xgbm.get(target, f"xgbm_{target}.joblib")
    else:
        filename = f"{model_type.lower()}_{target}.joblib"

    out_path = save_dir / filename
    joblib.dump(model, out_path)
    return out_path
