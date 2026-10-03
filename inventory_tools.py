"""
inventory_tools.py
------------------
LangChain tools for the Neuro-Symbolic Supply Chain Agent.
- Pydantic V2 compliant (uses @field_validator + ValidationInfo)
- Bare @tool decorators (description lives in the docstring)
- Graceful fallback if model files are missing
"""

import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, ValidationInfo

# ---------------------------------------------------------------------------
# Optional LangChain import (no-op stub if not installed)
# ---------------------------------------------------------------------------
try:
    from langchain_core.tools import tool
except ImportError:
    def tool(fn):  # type: ignore[misc]
        return fn

# ---------------------------------------------------------------------------
# 1.  Pydantic V2 input schema
# ---------------------------------------------------------------------------

class PolicySimulationInput(BaseModel):
    """Strict typed input for a single inventory-policy simulation.

    Fields
    ------
    adi : float
        Average Inter-Demand Interval (demand intermittency).
    cv2 : float
        Squared Coefficient of Variation of demand sizes.
    r : int
        Review period in days (>= 1).
    l : int
        Lead time in days (>= 0).
    tsl : float
        Target Service Level expressed as a probability [0.0, 1.0].
    """

    adi: float = Field(..., description="Average Inter-Demand Interval")
    cv2: float = Field(..., description="Squared Coefficient of Variation")
    r: int   = Field(..., ge=1, description="Review period (days)")
    l: int   = Field(..., ge=0, description="Lead time (days)")
    tsl: float = Field(..., ge=0.0, le=1.0, description="Target Service Level [0, 1]")

    @field_validator("adi", "cv2")
    @classmethod
    def must_be_non_negative(cls, v: float, info: ValidationInfo) -> float:
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative, got {v}")
        return v


# ---------------------------------------------------------------------------
# 2.  Model loading with silent fallback
# ---------------------------------------------------------------------------

_MODEL_DIR = Path(__file__).parent / "models"
_LGBM_PATHS = {
    "avg_inventory": _MODEL_DIR / "lgbm_avg_inventory.joblib",
    "lost_sales":    _MODEL_DIR / "lgbm_lost_sales.joblib",
    "num_orders":    _MODEL_DIR / "lgbm_num_orders.joblib",
}
_CATB_PATHS = {
    "avg_inventory": _MODEL_DIR / "catb_AvgInventory.joblib",
    "lost_sales":    _MODEL_DIR / "catb_LostSales.joblib",
    "num_orders":    _MODEL_DIR / "catb_NumOrders.joblib",
}


class _ZeroPredictor:
    """Fallback predictor that always returns 0.0 (used when model files are absent)."""
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0], dtype=float)


def _load_model(path: Path):
    """Load a joblib model from *path*; return a _ZeroPredictor on any failure."""
    try:
        import joblib
        model = joblib.load(path)
        if not hasattr(model, "predict"):
            raise AttributeError("Loaded object has no predict() method.")
        return model
    except Exception as exc:
        print(f"[inventory_tools] Warning: could not load {path.name} ({exc}). Using zero predictor.")
        return _ZeroPredictor()


def load_moic_models() -> Dict[str, Dict[str, Any]]:
    """Load both LightGBM and CatBoost MOIC surrogate models.

    Falls back to a zero-predicting stub when model files are not present.
    """
    return {
        "lightgbm": {key: _load_model(path) for key, path in _LGBM_PATHS.items()},
        "catboost": {key: _load_model(path) for key, path in _CATB_PATHS.items()},
    }


# Module-level cache — pays the I/O cost once per process.
_MODELS_CACHE: Dict[str, Dict[str, Any]] = load_moic_models()


# ---------------------------------------------------------------------------
# 3.  LangChain tools
# ---------------------------------------------------------------------------

@tool
def simulate_policy_outcomes(
    adi: float,
    cv2: float,
    r: int,
    l: int,
    tsl: float,
    model_framework: str = "lightgbm",
) -> Dict[str, float]:
    """Predict inventory outcomes for a given policy using the pre-trained MOIC surrogate models.

    You MUST call get_sku_profile first to obtain the real adi and cv2 values for the SKU
    before calling this tool. Never guess or assume adi/cv2 values.

    Parameters
    ----------
    adi : float
        Average Inter-Demand Interval retrieved from get_sku_profile.
    cv2 : float
        Squared Coefficient of Variation retrieved from get_sku_profile.
    r : int
        Review period in days (user-supplied).
    l : int
        Lead time in days (user-supplied).
    tsl : float
        Target Service Level between 0.0 and 1.0 (user-supplied).
    model_framework : str, optional
        Surrogate engine: 'lightgbm' (fastest), 'catboost' (cost-optimized), or 'ensemble' (average). Defaults to 'lightgbm'.

    Returns
    -------
    dict
        Keys: expected_avg_inventory, expected_lost_sales, expected_num_orders, model_used.
    """
    # Validate via Pydantic before running inference
    params = PolicySimulationInput(adi=adi, cv2=cv2, r=r, l=l, tsl=tsl)

    # --- Engineered features (must match Kaggle training schema exactly) ---
    # 1. Risk_Period  = R + L  (combined replenishment exposure)
    risk_period = params.r + params.l
    # 2. Policy_Interaction = R * L  (interaction term)
    policy_interaction = params.r * params.l
    # 3. Demand_Profile = ADI / CV2  (demand regularity ratio; 0 when CV2 == 0)
    demand_profile = (params.adi / params.cv2) if params.cv2 != 0 else 0.0

    features = pd.DataFrame(
        [[
            params.adi, params.cv2, params.r, params.l, params.tsl,
            risk_period, policy_interaction, demand_profile,
        ]],
        columns=["ADI", "CV2", "R", "L", "TSL",
                 "Risk_Period", "Policy_Interaction", "Demand_Profile"],
    )

    framework = model_framework.lower().strip()
    lgbm_models = _MODELS_CACHE["lightgbm"]
    catb_models = _MODELS_CACHE["catboost"]

    if framework == "catboost":
        avg_inv = float(catb_models["avg_inventory"].predict(features)[0])
        lost_sales = float(catb_models["lost_sales"].predict(features)[0])
        num_orders = float(catb_models["num_orders"].predict(features)[0])
        engine_used = "CatBoost"
    elif framework == "ensemble":
        l_inv = float(lgbm_models["avg_inventory"].predict(features)[0])
        l_ls = float(lgbm_models["lost_sales"].predict(features)[0])
        l_ord = float(lgbm_models["num_orders"].predict(features)[0])
        c_inv = float(catb_models["avg_inventory"].predict(features)[0])
        c_ls = float(catb_models["lost_sales"].predict(features)[0])
        c_ord = float(catb_models["num_orders"].predict(features)[0])
        avg_inv = (l_inv + c_inv) / 2.0
        lost_sales = (l_ls + c_ls) / 2.0
        num_orders = (l_ord + c_ord) / 2.0
        engine_used = "Ensemble (LightGBM + CatBoost)"
    else:
        avg_inv = float(lgbm_models["avg_inventory"].predict(features)[0])
        lost_sales = float(lgbm_models["lost_sales"].predict(features)[0])
        num_orders = float(lgbm_models["num_orders"].predict(features)[0])
        engine_used = "LightGBM"

    return {
        "expected_avg_inventory": round(max(0.0, avg_inv), 4),
        "expected_lost_sales":    round(max(0.0, lost_sales), 4),
        "expected_num_orders":    round(max(0.0, num_orders), 4),
        "model_used":             engine_used,
    }


@tool
def calculate_financial_impact(
    expected_avg_inventory: float,
    expected_lost_sales: float,
    expected_num_orders: float,
    holding_cost: float = 1.0,
    shortage_cost: float = 10.0,
    order_cost: float = 50.0,
) -> Dict[str, float]:
    """Compute the total monetary cost of an inventory policy.

    Call this tool ONLY after simulate_policy_outcomes has returned its predictions.
    Uses the linear cost equation: Cost = h * AvgInventory + b * LostSales + k * NumOrders.

    Parameters
    ----------
    expected_avg_inventory : float
        Output from simulate_policy_outcomes.
    expected_lost_sales : float
        Output from simulate_policy_outcomes.
    expected_num_orders : float
        Output from simulate_policy_outcomes.
    holding_cost : float
        Per-unit per-day holding cost h (default 1.0).
    shortage_cost : float
        Per-unit stock-out penalty b (default 10.0).
    order_cost : float
        Fixed cost per order k (default 50.0).

    Returns
    -------
    dict
        Keys: total_cost, holding_component, shortage_component, ordering_component.
    """
    holding   = holding_cost  * expected_avg_inventory
    shortage  = shortage_cost * expected_lost_sales
    ordering  = order_cost    * expected_num_orders
    total     = holding + shortage + ordering

    return {
        "total_cost":           round(total, 4),
        "holding_component":    round(holding, 4),
        "shortage_component":   round(shortage, 4),
        "ordering_component":   round(ordering, 4),
    }


__all__ = [
    "PolicySimulationInput",
    "load_moic_models",
    "simulate_policy_outcomes",
    "calculate_financial_impact",
]
