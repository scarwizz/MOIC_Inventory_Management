"""features.py
-----------
Feature engineering functions for MOIC Inventory Management.
Computes time-series characteristics (ADI, CV2) and surrogate model features.
"""

from typing import Union, Dict, Any, Tuple
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ADI",
    "CV2",
    "R",
    "L",
    "TSL",
    "Risk_Period",
    "Policy_Interaction",
    "Demand_Profile",
]


def calculate_ts_features(series: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
    """Calculate Average Inter-Demand Interval (ADI) and Squared Coefficient of Variation (CV²).

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Historical demand series.

    Returns
    -------
    dict
        {'ADI': float, 'CV2': float}
    """
    if isinstance(series, pd.Series):
        vals = series.values
    else:
        vals = np.asarray(series)

    total_periods = len(vals)
    non_zero = vals[vals > 0]
    non_zero_periods = len(non_zero)

    if non_zero_periods == 0:
        return {"ADI": float(total_periods), "CV2": 0.0}

    adi = float(total_periods / non_zero_periods)
    mean_d = float(np.mean(non_zero))
    if non_zero_periods > 1:
        std_d = float(np.std(non_zero, ddof=1))
    else:
        std_d = 0.0

    if mean_d == 0.0 or np.isnan(std_d):
        cv2 = 0.0
    else:
        cv2 = float((std_d / mean_d) ** 2)

    return {"ADI": adi, "CV2": cv2}


def engineer_surrogate_features(
    adi: Union[float, np.ndarray, pd.Series],
    cv2: Union[float, np.ndarray, pd.Series],
    r: Union[int, float, np.ndarray, pd.Series],
    l: Union[int, float, np.ndarray, pd.Series],
    tsl: Union[float, np.ndarray, pd.Series],
) -> pd.DataFrame:
    """Construct the exact 8-feature DataFrame used by MOIC surrogate models.

    Columns:
    - ADI: Average Inter-Demand Interval
    - CV2: Squared Coefficient of Variation
    - R: Review period (days)
    - L: Lead time (days)
    - TSL: Target Service Level
    - Risk_Period: R + L
    - Policy_Interaction: R * L
    - Demand_Profile: ADI / CV2 (0.0 if CV2 == 0)
    """
    df = pd.DataFrame({
        "ADI": np.asarray(adi, dtype=float),
        "CV2": np.asarray(cv2, dtype=float),
        "R": np.asarray(r, dtype=float),
        "L": np.asarray(l, dtype=float),
        "TSL": np.asarray(tsl, dtype=float),
    })

    df["Risk_Period"] = df["R"] + df["L"]
    df["Policy_Interaction"] = df["R"] * df["L"]
    df["Demand_Profile"] = np.where(df["CV2"] > 0, df["ADI"] / df["CV2"], 0.0)

    return df[FEATURE_COLUMNS]


def categorize_demand(adi: float, cv2: float, adi_threshold: float = 1.32, cv2_threshold: float = 0.49) -> str:
    """Categorize demand into one of four Syntetos-Boylan quadrants.

    Categories:
    - Smooth: ADI < 1.32, CV2 < 0.49
    - Intermittent: ADI >= 1.32, CV2 < 0.49
    - Erratic: ADI < 1.32, CV2 >= 0.49
    - Lumpy: ADI >= 1.32, CV2 >= 0.49
    """
    if adi < adi_threshold and cv2 < cv2_threshold:
        return "Smooth"
    elif adi >= adi_threshold and cv2 < cv2_threshold:
        return "Intermittent"
    elif adi < adi_threshold and cv2 >= cv2_threshold:
        return "Erratic"
    else:
        return "Lumpy"
