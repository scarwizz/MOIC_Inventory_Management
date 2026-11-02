"""features.py
-----------
Feature engineering functions for MOIC Inventory Management.
Computes time-series characteristics (ADI, CV2) and surrogate model features.
"""

from typing import Union, Dict, Any, Tuple, Optional
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


def demand_features(
    series: Union[pd.Series, np.ndarray, pd.DataFrame, list],
    history_end_day: int,
    history_start_day: Optional[int] = None,
) -> Dict[str, float]:
    """Calculate ADI and CV2 demand features using ONLY days <= history_end_day (inclusive).

    Convention: history_end_day is inclusive. Days > history_end_day are NEVER read.
    If history_start_day is provided, restricts to history_start_day <= day <= history_end_day.

    Parameters
    ----------
    series : pd.Series, np.ndarray, pd.DataFrame, or list
        Daily demand series.
    history_end_day : int
        Last day of history to include (inclusive).
    history_start_day : Optional[int]
        Optional earliest day of history to include (inclusive).

    Returns
    -------
    Dict[str, float]
        {'ADI': float, 'CV2': float}
    """
    if isinstance(series, pd.DataFrame):
        if len(series) == 1:
            series = series.iloc[0]
        else:
            raise ValueError("demand_features expects a single series or 1-row DataFrame")

    if isinstance(series, pd.Series):
        d_cols = [c for c in series.index if isinstance(c, str) and c.startswith("d_") and c[2:].isdigit()]
        if d_cols:
            valid_cols = []
            for c in d_cols:
                day_num = int(c[2:])
                if day_num <= history_end_day:
                    if history_start_day is None or day_num >= history_start_day:
                        valid_cols.append((day_num, c))
            valid_cols.sort(key=lambda x: x[0])
            vals = series[[c for _, c in valid_cols]].values.astype(float)
            return calculate_ts_features(vals)
        else:
            vals = series.values.astype(float)
    elif isinstance(series, (list, tuple)):
        vals = np.asarray(series, dtype=float)
    else:
        vals = np.asarray(series, dtype=float)

    start_idx = 0 if history_start_day is None else max(0, history_start_day - 1)
    end_idx = min(len(vals), history_end_day)
    return calculate_ts_features(vals[start_idx:end_idx])


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
        "ADI": np.atleast_1d(np.asarray(adi, dtype=float)),
        "CV2": np.atleast_1d(np.asarray(cv2, dtype=float)),
        "R": np.atleast_1d(np.asarray(r, dtype=float)),
        "L": np.atleast_1d(np.asarray(l, dtype=float)),
        "TSL": np.atleast_1d(np.asarray(tsl, dtype=float)),
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
