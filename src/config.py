"""config.py
---------
Central configuration for MOIC inventory management experiment design.
Enforces strict origin-based causality, chronological validation, and zero leakage.
"""

from typing import Dict, Any, List

RANDOM_SEED = 42

# Cost Parameters
H_ANNUAL = 0.01          # Annual holding cost per unit
H_DAILY = H_ANNUAL / 365.0  # Daily holding cost per unit
B_SHORTAGE = 0.25        # Shortage penalty per unit lost sales
K_ORDER = 0.50           # Fixed ordering cost per order placed
SCORING_HORIZON_DAYS = 28 # Scoring window length (days)
WARMUP_DAYS = 60         # Warm-up period length (days)
CALIBRATION_DAYS = 365   # Calibration history length (days)

# Retail Sensitivity Cost Parameters
H_ANNUAL_RETAIL = 0.20
H_DAILY_RETAIL = H_ANNUAL_RETAIL / 365.0

# Standard 75-policy grid (5 R x 5 L x 3 TSL)
R_GRID = [1, 8, 15, 22, 30]
L_GRID = [1, 8, 15, 22, 30]
TSL_GRID = [0.90, 0.95, 0.99]

# Evaluation Origin (M5 historical days: 1-indexed d_1 to d_1913)
# Calibration: d_1186 to d_1550 (365 days)
# Warm-up: d_1551 to d_1610 (60 days)
# Scoring: d_1611 to d_1638 (28 days)
EVAL_ORIGIN: Dict[str, Any] = {
    "origin_id": "eval",
    "history_end_day": 1550,
    "calib_start_day": 1186,
    "calib_end_day": 1550,
    "warmup_start_day": 1551,
    "warmup_end_day": 1610,
    "scoring_start_day": 1611,
    "scoring_end_day": 1638,
    "calib_days": 365,
    "warmup_days": 60,
    "scoring_days": 28,
}

# Training and Chronological Validation Origins:
# All scoring windows must end by d_1185 or earlier to guarantee ZERO leakage
# into the evaluation period (which starts at calibration day d_1186).
#
# Validation Origin: decision origin at d_1097
# - Calibration: d_733 to d_1097 (365 days)
# - Warm-up: d_1098 to d_1157 (60 days)
# - Scoring: d_1158 to d_1185 (28 days) -> ends at d_1185 (<= 1185)
#
# Training Origin 1: decision origin at d_1009
# - Calibration: d_645 to d_1009 (365 days)
# - Warm-up: d_1010 to d_1069 (60 days)
# - Scoring: d_1070 to d_1097 (28 days) -> fully realized by val decision day 1097
#
# Training Origin 0: decision origin at d_921
# - Calibration: d_557 to d_921 (365 days)
# - Warm-up: d_922 to d_981 (60 days)
# - Scoring: d_982 to d_1009 (28 days) -> fully realized by origin 1 decision day 1009

VALIDATION_ORIGIN: Dict[str, Any] = {
    "origin_id": "val_1097",
    "history_end_day": 1097,
    "calib_start_day": 733,
    "calib_end_day": 1097,
    "warmup_start_day": 1098,
    "warmup_end_day": 1157,
    "scoring_start_day": 1158,
    "scoring_end_day": 1185,
    "calib_days": 365,
    "warmup_days": 60,
    "scoring_days": 28,
}

TRAINING_ORIGINS: List[Dict[str, Any]] = [
    {
        "origin_id": "train_921",
        "history_end_day": 921,
        "calib_start_day": 557,
        "calib_end_day": 921,
        "warmup_start_day": 922,
        "warmup_end_day": 981,
        "scoring_start_day": 982,
        "scoring_end_day": 1009,
        "calib_days": 365,
        "warmup_days": 60,
        "scoring_days": 28,
    },
    {
        "origin_id": "train_1009",
        "history_end_day": 1009,
        "calib_start_day": 645,
        "calib_end_day": 1009,
        "warmup_start_day": 1010,
        "warmup_end_day": 1069,
        "scoring_start_day": 1070,
        "scoring_end_day": 1097,
        "calib_days": 365,
        "warmup_days": 60,
        "scoring_days": 28,
    },
]

ALL_EXPERIMENT_ORIGINS = TRAINING_ORIGINS + [VALIDATION_ORIGIN]

# History-Tuned Baseline Backtest Fold (strictly within days <= 1550)
# Calibrates (s, S) strictly on days <= 1462 before the backtest warm-up.
# Simulates warm-up d_1463 to d_1522 (60 days) and scoring d_1523 to d_1550 (28 days).
BACKTEST_TUNING_FOLD: Dict[str, Any] = {
    "history_end_day": 1462,
    "calib_start_day": 1098,
    "calib_end_day": 1462,
    "warmup_start_day": 1463,
    "warmup_end_day": 1522,
    "scoring_start_day": 1523,
    "scoring_end_day": 1550,
    "calib_days": 365,
    "warmup_days": 60,
    "scoring_days": 28,
}


def purge_overlapping_labels(train_origins: List[Dict[str, Any]], val_origin: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Verify and purge any training origin whose target window overlaps with validation
    or whose scoring window is not fully realized by the validation decision origin time.
    """
    purged = []
    val_decision_day = val_origin["history_end_day"]
    val_target_start = val_origin["warmup_start_day"]
    val_target_end = val_origin["scoring_end_day"]

    for origin in train_origins:
        # Rule 1: Every training label must be fully realized by the validation decision time
        if origin["scoring_end_day"] > val_decision_day:
            continue
        # Rule 2: Train and validation target (warm-up + scoring) windows must be non-overlapping
        t_start = origin["warmup_start_day"]
        t_end = origin["scoring_end_day"]
        if not (t_end < val_target_start or t_start > val_target_end):
            continue
        purged.append(origin)
    return purged
