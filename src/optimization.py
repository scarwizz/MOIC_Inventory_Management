"""optimization.py
---------------
Inventory policy optimization via surrogate model cost minimization.
Evaluates candidate (R, L, TSL) policies on a discrete grid to find
the cost-minimizing policy for each SKU.
"""

from typing import Dict, Any, List, Tuple, Union
import itertools
import numpy as np
import pandas as pd

from src.features import engineer_surrogate_features, FEATURE_COLUMNS

# Standard 75-policy grid
R_GRID = [1, 8, 15, 22, 30]
L_GRID = [1, 8, 15, 22, 30]
TSL_GRID = [0.90, 0.95, 0.99]


def compute_inventory_cost(
    avg_inventory: Union[float, np.ndarray],
    lost_sales: Union[float, np.ndarray],
    num_orders: Union[float, np.ndarray],
    h_annual: float = 0.01,
    b: float = 0.25,
    k: float = 0.50,
    horizon_days: int = 28,
) -> Union[float, np.ndarray]:
    """Compute total inventory cost over the evaluation horizon.

    C = (h_annual / 365) * avg_inventory * horizon_days + b * lost_sales + k * num_orders
    """
    h_daily = h_annual / 365.0
    return (h_daily * avg_inventory * horizon_days) + (b * lost_sales) + (k * num_orders)


def generate_candidate_policies() -> List[Tuple[int, int, float]]:
    """Return all 75 candidate (R, L, TSL) policy combinations."""
    return list(itertools.product(R_GRID, L_GRID, TSL_GRID))


def build_candidate_grid_for_sku(adi: float, cv2: float) -> pd.DataFrame:
    """Generate 75-row feature DataFrame for a single SKU across the full policy grid."""
    policies = generate_candidate_policies()
    r_vals = [p[0] for p in policies]
    l_vals = [p[1] for p in policies]
    tsl_vals = [p[2] for p in policies]
    adi_vals = [adi] * len(policies)
    cv2_vals = [cv2] * len(policies)

    grid_df = engineer_surrogate_features(adi_vals, cv2_vals, r_vals, l_vals, tsl_vals)
    return grid_df


def select_optimal_policy_for_sku(
    models: Dict[str, Any],
    adi: float,
    cv2: float,
    h_annual: float = 0.01,
    b: float = 0.25,
    k: float = 0.50,
    horizon_days: int = 28,
) -> Dict[str, Any]:
    """Find the cost-minimizing policy among the 75 candidates for a single SKU.

    Parameters
    ----------
    models : dict
        {'avg_inventory': model, 'lost_sales': model, 'num_orders': model}
    """
    grid_df = build_candidate_grid_for_sku(adi, cv2)

    pred_inv = np.maximum(0.0, models["avg_inventory"].predict(grid_df[FEATURE_COLUMNS]))
    pred_ls = np.maximum(0.0, models["lost_sales"].predict(grid_df[FEATURE_COLUMNS]))
    pred_ord = np.maximum(0.0, models["num_orders"].predict(grid_df[FEATURE_COLUMNS]))

    costs = compute_inventory_cost(pred_inv, pred_ls, pred_ord, h_annual=h_annual, b=b, k=k, horizon_days=horizon_days)

    best_idx = int(np.argmin(costs))
    best_row = grid_df.iloc[best_idx]

    return {
        "R": int(best_row["R"]),
        "L": int(best_row["L"]),
        "TSL": float(best_row["TSL"]),
        "pred_cost": float(costs[best_idx]),
        "pred_avg_inventory": float(pred_inv[best_idx]),
        "pred_lost_sales": float(pred_ls[best_idx]),
        "pred_num_orders": float(pred_ord[best_idx]),
    }


def optimize_all_skus(
    models: Dict[str, Any],
    sku_features_df: pd.DataFrame,
    h_annual: float = 0.01,
    b: float = 0.25,
    k: float = 0.50,
    horizon_days: int = 28,
) -> pd.DataFrame:
    """Run fully vectorized policy optimization across all SKUs.

    Evaluates all (N_SKUs x 75) combinations in a single vectorized batch inference.
    """
    policies = generate_candidate_policies()
    n_policies = len(policies)
    n_skus = len(sku_features_df)

    r_arr = np.array([p[0] for p in policies], dtype=float)
    l_arr = np.array([p[1] for p in policies], dtype=float)
    tsl_arr = np.array([p[2] for p in policies], dtype=float)

    r_tiled = np.tile(r_arr, n_skus)
    l_tiled = np.tile(l_arr, n_skus)
    tsl_tiled = np.tile(tsl_arr, n_skus)

    adis = np.repeat(sku_features_df["ADI"].values.astype(float), n_policies)
    cv2s = np.repeat(sku_features_df["CV2"].values.astype(float), n_policies)

    full_grid_df = engineer_surrogate_features(adis, cv2s, r_tiled, l_tiled, tsl_tiled)

    # Vectorized inference in 3 calls
    pred_inv = np.maximum(0.0, models["avg_inventory"].predict(full_grid_df[FEATURE_COLUMNS]))
    pred_ls = np.maximum(0.0, models["lost_sales"].predict(full_grid_df[FEATURE_COLUMNS]))
    pred_ord = np.maximum(0.0, models["num_orders"].predict(full_grid_df[FEATURE_COLUMNS]))

    costs = compute_inventory_cost(
        pred_inv, pred_ls, pred_ord,
        h_annual=h_annual, b=b, k=k, horizon_days=horizon_days
    )

    cost_matrix = costs.reshape((n_skus, n_policies))
    best_indices = np.argmin(cost_matrix, axis=1)

    flat_best_indices = np.arange(n_skus) * n_policies + best_indices

    records = {
        "item_id": sku_features_df["item_id"].values if "item_id" in sku_features_df else [""] * n_skus,
        "state_id": sku_features_df["state_id"].values if "state_id" in sku_features_df else [""] * n_skus,
        "R": r_arr[best_indices].astype(int),
        "L": l_arr[best_indices].astype(int),
        "TSL": tsl_arr[best_indices],
        "pred_cost": costs[flat_best_indices],
        "pred_avg_inventory": pred_inv[flat_best_indices],
        "pred_lost_sales": pred_ls[flat_best_indices],
        "pred_num_orders": pred_ord[flat_best_indices],
    }
    if "id" in sku_features_df:
        records["id"] = sku_features_df["id"].values

    return pd.DataFrame(records)
