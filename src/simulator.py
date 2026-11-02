"""
audit_fixed/simulator.py
------------------------
Corrected discrete-event periodic-review (R, s, S) inventory simulator with:
- Exact replenishment arrival tracking at t + L
- Correct Inventory Position (IP = on_hand + on_order) reordering logic
- Strict stock-flow conservation
- Support for optional warm-up / burn-in period to eliminate initial-condition bias
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.stats import norm


def calculate_policy_parameters(
    historical_sales: np.ndarray,
    R: int,
    L: int,
    TSL: float,
) -> Tuple[float, float, float, float, float]:
    """Calculate empirical policy parameters from historical sales.

    Parameters
    ----------
    historical_sales : np.ndarray
        Array of daily demand observations.
    R : int
        Review period (days).
    L : int
        Replenishment lead time (days).
    TSL : float
        Target Service Level in [0, 1].

    Returns
    -------
    Tuple[float, float, float, float, float]
        (D_hat, sigma_D, SS, s, S)
    """
    if len(historical_sales) == 0:
        return 0.0, 1.0, 0.0, 0.0, 0.0

    D_hat = float(np.mean(historical_sales))
    sigma_D = float(np.std(historical_sales, ddof=1)) if len(historical_sales) > 1 else 1.0
    if sigma_D <= 0:
        sigma_D = 1.0

    # Clamp TSL to avoid inf Z-score
    tsl_clamped = float(np.clip(TSL, 1e-4, 1.0 - 1e-4))
    Z = float(norm.ppf(tsl_clamped))

    # Safety Stock formula: SS = Z * sigma_D * sqrt(R + L)
    SS = max(0.0, Z * sigma_D * np.sqrt(R + L))

    # Expected demand over lead time + review period
    lead_demand = D_hat * (R + L)

    # Order point s and Order-up-to level S
    s = lead_demand + SS
    S = lead_demand + SS

    return D_hat, sigma_D, SS, s, S


def run_inventory_simulation(
    demand_series: np.ndarray,
    R: int,
    L: int,
    s: float,
    S: float,
    initial_on_hand: Optional[float] = None,
    initial_pipeline: Optional[List[Tuple[int, float]]] = None,
    warmup_days: int = 0,
    h_daily: float = 0.01 / 365.0,
    b_shortage: float = 0.25,
    k_order: float = 0.50,
) -> Dict[str, Any]:
    """Simulate daily periodic-review (R, s, S) inventory control under lost sales.

    Event Sequence on Day t:
    1. Arrivals: Orders scheduled to arrive on day t arrive at the start of day.
       On-hand inventory increases, pipeline decreases.
    2. Review & Order: If t % R == 0:
       Evaluate Inventory Position: IP = on_hand + on_order.
       If IP <= s:
           Order quantity Q = max(0, S - IP).
           Schedule Q to arrive on day t + L.
           Increment orders placed counter.
           Update pipeline and IP.
    3. Demand Fulfillment:
       Satisfied sales = min(on_hand, demand_t).
       Lost sales = demand_t - satisfied sales.
       Ending on_hand = on_hand - satisfied sales.
    4. Metrics recorded after warmup_days.

    Parameters
    ----------
    demand_series : np.ndarray
        Daily demand for the full horizon (warmup + evaluation).
    R : int
        Review period in days (>= 1).
    L : int
        Lead time in days (>= 0).
    s : float
        Reorder point.
    S : float
        Order-up-to level.
    initial_on_hand : Optional[float]
        Initial on-hand inventory at t=0. Defaults to S.
    initial_pipeline : Optional[List[Tuple[int, float]]]
        List of (arrival_day, quantity) tuples for pre-existing orders.
    warmup_days : int
        Number of initial days to treat as warm-up (excluded from final metrics).
    h_daily : float
        Daily holding cost per unit.
    b_shortage : float
        Shortage penalty per unit lost sales.
    k_order : float
        Fixed cost per order placed.

    Returns
    -------
    Dict[str, Any]
        Dictionary with simulation metrics, conservation verification, and daily traces.
    """
    total_days = len(demand_series)
    eval_days = max(1, total_days - warmup_days)

    on_hand = S if initial_on_hand is None else float(initial_on_hand)
    initial_inventory = on_hand

    # Track scheduled arrivals: mapping arrival_day -> list of quantities
    arrivals_schedule: Dict[int, float] = {}
    if initial_pipeline:
        for arr_day, qty in initial_pipeline:
            arrivals_schedule[arr_day] = arrivals_schedule.get(arr_day, 0.0) + qty

    # Track daily records
    daily_start_inventory: List[float] = []
    daily_receipts: List[float] = []
    daily_orders_placed: List[int] = []
    daily_order_quantities: List[float] = []
    daily_demand: List[float] = []
    daily_satisfied: List[float] = []
    daily_lost_sales: List[float] = []
    daily_end_inventory: List[float] = []
    daily_inventory_position: List[float] = []
    daily_pipeline_stock: List[float] = []

    for t in range(total_days):
        # 1. Replenishment Arrival
        receipt = arrivals_schedule.pop(t, 0.0)
        on_hand += receipt

        start_inv_today = on_hand

        # Outstanding orders in pipeline (arriving on future days > t)
        pipeline_stock = sum(arrivals_schedule.values())

        # Inventory Position = On-hand + Pipeline
        inventory_position = on_hand + pipeline_stock

        # 2. Review and Ordering Decision
        order_placed_today = 0
        order_qty_today = 0.0
        if t % R == 0:
            if inventory_position <= s:
                order_qty_today = max(0.0, S - inventory_position)
                if order_qty_today > 0:
                    order_placed_today = 1
                    arrival_day = t + L
                    arrivals_schedule[arrival_day] = arrivals_schedule.get(arrival_day, 0.0) + order_qty_today
                    pipeline_stock += order_qty_today
                    inventory_position += order_qty_today

        # 3. Demand Fulfillment
        d = float(demand_series[t])
        satisfied = min(on_hand, d)
        lost = d - satisfied
        on_hand -= satisfied

        # Ensure no negative on-hand due to float inaccuracies
        if on_hand < 0:
            on_hand = 0.0

        end_inv_today = on_hand

        # Record daily trace
        daily_start_inventory.append(start_inv_today)
        daily_receipts.append(receipt)
        daily_orders_placed.append(order_placed_today)
        daily_order_quantities.append(order_qty_today)
        daily_demand.append(d)
        daily_satisfied.append(satisfied)
        daily_lost_sales.append(lost)
        daily_end_inventory.append(end_inv_today)
        daily_inventory_position.append(inventory_position)
        daily_pipeline_stock.append(pipeline_stock)

    # 4. Stock-Flow Conservation Verification over FULL horizon
    total_receipts_full = float(np.sum(daily_receipts))
    total_satisfied_full = float(np.sum(daily_satisfied))
    final_on_hand = daily_end_inventory[-1]
    expected_final_on_hand = initial_inventory + total_receipts_full - total_satisfied_full
    conservation_error = abs(final_on_hand - expected_final_on_hand)
    conservation_passed = bool(conservation_error < 1e-6)

    # 5. Extract Evaluation Window (Post-Warmup)
    eval_receipts = daily_receipts[warmup_days:]
    eval_orders = daily_orders_placed[warmup_days:]
    eval_order_qtys = daily_order_quantities[warmup_days:]
    eval_demand = daily_demand[warmup_days:]
    eval_satisfied = daily_satisfied[warmup_days:]
    eval_lost_sales = daily_lost_sales[warmup_days:]
    eval_end_inventory = daily_end_inventory[warmup_days:]

    total_eval_orders = int(np.sum(eval_orders))
    total_eval_demand = float(np.sum(eval_demand))
    total_eval_lost_sales = float(np.sum(eval_lost_sales))
    total_eval_satisfied = float(np.sum(eval_satisfied))
    avg_eval_inventory = float(np.mean(eval_end_inventory))

    fill_rate = (1.0 - (total_eval_lost_sales / total_eval_demand)) if total_eval_demand > 0 else 1.0

    holding_cost = h_daily * avg_eval_inventory * eval_days
    shortage_cost = b_shortage * total_eval_lost_sales
    order_cost = k_order * total_eval_orders
    total_cost = holding_cost + shortage_cost + order_cost

    return {
        "avg_inventory": avg_eval_inventory,
        "total_lost_sales": total_eval_lost_sales,
        "total_orders": total_eval_orders,
        "total_demand": total_eval_demand,
        "total_satisfied": total_eval_satisfied,
        "fill_rate": fill_rate,
        "holding_cost": holding_cost,
        "shortage_cost": shortage_cost,
        "order_cost": order_cost,
        "total_cost": total_cost,
        "eval_days": eval_days,
        "warmup_days": warmup_days,
        "conservation_passed": conservation_passed,
        "conservation_error": conservation_error,
        "initial_inventory": initial_inventory,
        "final_inventory": final_on_hand,
        "daily_trace": {
            "receipts": daily_receipts,
            "orders_placed": daily_orders_placed,
            "order_quantities": daily_order_quantities,
            "demand": daily_demand,
            "satisfied": daily_satisfied,
            "lost_sales": daily_lost_sales,
            "end_inventory": daily_end_inventory,
            "inventory_position": daily_inventory_position,
            "pipeline_stock": daily_pipeline_stock,
        },
    }
