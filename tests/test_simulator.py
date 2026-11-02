"""
audit_fixed/test_simulator.py
-----------------------------
Unit tests for the corrected discrete-event inventory simulator:
1. Stock/flow conservation across diverse random scenarios.
2. Exact arrival timing at t + L.
3. Invariant inventory when demand = 0, pipeline = 0, new orders = 0.
4. Outstanding orders with zero demand arrive at t + L and increase inventory.
5. Deterministic hand-calculated test matching manual analytical derivation.
"""

import sys
import numpy as np

# Ensure local import
from src.simulator import run_inventory_simulation, calculate_policy_parameters


def test_stock_flow_conservation():
    print("Test 1: Stock/Flow Conservation across stochastic scenarios...")
    np.random.seed(42)
    for trial in range(10):
        horizon = np.random.randint(20, 100)
        demand = np.random.poisson(lam=5.0, size=horizon).astype(float)
        R = np.random.randint(1, 14)
        L = np.random.randint(0, 10)
        s = float(np.random.randint(10, 40))
        S = s + float(np.random.randint(10, 50))
        I0 = float(np.random.randint(0, int(S)))

        res = run_inventory_simulation(
            demand_series=demand,
            R=R,
            L=L,
            s=s,
            S=S,
            initial_on_hand=I0,
        )
        assert res["conservation_passed"], f"Conservation failed on trial {trial}: error={res['conservation_error']}"
    print("  -> PASSED: All 10 random trials satisfy Ending = Starting + Receipts - Satisfied.\n")


def test_exact_arrival_timing():
    print("Test 2: Exact arrival timing at day t + L...")
    # Horizon 10 days, zero demand. Start with 0 inventory, s=10, S=20.
    # At t=0, R=5, L=3. IP=0 <= 10 -> Order Q=20 placed.
    # It must arrive exactly on day t=0 + L=3.
    # Days 0, 1, 2 must have receipts=0 and on_hand=0.
    # Day 3 must have receipt=20 and ending on_hand=20.
    demand = np.zeros(10, dtype=float)
    res = run_inventory_simulation(
        demand_series=demand,
        R=5,
        L=3,
        s=10.0,
        S=20.0,
        initial_on_hand=0.0,
    )
    trace = res["daily_trace"]
    print("  Day-by-day receipts:", trace["receipts"][:6])
    print("  Day-by-day ending on-hand:", trace["end_inventory"][:6])

    assert trace["receipts"][0] == 0.0, "Receipt on day 0 should be 0"
    assert trace["receipts"][1] == 0.0, "Receipt on day 1 should be 0"
    assert trace["receipts"][2] == 0.0, "Receipt on day 2 should be 0"
    assert trace["receipts"][3] == 20.0, "Receipt on day 3 should be 20.0"
    assert trace["receipts"][4] == 0.0, "Receipt on day 4 should be 0"

    assert trace["end_inventory"][0] == 0.0
    assert trace["end_inventory"][1] == 0.0
    assert trace["end_inventory"][2] == 0.0
    assert trace["end_inventory"][3] == 20.0
    assert trace["end_inventory"][4] == 20.0
    print("  -> PASSED: Order arrived exactly at t=3 (t=0 + L=3), not before or after.\n")


def test_invariant_inventory_zero_demand_no_orders():
    print("Test 3: Invariant inventory with zero demand and no orders...")
    # Start with I0 = 15, s = 10, S = 20, R = 3, L = 2.
    # Since IP = 15 > s = 10, no order is placed.
    # Since demand = 0, on-hand inventory should remain exactly 15 every single day.
    demand = np.zeros(15, dtype=float)
    res = run_inventory_simulation(
        demand_series=demand,
        R=3,
        L=2,
        s=10.0,
        S=20.0,
        initial_on_hand=15.0,
    )
    trace = res["daily_trace"]
    assert all(x == 15.0 for x in trace["end_inventory"]), f"Inventory drifted: {trace['end_inventory']}"
    assert res["total_orders"] == 0, f"Unexpected orders placed: {res['total_orders']}"
    assert res["total_lost_sales"] == 0.0
    print("  -> PASSED: Inventory remained constant at 15.0 across all 15 days.\n")


def test_outstanding_orders_zero_demand():
    print("Test 4: Inventory increases when pipeline order arrives despite zero demand...")
    # Start with I0 = 5, but with an existing pipeline order of 10 arriving on day 3.
    # Zero demand throughout.
    demand = np.zeros(6, dtype=float)
    res = run_inventory_simulation(
        demand_series=demand,
        R=10,
        L=5,
        s=4.0,  # s=4, so with IP = 5 + 10 = 15 > 4, no new order is placed
        S=20.0,
        initial_on_hand=5.0,
        initial_pipeline=[(3, 10.0)],
    )
    trace = res["daily_trace"]
    print("  Daily ending inventory with incoming order on day 3:", trace["end_inventory"])
    assert trace["end_inventory"][0] == 5.0
    assert trace["end_inventory"][1] == 5.0
    assert trace["end_inventory"][2] == 5.0
    assert trace["end_inventory"][3] == 15.0  # Arrived!
    assert trace["end_inventory"][4] == 15.0
    assert trace["end_inventory"][5] == 15.0
    print("  -> PASSED: Inventory changed from 5.0 to 15.0 when pipeline order arrived under zero demand.\n")


def test_hand_calculated_scenario():
    print("Test 5: Deterministic hand-calculated demand-and-replenishment scenario...")
    # Parameters: R=2, L=2, s=10.0, S=20.0, I0=12.0
    # Demand series for 6 days: [5.0, 4.0, 5.0, 3.0, 4.0, 2.0]
    demand = np.array([5.0, 4.0, 5.0, 3.0, 4.0, 2.0], dtype=float)

    # Expected analytical trace derived by hand:
    # Day 0: start=12, order=0, dem=5, sat=5, lost=0, end=7
    # Day 1: start=7, order=0, dem=4, sat=4, lost=0, end=3
    # Day 2: start=3, review(IP=3<=10)->order Q=17 (arrives day 4), dem=5, sat=3, lost=2, end=0
    # Day 3: start=0, order=0, dem=3, sat=0, lost=3, end=0
    # Day 4: receipt=17, start=17, review(IP=17>10)->order=0, dem=4, sat=4, lost=0, end=13
    # Day 5: start=13, order=0, dem=2, sat=2, lost=0, end=11
    expected_receipts = [0.0, 0.0, 0.0, 0.0, 17.0, 0.0]
    expected_orders = [0, 0, 1, 0, 0, 0]
    expected_order_qtys = [0.0, 0.0, 17.0, 0.0, 0.0, 0.0]
    expected_satisfied = [5.0, 4.0, 3.0, 0.0, 4.0, 2.0]
    expected_lost_sales = [0.0, 0.0, 2.0, 3.0, 0.0, 0.0]
    expected_end_inv = [7.0, 3.0, 0.0, 0.0, 13.0, 11.0]

    res = run_inventory_simulation(
        demand_series=demand,
        R=2,
        L=2,
        s=10.0,
        S=20.0,
        initial_on_hand=12.0,
    )
    trace = res["daily_trace"]

    print("  Comparison of Hand-Calculated vs Simulator Output:")
    print(f"    {'Metric':<20} | {'Expected':<30} | {'Actual':<30}")
    print("    " + "-" * 75)
    print(f"    {'Receipts':<20} | {str(expected_receipts):<30} | {str(trace['receipts']):<30}")
    print(f"    {'Orders Placed':<20} | {str(expected_orders):<30} | {str(trace['orders_placed']):<30}")
    print(f"    {'Order Quantities':<20} | {str(expected_order_qtys):<30} | {str(trace['order_quantities']):<30}")
    print(f"    {'Satisfied Demand':<20} | {str(expected_satisfied):<30} | {str(trace['satisfied']):<30}")
    print(f"    {'Lost Sales':<20} | {str(expected_lost_sales):<30} | {str(trace['lost_sales']):<30}")
    print(f"    {'Ending Inventory':<20} | {str(expected_end_inv):<30} | {str(trace['end_inventory']):<30}")

    assert trace["receipts"] == expected_receipts
    assert trace["orders_placed"] == expected_orders
    assert trace["order_quantities"] == expected_order_qtys
    assert trace["satisfied"] == expected_satisfied
    assert trace["lost_sales"] == expected_lost_sales
    assert trace["end_inventory"] == expected_end_inv

    expected_avg_inv = 34.0 / 6.0
    expected_total_ls = 5.0
    expected_total_orders = 1
    expected_fill_rate = 18.0 / 23.0

    print("    " + "-" * 75)
    print(f"    {'Avg Inventory':<20} | {expected_avg_inv:<30.4f} | {res['avg_inventory']:<30.4f}")
    print(f"    {'Total Lost Sales':<20} | {expected_total_ls:<30.4f} | {res['total_lost_sales']:<30.4f}")
    print(f"    {'Total Orders':<20} | {expected_total_orders:<30} | {res['total_orders']:<30}")
    print(f"    {'Fill Rate':<20} | {expected_fill_rate:<30.4f} | {res['fill_rate']:<30.4f}")

    assert abs(res["avg_inventory"] - expected_avg_inv) < 1e-6
    assert abs(res["total_lost_sales"] - expected_total_ls) < 1e-6
    assert res["total_orders"] == expected_total_orders
    assert abs(res["fill_rate"] - expected_fill_rate) < 1e-6

    print("  -> PASSED: Exact match between manual derivation and simulator output!\n")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING VERIFICATION TEST SUITE FOR CORRECTED INVENTORY SIMULATOR")
    print("=" * 70 + "\n")
    test_stock_flow_conservation()
    test_exact_arrival_timing()
    test_invariant_inventory_zero_demand_no_orders()
    test_outstanding_orders_zero_demand()
    test_hand_calculated_scenario()
    print("=" * 70)
    print("ALL 5 UNIT & SANITY TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
