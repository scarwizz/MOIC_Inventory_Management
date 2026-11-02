"""Integration unit tests for MOIC pipeline components."""

import unittest
import numpy as np
import pandas as pd

from src.features import engineer_surrogate_features, calculate_ts_features
from src.optimization import (
    compute_inventory_cost,
    generate_candidate_policies,
    select_optimal_policy_for_sku,
)
from src.evaluation import evaluate_sku_policy


class DummyModel:
    def __init__(self, const_val: float):
        self.const_val = const_val

    def predict(self, X):
        return np.full(len(X), self.const_val)


class TestPipeline(unittest.TestCase):
    def test_candidate_policies_count(self):
        policies = generate_candidate_policies()
        self.assertEqual(len(policies), 75)

    def test_cost_calculation(self):
        # 10 avg inv, 2 lost sales, 3 orders
        # h=0.01/365, b=0.25, k=0.50, horizon=365
        # h_term = (0.01/365) * 10 * 365 = 0.10
        # b_term = 0.25 * 2 = 0.50
        # k_term = 0.50 * 3 = 1.50
        # Total = 2.10
        cost = compute_inventory_cost(10.0, 2.0, 3.0, h_annual=0.01, b=0.25, k=0.50, horizon_days=365)
        self.assertAlmostEqual(cost, 2.10)

    def test_select_optimal_policy_dummy(self):
        models = {
            "avg_inventory": DummyModel(10.0),
            "lost_sales": DummyModel(1.0),
            "num_orders": DummyModel(2.0),
        }
        res = select_optimal_policy_for_sku(models, adi=1.5, cv2=0.5)
        self.assertIn("R", res)
        self.assertIn("L", res)
        self.assertIn("TSL", res)
        self.assertIn("pred_cost", res)

    def test_evaluate_sku_policy(self):
        # Synthetic demand: 365 + 60 + 28 days = 453 days
        np.random.seed(42)
        demand = np.random.poisson(lam=2.0, size=453).astype(float)
        hist_slice = slice(0, 365)
        warmup_slice = slice(365, 425)
        eval_slice = slice(425, 453)

        res = evaluate_sku_policy(
            demand_series=demand,
            history_slice=hist_slice,
            warmup_slice=warmup_slice,
            eval_slice=eval_slice,
            r=7,
            l=3,
            tsl=0.95,
        )
        self.assertIn("cost", res)
        self.assertIn("fill_rate", res)
        self.assertGreaterEqual(res["fill_rate"], 0.0)
        self.assertLessEqual(res["fill_rate"], 1.0)
        self.assertGreaterEqual(res["cost"], 0.0)


if __name__ == "__main__":
    unittest.main()
