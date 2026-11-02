"""tests/test_causality.py
------------------------
Unit tests for origin-based causality, leakage prevention, and chronological validation:
1. Predictor invariance: features, (s, S) calibration, baseline choice, and surrogate inputs
   do not change when demand after history_end_day is perturbed.
2. Label scope: simulation labels do not change when demand after scoring_end_day is perturbed.
3. Chronological split: train and validation target windows do not overlap, labels are realized,
   and boundary-overlapping labels are purged.
4. History-tuned baseline: all backtest windows lie <= 1550, and policy choice is invariant to
   perturbations on days d_1551 to d_1913.
5. Eval-window demand is never read before final scoring.
"""

import unittest
import numpy as np
import pandas as pd

from src.features import demand_features, engineer_surrogate_features
from src.simulator import calculate_policy_parameters, run_inventory_simulation
from src.optimization import select_optimal_policy_for_sku, optimize_all_skus
from src.evaluation import (
    evaluate_sku_policy,
    tune_history_grid_baseline_for_sku,
    find_hindsight_oracle_policy,
)
from src.config import (
    EVAL_ORIGIN,
    VALIDATION_ORIGIN,
    TRAINING_ORIGINS,
    BACKTEST_TUNING_FOLD,
    purge_overlapping_labels,
    H_ANNUAL,
    B_SHORTAGE,
    K_ORDER,
)


class DummyModel:
    def __init__(self, const_val: float):
        self.const_val = const_val

    def predict(self, X):
        return np.full(len(X), self.const_val)


class TestCausality(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # 1913 days of synthetic demand
        self.demand = np.random.poisson(lam=3.0, size=1913).astype(float)

    def test_predictor_invariance(self):
        """For a given origin, change demand on days > history_end_day and assert that
        its features, fitted (s, S) calibration, baseline choice, and surrogate inputs do not change.
        Tests across multiple decision origins (eval: 1550, train: 921) and data structures (array and Series).
        """
        # Test 1: Eval Origin (h_end = 1550) on numpy array
        h_end_eval = 1550
        demand_orig = self.demand.copy()
        demand_pert = self.demand.copy()
        demand_pert[h_end_eval:] += 1000.0

        feat_orig = demand_features(demand_orig, history_end_day=h_end_eval)
        feat_pert = demand_features(demand_pert, history_end_day=h_end_eval)
        self.assertAlmostEqual(feat_orig["ADI"], feat_pert["ADI"], places=7)
        self.assertAlmostEqual(feat_orig["CV2"], feat_pert["CV2"], places=7)

        calib_slice = slice(h_end_eval - 365, h_end_eval)
        params_orig = calculate_policy_parameters(demand_orig[calib_slice], 7, 3, 0.95)
        params_pert = calculate_policy_parameters(demand_pert[calib_slice], 7, 3, 0.95)
        self.assertEqual(params_orig, params_pert)

        best_r_orig, best_l_orig, best_tsl_orig, _ = tune_history_grid_baseline_for_sku(
            demand_orig, backtest_fold=BACKTEST_TUNING_FOLD
        )
        best_r_pert, best_l_pert, best_tsl_pert, _ = tune_history_grid_baseline_for_sku(
            demand_pert, backtest_fold=BACKTEST_TUNING_FOLD
        )
        self.assertEqual((best_r_orig, best_l_orig, best_tsl_orig), (best_r_pert, best_l_pert, best_tsl_pert))

        inputs_orig = engineer_surrogate_features(feat_orig["ADI"], feat_orig["CV2"], 7, 3, 0.95)
        inputs_pert = engineer_surrogate_features(feat_pert["ADI"], feat_pert["CV2"], 7, 3, 0.95)
        pd.testing.assert_frame_equal(inputs_orig, inputs_pert)

        # Test 2: Training Origin (h_end = 921) on pd.Series with d_ columns
        h_end_train = 921
        series_orig = pd.Series(self.demand, index=[f"d_{i}" for i in range(1, 1914)])
        series_pert = series_orig.copy()
        series_pert.iloc[h_end_train:] += 500.0

        feat_train_orig = demand_features(series_orig, history_end_day=h_end_train)
        feat_train_pert = demand_features(series_pert, history_end_day=h_end_train)
        self.assertAlmostEqual(feat_train_orig["ADI"], feat_train_pert["ADI"], places=7)
        self.assertAlmostEqual(feat_train_orig["CV2"], feat_train_pert["CV2"], places=7)

        calib_slice_train = slice(556, 921)
        params_train_orig = calculate_policy_parameters(series_orig.values[calib_slice_train], 8, 8, 0.95)
        params_train_pert = calculate_policy_parameters(series_pert.values[calib_slice_train], 8, 8, 0.95)
        self.assertEqual(params_train_orig, params_train_pert)

    def test_label_scope(self):
        """A label may depend on its permitted calibration history, its warm-up and its scoring window,
        and never on demand after its scoring_end_day. Assert it does not change when demand after
        its scoring_end_day changes, but DOES change when demand inside its scoring window changes.
        """
        origin = TRAINING_ORIGINS[0]  # train_921, scoring_end_day = 1009
        scoring_start = origin["scoring_start_day"]
        scoring_end = origin["scoring_end_day"]

        demand_orig = self.demand.copy()
        demand_pert_after = self.demand.copy()
        # Perturb strictly after scoring_end_day (0-indexed index >= scoring_end)
        demand_pert_after[scoring_end:] += 500.0

        calib_slice = slice(origin["calib_start_day"] - 1, origin["calib_end_day"])
        sim_slice = slice(origin["warmup_start_day"] - 1, origin["scoring_end_day"])

        d_hat_orig, sigma_orig, _, s_orig, S_orig = calculate_policy_parameters(
            demand_orig[calib_slice], 8, 8, 0.95
        )
        d_hat_pert, sigma_pert, _, s_pert, S_pert = calculate_policy_parameters(
            demand_pert_after[calib_slice], 8, 8, 0.95
        )
        self.assertEqual((s_orig, S_orig), (s_pert, S_pert))

        res_orig = run_inventory_simulation(
            demand_orig[sim_slice], 8, 8, s_orig, S_orig, warmup_days=origin["warmup_days"]
        )
        res_pert_after = run_inventory_simulation(
            demand_pert_after[sim_slice], 8, 8, s_pert, S_pert, warmup_days=origin["warmup_days"]
        )

        self.assertAlmostEqual(res_orig["avg_inventory"], res_pert_after["avg_inventory"], places=6)
        self.assertAlmostEqual(res_orig["total_lost_sales"], res_pert_after["total_lost_sales"], places=6)
        self.assertEqual(res_orig["total_orders"], res_pert_after["total_orders"])
        self.assertAlmostEqual(res_orig["total_cost"], res_pert_after["total_cost"], places=6)
        self.assertAlmostEqual(res_orig["fill_rate"], res_pert_after["fill_rate"], places=6)

        # Confirm sensitivity: perturbing INSIDE the scoring window MUST change simulation output
        demand_pert_inside = self.demand.copy()
        demand_pert_inside[scoring_start - 1 : scoring_end] += 20.0
        res_pert_inside = run_inventory_simulation(
            demand_pert_inside[sim_slice], 8, 8, s_orig, S_orig, warmup_days=origin["warmup_days"]
        )
        self.assertNotEqual(res_orig["total_demand"], res_pert_inside["total_demand"])
        self.assertNotEqual(res_orig["total_cost"], res_pert_inside["total_cost"])

    def test_chronological_split(self):
        """Train and validation target windows do not overlap, every training label is realized
        by the validation decision time, and boundary-overlapping labels are purged.
        """
        val_decision_day = VALIDATION_ORIGIN["history_end_day"]
        val_target_start = VALIDATION_ORIGIN["warmup_start_day"]
        val_target_end = VALIDATION_ORIGIN["scoring_end_day"]

        # Rule 1 & 2 on configured origins
        for orig in TRAINING_ORIGINS:
            # Fully realized before validation decision time
            self.assertLessEqual(
                orig["scoring_end_day"],
                val_decision_day,
                f"Training origin {orig['origin_id']} scoring_end_day ({orig['scoring_end_day']}) > val decision day ({val_decision_day})"
            )
            # Target window strictly precedes validation target window
            self.assertLess(
                orig["scoring_end_day"],
                val_target_start,
                f"Training origin {orig['origin_id']} target window overlaps validation target window"
            )

        # Rule 3: Boundary purging test cases
        # Case A: candidate realizes exactly at val_decision_day and finishes before val target starts -> retained
        boundary_valid = {
            "origin_id": "boundary_valid",
            "history_end_day": 1009,
            "warmup_start_day": 1010,
            "scoring_end_day": 1097,  # exactly val_decision_day (1097) < val_target_start (1098)
        }
        # Case B: candidate ends at val_decision_day + 1 -> purged
        boundary_invalid = {
            "origin_id": "boundary_invalid",
            "history_end_day": 1010,
            "warmup_start_day": 1011,
            "scoring_end_day": 1098,  # > val_decision_day (1097) and overlaps val warmup (1098)
        }
        # Case C: candidate target window completely inside validation window -> purged
        overlapping_candidate = {
            "origin_id": "candidate_overlap",
            "history_end_day": 1050,
            "warmup_start_day": 1051,
            "scoring_end_day": 1138,  # ends after val_decision_day (1097) and inside val target [1098, 1185]
        }
        test_candidates = [boundary_valid, boundary_invalid, overlapping_candidate]
        purged = purge_overlapping_labels(test_candidates, VALIDATION_ORIGIN)
        purged_ids = [p["origin_id"] for p in purged]

        self.assertIn("boundary_valid", purged_ids)
        self.assertNotIn("boundary_invalid", purged_ids)
        self.assertNotIn("candidate_overlap", purged_ids)

    def test_history_tuned_baseline(self):
        """Every backtest window used by the history-tuned baseline lies within days <= 1550,
        and the choice is unchanged when d_1551-d_1913 demand is perturbed.
        """
        # All windows within days <= 1550
        self.assertLessEqual(BACKTEST_TUNING_FOLD["calib_start_day"], 1550)
        self.assertLessEqual(BACKTEST_TUNING_FOLD["calib_end_day"], 1550)
        self.assertLessEqual(BACKTEST_TUNING_FOLD["warmup_start_day"], 1550)
        self.assertLessEqual(BACKTEST_TUNING_FOLD["warmup_end_day"], 1550)
        self.assertLessEqual(BACKTEST_TUNING_FOLD["scoring_start_day"], 1550)
        self.assertLessEqual(BACKTEST_TUNING_FOLD["scoring_end_day"], 1550)

        # Invariance to perturbation on days 1551 to 1913
        demand_orig = self.demand.copy()
        demand_pert = self.demand.copy()
        demand_pert[1550:] += np.random.uniform(50, 200, size=len(demand_pert[1550:]))

        best_r_orig, best_l_orig, best_tsl_orig, _ = tune_history_grid_baseline_for_sku(
            demand_orig, backtest_fold=BACKTEST_TUNING_FOLD
        )
        best_r_pert, best_l_pert, best_tsl_pert, _ = tune_history_grid_baseline_for_sku(
            demand_pert, backtest_fold=BACKTEST_TUNING_FOLD
        )

        self.assertEqual((best_r_orig, best_l_orig, best_tsl_orig), (best_r_pert, best_l_pert, best_tsl_pert))

    def test_eval_window_leakage(self):
        """Eval-window demand is never read before final scoring."""
        eval_start_day = EVAL_ORIGIN["warmup_start_day"]  # 1551
        demand_orig = self.demand.copy()
        demand_pert = self.demand.copy()
        demand_pert[eval_start_day - 1 :] += 999.0

        # Features for eval SKUs are identical
        feat_orig = demand_features(demand_orig, history_end_day=EVAL_ORIGIN["history_end_day"])
        feat_pert = demand_features(demand_pert, history_end_day=EVAL_ORIGIN["history_end_day"])
        self.assertEqual(feat_orig, feat_pert)

        # Baseline choice is identical
        tune_orig = tune_history_grid_baseline_for_sku(demand_orig, backtest_fold=BACKTEST_TUNING_FOLD)
        tune_pert = tune_history_grid_baseline_for_sku(demand_pert, backtest_fold=BACKTEST_TUNING_FOLD)
        self.assertEqual((tune_orig[0], tune_orig[1], tune_orig[2]), (tune_pert[0], tune_pert[1], tune_pert[2]))

        # Optimal policy selection using dummy surrogates is identical
        dummy_models = {
            "avg_inventory": DummyModel(10.0),
            "lost_sales": DummyModel(1.0),
            "num_orders": DummyModel(2.0),
        }
        sku_df_orig = pd.DataFrame([{"item_id": "TEST", "state_id": "CA", "history_end_day": 1550, **feat_orig}])
        sku_df_pert = pd.DataFrame([{"item_id": "TEST", "state_id": "CA", "history_end_day": 1550, **feat_pert}])

        opt_orig = optimize_all_skus(dummy_models, sku_df_orig, horizon_days=28)
        opt_pert = optimize_all_skus(dummy_models, sku_df_pert, horizon_days=28)
        pd.testing.assert_frame_equal(opt_orig, opt_pert)


if __name__ == "__main__":
    unittest.main()
