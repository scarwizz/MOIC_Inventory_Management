"""Unit tests for feature engineering."""

import unittest
import numpy as np
import pandas as pd
from src.features import calculate_ts_features, engineer_surrogate_features, categorize_demand, FEATURE_COLUMNS


class TestFeatures(unittest.TestCase):
    def test_calculate_ts_features_constant(self):
        # Constant demand of 5 every day for 10 days
        series = np.array([5.0] * 10)
        res = calculate_ts_features(series)
        self.assertAlmostEqual(res["ADI"], 1.0)
        self.assertAlmostEqual(res["CV2"], 0.0)

    def test_calculate_ts_features_intermittent(self):
        # Demand every 2 days: [5, 0, 5, 0, 5, 0]
        series = np.array([5.0, 0.0, 5.0, 0.0, 5.0, 0.0])
        res = calculate_ts_features(series)
        self.assertAlmostEqual(res["ADI"], 2.0)
        self.assertAlmostEqual(res["CV2"], 0.0)

    def test_calculate_ts_features_zero_demand(self):
        # All zeros
        series = np.array([0.0, 0.0, 0.0])
        res = calculate_ts_features(series)
        self.assertEqual(res["ADI"], 3.0)
        self.assertEqual(res["CV2"], 0.0)

    def test_engineer_surrogate_features_schema(self):
        df = engineer_surrogate_features(
            adi=[1.5, 2.0],
            cv2=[0.5, 0.0],
            r=[7, 14],
            l=[3, 5],
            tsl=[0.95, 0.99],
        )
        self.assertListEqual(list(df.columns), FEATURE_COLUMNS)
        self.assertEqual(len(df), 2)
        # Check Risk_Period = R + L
        self.assertEqual(df.loc[0, "Risk_Period"], 10.0)
        self.assertEqual(df.loc[1, "Risk_Period"], 19.0)
        # Check Policy_Interaction = R * L
        self.assertEqual(df.loc[0, "Policy_Interaction"], 21.0)
        self.assertEqual(df.loc[1, "Policy_Interaction"], 70.0)
        # Check Demand_Profile = ADI / CV2 (or 0 if CV2==0)
        self.assertAlmostEqual(df.loc[0, "Demand_Profile"], 3.0)
        self.assertEqual(df.loc[1, "Demand_Profile"], 0.0)

    def test_categorize_demand(self):
        self.assertEqual(categorize_demand(1.1, 0.3), "Smooth")
        self.assertEqual(categorize_demand(1.5, 0.3), "Intermittent")
        self.assertEqual(categorize_demand(1.1, 0.6), "Erratic")
        self.assertEqual(categorize_demand(1.5, 0.6), "Lumpy")


if __name__ == "__main__":
    unittest.main()
