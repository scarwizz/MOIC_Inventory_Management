# MOIC Inventory Management — Data Directory

This directory manages datasets across raw inputs and processed pipeline artifacts.

## Structure

```
data/
├── README.md               # This documentation file
├── raw/                    # Raw M5 competition datasets (gitignored)
│   ├── sales_train_validation.csv
│   ├── calendar.csv
│   └── sell_prices.csv
└── processed/              # Derived and simulated artifacts
    ├── sales_train_level_11.csv               # Aggregated product-state level (9,147 series)
    ├── m5_demand_features.csv                 # ADI and CV2 time-series features
    ├── ml_training_data_v2_corrected.csv      # Corrected simulation training labels (686,025 rows)
    ├── optimal_policies_v2_LGBM.csv           # Selected optimal policies via v2 LightGBM
    ├── optimal_policies_v2_CATB.csv           # Selected optimal policies via v2 CatBoost
    ├── optimal_policies_v2_XGBM.csv           # Selected optimal policies via v2 XGBoost
    ├── evaluation_results_v2.csv              # Full hold-out evaluation results
    └── evaluation_summary_v2.json             # Aggregate metrics and comparisons
```

## Data Sources

The raw data comes from the **M5 Forecasting - Accuracy** competition hosted on Kaggle:
- `sales_train_validation.csv`: Daily unit sales from day `d_1` to `d_1913` across 30,490 items and 10 stores.
- `calendar.csv`: Dates, event types, and SNAP indicators.
- `sell_prices.csv`: Store-item-week selling prices.

Note: Local datasets currently exist in `Dataset/` for backwards compatibility. Pipeline scripts automatically check both `data/raw/` / `data/processed/` and `Dataset/`.

## Aggregation (Level 11)
The M5 hierarchical level 11 aggregates item sales across stores within each state (CA, TX, WI):
- 3,049 items × 3 states = 9,147 time series.
- Columns: `item_id`, `state_id`, `d_1` ... `d_1913`.
