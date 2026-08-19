import os
import pandas as pd
import numpy as np

# --- 1. Constants and File Paths ---
# Use the corrected raw string path
data_dir = r'C:\Users\ARPIT SINGH\OneDrive\Desktop\BEProject\Dataset'

# File paths from previous steps
LEVEL_11_SALES_FILE = os.path.join(data_dir, 'sales_train_level_11.csv')
FEATURES_FILE = os.path.join(data_dir, 'm5_demand_features.csv')


# --- 2. Utility for Calculating ADI and CV² ---

def calculate_ts_features(series):
    """
    Calculates Average Inter-Demand Interval (ADI) and Squared Coefficient
    of Variation (CV²) for a time series.
    """
    # Filter for non-zero demand values (Di)
    demand = series[series > 0]
    
    # 1. ADI (Average Inter-Demand Interval)
    # The M5 data often contains intermittent and slow-moving items.
    # ADI = Total Periods / Number of Periods with Non-Zero Demand 
    
    total_periods = len(series)
    non_zero_periods = len(demand)
    
    # Handle the case where a series is all zeros to avoid division by zero
    if non_zero_periods == 0:
        adi = total_periods  # Maximum possible intermittency
        cv2 = 0.0            # No demand, so no variability in demand size
    else:
        # ADI Calculation
        adi = total_periods / non_zero_periods
        
        # 2. CV² (Squared Coefficient of Variation of Demand Sizes)
        # CV² = (Standard Deviation of Non-Zero Demand / Mean of Non-Zero Demand) ^ 2 
        
        # Calculate CV² using the standard formula for non-zero demand sizes
        mean_demand = demand.mean()
        std_demand = demand.std(ddof=1) # ddof=1 for sample standard deviation
        
        if mean_demand == 0:
            cv2 = 0.0 # Should be covered by non_zero_periods check, but for safety
        else:
            cv2 = (std_demand / mean_demand) ** 2

    return pd.Series({'ADI': adi, 'CV2': cv2})


# --- 3. Main Execution for Feature Calculation ---

print("4. Calculating ADI and CV² Time Series Features...")

# Load the aggregated data (Level 11: Product-State)
try:
    sales_df = pd.read_csv(LEVEL_11_SALES_FILE)
except FileNotFoundError:
    print(f"Error: Aggregated sales file not found at {LEVEL_11_SALES_FILE}.")
    # IMPORTANT: If the previous step failed (as indicated by your output of 0 series), 
    # you must re-run the aggregation step before this one can succeed.
    exit()

# Identify the columns containing daily sales ('d_1', 'd_2', etc.)
# We assume the columns after the ID column are the sales days
# In M5 data, the days start with d_1 and go up to d_1913/d_1941
day_cols = [col for col in sales_df.columns if col.startswith('d_')]

if not day_cols:
    print("Error: No 'd_' sales columns found. Check the aggregation step output.")
    exit()

# Apply the feature calculation function row-wise (axis=1) to the sales columns
# The ID columns are expected to be the first few columns (e.g., 'item_id', 'state_id')
id_cols = sales_df.columns.drop(day_cols)
features_df = sales_df[id_cols].copy()

# Apply the feature calculation function
ts_features = sales_df[day_cols].apply(calculate_ts_features, axis=1)

# Merge the new features with the ID columns
features_df = pd.concat([features_df, ts_features], axis=1)

# --- 4. Save the Features Data ---
features_df.to_csv(FEATURES_FILE, index=False)
print(f"Feature calculation complete. Number of series: {len(features_df)}")
print(f"Time series features (ADI and CV²) saved to: {FEATURES_FILE}\n")

print("5. Next Step: Stock Control Simulations (Off-line Phase Step 5)...")