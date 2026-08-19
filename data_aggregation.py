import os
import pandas as pd
import numpy as np

# --- 1. Constants and File Paths ---
data_dir = r'C:\Users\ARPIT SINGH\OneDrive\Desktop\BEProject\Dataset'

# Input files (assuming you are using the validation file first)
SALES_FILE = os.path.join(data_dir, 'sales_train_validation.csv')

# Output file
LEVEL_11_SALES_FILE = os.path.join(data_dir, 'sales_train_level_11.csv')

# --- 2. Data Loading and Downcasting (for memory efficiency) ---

def downcast_df(df):
    """Reduces memory usage by changing column dtypes."""
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df

print("1. Loading Sales Data...")
try:
    # Use the evaluation file if available, or validation if not.
    # We load the full sales file which has item_id, state_id, etc.
    sales_df = pd.read_csv(SALES_FILE)
    sales_df = downcast_df(sales_df)
except FileNotFoundError:
    print(f"Error: Sales file not found at {SALES_FILE}.")
    exit()

# --- 3. Aggregating to Product-State (Level 11) ---

print("2. Aggregating to Product-State (Level 11)...")

# Level 11 aggregation: Unit sales of product, aggregated for each state[cite: 339].
# The grouping keys are 'item_id' and 'state_id'.
grouping_keys = ['item_id', 'state_id']

# Identify the daily sales columns
day_cols = [col for col in sales_df.columns if col.startswith('d_')]

if not day_cols:
    print("Error: Could not identify daily sales columns ('d_1', 'd_2', etc.) in the input file.")
    exit()

# Perform the aggregation by summing the daily sales columns for each combination
# of item_id and state_id.
sales_agg_df = sales_df.groupby(grouping_keys)[day_cols].sum().reset_index()

# --- 4. Verification and Saving ---

num_series = len(sales_agg_df)
# The paper states Level 11 should have 9,147 series (3,049 products * 3 states)[cite: 339].
print(f"Aggregation Complete. Number of Level 11 series: {num_series}")

if num_series == 0:
    print("\nWARNING: Aggregation resulted in 0 series. Check the raw input file contents.")
    
sales_agg_df.to_csv(LEVEL_11_SALES_FILE, index=False)
print(f"Level 11 data saved to: {LEVEL_11_SALES_FILE}")

print("\n--- Next Step ---")
print("Now, re-run 'python feature_engineering.py'.")