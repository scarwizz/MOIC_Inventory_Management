import pandas as pd
import numpy as np
import os

# --- Configuration: Define File Paths ---
# IMPORTANT: Update this path to where your M5 files are located.
data_dir = r'C:\Users\ARPIT SINGH\OneDrive\Desktop\BEProject\Dataset' 

SALES_FILE = os.path.join(data_dir, 'sales_train_validation.csv')
CALENDAR_FILE = os.path.join(data_dir, 'calendar.csv')
PRICES_FILE = os.path.join(data_dir, 'sell_prices.csv')

# --- 1. Load and Prepare Calendar Data ---
# This is needed to map store_id to state_id and get date columns.
print("1. Loading Calendar Data...")
calendar_df = pd.read_csv(CALENDAR_FILE)

# Create a mapping dictionary for date columns (d_1, d_2, ...) to actual dates
date_cols = [col for col in calendar_df.columns if col.startswith('d_')]
date_map = calendar_df[['d', 'date']].set_index('d')['date'].to_dict()

# --- 2. Load and Prepare Sales Data ---
print("2. Loading Sales Data...")
# Load only the necessary columns to save memory
cols_to_use = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'] + date_cols
sales_df = pd.read_csv(SALES_FILE, usecols=cols_to_use)

# --- 3. Isolate the Product-State Aggregation Level (Level 11) ---
print("3. Aggregating to Product-State (Level 11)...")

# The aggregation is done by grouping all individual item sales
# by their unique 'item_id' and 'state_id' and then summing the sales for each day.
# The `id` and other columns ('dept_id', 'cat_id', 'store_id') are dropped 
# as they are not needed for this aggregation level.

# 3a. Melt the DataFrame to long format for easier aggregation
# This converts the daily sales columns (d_1, d_2, ...) into a single 'sales' column.
sales_long = sales_df.melt(
    id_vars=['item_id', 'state_id'], 
    value_vars=date_cols, 
    var_name='d', 
    value_name='sales'
)

# 3b. Perform the core aggregation
# Sum the 'sales' for each unique combination of 'item_id' and 'state_id'.
level_11_df = sales_long.groupby(['item_id', 'state_id', 'd'])['sales'].sum().reset_index()

# --- 4. Pivot Back to Wide Format and Final Cleanup ---

# 4a. Pivot back to the wide format where each row is a time series
# and columns are the daily sales (d_1, d_2, ...)
level_11_pivot = level_11_df.pivot(
    index=['item_id', 'state_id'], 
    columns='d', 
    values='sales'
).reset_index()

# 4b. Create the new unique series ID for Level 11
# Format: ITEM_ID_STATE_ID (e.g., HOBBIES_1_001_CA)
level_11_pivot['id'] = level_11_pivot['item_id'] + '_' + level_11_pivot['state_id']

# 4c. Select and reorder columns (to match typical sales file structure)
level_11_cols = ['id', 'item_id', 'state_id'] + date_cols
level_11_final = level_11_pivot[level_11_cols]

print(f"Aggregation Complete. Number of Level 11 series: {len(level_11_final)}")
# The paper states 9,147 series (3,049 products * 3 states)[cite: 338, 327]. 
# The aggregated data should match this count.

# --- 5. Save the Result ---
OUTPUT_FILE = os.path.join(data_dir, 'sales_train_level_11.csv')
level_11_final.to_csv(OUTPUT_FILE, index=False)
print(f"Level 11 data saved to: {OUTPUT_FILE}")

# --- Next Steps for Your Project ---
# You'll also need the price data aligned with this new structure.
print("\nLoading and Preparing Price Data...")
prices_df = pd.read_csv(PRICES_FILE)

# The price is a feature of the item and store, so for Level 11 (state-level), 
# you'll need to decide how to handle the price. The paper uses the item price (p_i) 
# which is constant for a given state-day, so you can likely merge the original 
# prices data to your new Level 11 series.

# You will proceed with the filtering and feature calculation (Step 2 & 3) 
# using the 'level_11_final' DataFrame.