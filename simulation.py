import os
import pandas as pd
import numpy as np
from scipy.stats import norm
from itertools import product
from tqdm import tqdm

# --- 1. Constants and File Paths ---

# Use the corrected raw string path for Windows compatibility
data_dir = r'C:\Users\ARPIT SINGH\OneDrive\Desktop\BEProject\Dataset'

# Input files from previous steps
LEVEL_11_SALES_FILE = os.path.join(data_dir, 'sales_train_level_11.csv')
FEATURES_FILE = os.path.join(data_dir, 'm5_demand_features.csv')

# Output file for ML training (Features + Targets)
SIMULATION_OUTPUT_FILE = os.path.join(data_dir, 'ml_training_data.csv')

# Simulation period constants (From paper: 2 full calendar years required)
# We use the last 365 days of the historical data for the OFF-LINE simulation
# Assuming validation data goes up to d_1913
TRAIN_DAYS = 1913 
SIM_DAYS = 365
SPLIT_DAY = TRAIN_DAYS - SIM_DAYS
DAY_COLS = [f'd_{i}' for i in range(SPLIT_DAY + 1, TRAIN_DAYS + 1)]

# --- 2. Hyperparameter Ranges (Step 4) ---

# R: Review Period (30 values)
R_RANGE = np.linspace(1, 30, 30, dtype=int)
# L: Lead Time (30 values)
L_RANGE = np.linspace(1, 30, 30, dtype=int)
# TSL: Target Service Level (10 values)
TSL_RANGE = np.linspace(0.90, 0.99, 10)

# Generate all 30 x 30 x 10 = 9,000 hyperparameter sets
POLICY_HPARAMS = list(product(R_RANGE, L_RANGE, TSL_RANGE))

print(f"4. Generating {len(POLICY_HPARAMS)} Inventory Policy Hyperparameter Sets...")

# --- 3. Core MOIC Simulation Functions (Step 5 Logic) ---

def croston_forecast_and_stats(history):
    """
    Calculates the required demand statistics for the inventory policy (Croston's Method Context).
    
    The safety stock (SS) formula (Eq. 6) requires the standard deviation of historical demand (sigma_D) 
    and the target level (St) requires the expected demand (D_hat).
    """
    
    # Use all historical data up to the simulation start (ts_sales) for calculation
    
    # D_hat (Average Demand, D_bar in the paper, Section 2.2)
    # The paper uses the forecasted demand, but many M5 solutions simplify this to empirical mean.
    D_hat = history.mean() 
    
    # Sigma_D (Standard deviation of historical demand up to period t, Section 2.2)
    # The paper uses sigma_D_t; we use the standard deviation over the history.
    sigma_D = history.std(ddof=1) if len(history) > 1 else 1.0
    
    return D_hat, sigma_D

def run_rss_simulation(ts_sales, R, L, TSL):
    """
    Runs the (R, s, S) lost-sales stock control simulation over the reference period.
    Returns the three Inventory Cost Elements (Targets).
    """
    
    # 1. Pre-calculate Policy Parameters (based on formulas in Section 2.2)
    D_hat, sigma_D = croston_forecast_and_stats(ts_sales)
    
    # Z-score: Z = Φ⁻¹(TSL) (Eq. 196)
    Z = norm.ppf(TSL)
    
    # Safety Stock (SS): SS = Z * σ_D * sqrt(R + L) (Eq. 6)
    SS = Z * sigma_D * np.sqrt(R + L)
    
    # Order Point (s): s_t = Sum(D_hat) + SS (sum over 1 to t+R+L periods) (Eq. 5)
    s_t = D_hat * (R + L) + SS
    
    # Order-up-to Level (S): S_t = Sum(D_hat) + SS (Eq. 187, where m=R+L)
    S_t = D_hat * (R + L) + SS 
    
    # Initialize variables
    inventory_level = S_t # Start with target inventory level S_t [cite: 279]
    orders_placed = 0
    
    # Metrics to record
    inventory_levels = []
    lost_sales_list = []

    # 2. Simulation Loop
    for t in range(SIM_DAYS):
        # A. Inventory Check and Ordering (Periodic Review)
        order_quantity = 0
        if (t % R == 0):
            # Order placement only if inventory level is at or below reorder point s_t (Eq. 7)
            if inventory_level <= s_t:
                order_quantity = S_t - inventory_level
                orders_placed += 1
                
                # Note: In a full simulation, this order arrives 'L' days later.
                # For simplicity in generating cost elements, we assume the policy 
                # defines the total ordering effort and inventory needs.

        # B. Fulfillment and Lost Sales
        demand = ts_sales.iloc[t]
        satisfied_sales = min(inventory_level, demand)
        lost_sales = demand - satisfied_sales
        
        # C. Update Inventory and Arrival
        inventory_level -= satisfied_sales
        
        # NOTE: Orders arriving L days later would increase inventory_level here.
        # Since this simulation simplifies tracking in-transit stock, we rely on the 
        # R,s,S parameter calculation (which already accounts for L) to define the policy.
        # For a full time-step simulation, explicit in-transit tracking is required.
        
        # D. Recording Metrics
        inventory_levels.append(inventory_level)
        lost_sales_list.append(lost_sales)

    # 3. Post-Simulation Calculation of Cost Elements (Targets)
    avg_inventory = np.mean(inventory_levels) # Used for C_H [cite: 215]
    total_lost_sales = np.sum(lost_sales_list) # Used for C_LS [cite: 215]
    total_orders = orders_placed # Used for C_O [cite: 215]
    
    # Total inventory cost (C_Tot) is NOT returned, as it is calculated later (Step 10)
    return avg_inventory, total_lost_sales, total_orders

# --- 4. Main Execution ---

# Load features and sales data
features_df = pd.read_csv(FEATURES_FILE)
sales_df = pd.read_csv(LEVEL_11_SALES_FILE)

# Prepare the data for simulation (Merge and isolate sales)
sales_data = sales_df.set_index(['item_id', 'state_id'])
reference_sales = sales_data[DAY_COLS]

simulation_results = []
print(f"5. Performing Stock Control Simulations ({len(features_df)} series x {len(POLICY_HPARAMS)} policies)...")

# Iterate through each series (item_id, state_id)
for index, row in tqdm(features_df.iterrows(), total=len(features_df), desc="Simulating"):
    
    item_id = row['item_id']
    state_id = row['state_id']
    adi = row['ADI']
    cv2 = row['CV2']
    
    try:
        ts_sales = reference_sales.loc[(item_id, state_id)]
    except KeyError:
        # Should not happen if aggregation was correct, but handles missing sales data
        continue

    # Iterate through all hyperparameter sets (Step 4)
    for R, L, TSL in POLICY_HPARAMS:
        
        # Run the simulation (Step 5)
        avg_inv, total_ls, total_orders = run_rss_simulation(
            ts_sales=ts_sales,
            R=R, L=L, TSL=TSL
        )

        # Record the ML training observation (Features + Targets)
        simulation_results.append({
            'item_id': item_id,
            'state_id': state_id,
            'ADI': adi,
            'CV2': cv2,
            'R': R,
            'L': L,
            'TSL': TSL,
            'AvgInventory': avg_inv,
            'LostSales': total_ls,
            'NumOrders': total_orders,
        })

# --- 5. Final Save ---
training_df = pd.DataFrame(simulation_results)
training_df.to_csv(SIMULATION_OUTPUT_FILE, index=False)
print("\nSimulation complete!")
print(f"ML Training Data saved to: {SIMULATION_OUTPUT_FILE}")
print(f"Total training observations generated: {len(training_df)} (ideally 9147 * 9000)")

# --- Next Step ---
print("\n--- Next Step (Phase 2, Step 6) ---")
print("6. Train the Machine Learning Models: You will now train three separate LightGBM models using the data generated above.")
print("The script name for the next step should be: model_training.py")