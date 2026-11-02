# MOIC Inventory Management — Audit & v2 Changelog

**Date:** November 2026  
**Status:** Audit Completed & Corrected Pipeline Validated (v2.0.0)  
**Historical Notebook:** [`notebooks/MAIN_NOTEBOOK.ipynb`](../notebooks/MAIN_NOTEBOOK.ipynb) (Preserved as historical record)  
**Corrected Implementation:** [`src/simulator.py`](../src/simulator.py), [`src/training.py`](../src/training.py), [`src/evaluation.py`](../src/evaluation.py)  

---

## 1. Executive Summary

A forensic audit of the Model-Independent Optimal Inventory Control (MOIC) pipeline identified a foundational bug in the discrete-event simulation functions of the original Kaggle notebook (`MAIN_NOTEBOOK.ipynb` Cells 1, 8, and 12). 

In the original implementation, replenishment orders were calculated and fixed ordering costs were accrued, but **ordered quantities were never added back to on-hand inventory**. As a consequence, all 686,025 training labels and hold-out evaluation runs were derived from a non-replenishing system where stock monotonically depleted toward zero.

This artifact distorted the objective function, caused artificial boundary collapse in policy optimization, and produced extreme baseline numbers. The entire pipeline has been redesigned, retrained, and re-evaluated under a verified, conservation-checked discrete-event simulator.

---

## 2. Root Cause Analysis

### The Flaw
In the original simulation functions:
```python
# Flawed logic from MAIN_NOTEBOOK.ipynb Cells 1, 8, 12
if current_inventory <= s:
    orders_placed += 1
    # order_quantity was calculated, but NEVER added to current_inventory!
```
Orders placed at review intervals were recorded in the order counter, but no arrival schedule or pipeline queue existed. Stock never arrived at day $t + L$.

### Systemic Impacts
1. **Initial Buffer Dominance**:
   Initial inventory was set to the order-up-to level $S = \hat{D}(R+L) + z\sigma_D\sqrt{R+L}$. Without subsequent replenishment, policies with maximum review period ($R=30$) and maximum lead time ($L=30$) received huge initial buffers ($R+L = 60$ days of demand) that buffered demand across the simulation horizon.
2. **Boundary Collapse**:
   Policies with realistic, short review periods (e.g., $R=7, L=3$) ran out of initial stock in 10 days and suffered massive unreplenished lost sales for the remaining days. Consequently:
   - **CatBoost** selected $(R=30, L=30, \text{TSL}=0.99)$ for **100% of 9,147 SKUs**.
   - **LightGBM** selected $(R=30, L=30)$ for **98.3% of SKUs**.
3. **The "43% Advantage" Artifact**:
   The 43% CatBoost cost advantage originated from comparison under distorted boundary conditions. When evaluated under true replenishment, CatBoost and LightGBM perform within **0.61%** of each other.
4. **Distorted Heuristic Baseline (FSH)**:
   The Fixed Standard Heuristic ($R=7, L=3, \text{TSL}=0.95$) suffered an artificial unit fill rate of 51.6% and an inflated cost of $21.31 in v1 solely because its replenishment orders were lost. With replenishment, FSH achieves a **94.27%** fill rate at a cost of **$4.23**.

---

## 3. What Was Corrected

1. **Discrete-Event Ground Truth Simulator ([`src/simulator.py`](../src/simulator.py))**:
   - Explicit daily event sequence: (1) Order receipts at start of day $t$, (2) Review & pipeline-aware Inventory Position ($IP = \text{On-Hand} + \text{On-Order}$), (3) Order release to arrival schedule at $t+L$, (4) Customer demand fulfillment and lost sales tracking.
   - Built-in **stock-flow conservation test** on every simulation run:
     $$\text{Ending Inventory} = \text{Initial Inventory} + \text{Total Receipts} - \text{Total Satisfied Demand}$$
2. **Test Suite Verification ([`tests/`](../tests/))**:
   - 9 unit and integration tests passing in 0.01s:
     - Stock-flow conservation across stochastic scenarios
     - Exact $t+L$ receipt timing
     - Zero-demand constant inventory preservation
     - Pipeline arrival under zero demand
     - Deterministic hand-calculated 6-day mathematical trace match
3. **Training Label Regeneration ([`scripts/02_generate_labels.py`](../scripts/02_generate_labels.py))**:
   - Regenerated all **686,025 simulations** (9,147 SKUs $\times$ 75 policies) over 365 days.
   - **0 conservation failures** across 686,025 runs.
4. **Surrogate Model Retraining ([`scripts/03_train_surrogates.py`](../scripts/03_train_surrogates.py))**:
   - Retrained LightGBM and CatBoost (CPU MAE) on corrected 80/20 split with 8 production features.
5. **Fair Evaluation with Warm-Up ([`scripts/05_evaluate.py`](../scripts/05_evaluate.py))**:
   - Protocol: 365-day history ($d_{1186}$–$d_{1550}$) for parameter calibration, 60-day warm-up ($d_{1551}$–$d_{1610}$) to initialize pipeline stock, and 28-day evaluation ($d_{1611}$–$d_{1638}$).
   - Evaluated under both primary parameters ($h=0.01/\text{yr}, b=\$0.25, k=\$0.50$) and realistic retail holding sensitivity ($h=0.20/\text{yr}$).

---

## 4. Empirical Comparison: v1 (Flawed) vs. v2 (Corrected)

### A. Surrogate Model Validation Error (MAE)

| Model | Target Metric | v1 MAE (Flawed Sim) | v2 MAE (Corrected Sim) | Error Change |
| :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | `AvgInventory` | 7.67 | 29.62 | Refined scale |
| **LightGBM** | `LostSales` | 659.70 | **52.44** | **-92.1% error** |
| **LightGBM** | `NumOrders` | 0.0007 | 11.18 | Reflects ongoing orders |
| **CatBoost** | `AvgInventory` | 9.52 | 34.34 | Refined scale |
| **CatBoost** | `LostSales` | 1,091.70 | **61.78** | **-94.3% error** |
| **CatBoost** | `NumOrders` | 66.90 (GPU bug) | **11.76** (CPU) | **-82.4% error** |

### B. Policy Selection Distribution (9,147 SKUs)

| Characteristic | v1 (Flawed Simulator) | v2 (Corrected Simulator) | Interpretation |
| :--- | :--- | :--- | :--- |
| **CatBoost Boundary Collapse** | **100.0%** ($R=30, L=30, \text{TSL}=0.99$) | **0.0%** (0 / 9,147 SKUs) | Collapse completely cured |
| **LightGBM Boundary Collapse** | **98.3%** ($R=30, L=30$) | **0.4%** (39 / 9,147 SKUs) | Collapse completely cured |
| **Selected Lead Time ($L=1$)** | 0.0% | **89.2% (CATB)** / **74.2% (LGBM)** | Fast replenishment preferred |
| **Identical Policy Overlap** | 82.1% (all boundary) | 67.7% | Distinct, SKU-tailored policies |

### C. Hold-Out Evaluation Results (28-day window, 9,147 SKUs)

| Method / Policy | v1 Cost (Claimed) | v2 Cost (Measured) | v1 Fill Rate (Claimed) | v2 Unit Fill Rate (Measured) |
| :--- | :--- | :--- | :--- | :--- |
| **MOIC CatBoost** | \$0.9206* | **\$2.82** | 99.4% | **95.01%** |
| **MOIC LightGBM** | \$1.6850* | **\$2.84** | ~99% | **94.92%** |
| **True Optimal (Oracle)** | \$0.9206* | **\$1.34** | ~100% | **98.20%** |
| **Fixed Standard Heuristic (FSH)** | \$21.3093 | **\$4.23** | 51.6% | **94.27%** |

*\*Note: v1 costs were raw dollar figures under non-replenishing initial buffers, not "normalized cost".*

### D. Key Metric Comparisons

1. **CatBoost vs. LightGBM**:
   - v1 Claim: *"CatBoost policies achieve 43% lower total cost than LightGBM."*
   - v2 Fact: Relative cost reduction is **0.61%** (\$2.8208 vs. \$2.8381). Median cost is identical (\$0.5571). CatBoost is strictly cheaper on 18.5% of SKUs; LightGBM is strictly cheaper on 13.8% of SKUs.
2. **MOIC vs. Fixed Standard Heuristic (FSH)**:
   - v1 Claim: *"Normalized inventory cost reduced from 21.3 to 0.92 (-95.7%)."*
   - v2 Fact: MOIC achieves a **33.3% cost reduction** compared to FSH (\$2.82 vs. \$4.23) while maintaining a **95.01% unit fill rate** (vs. 94.27% for FSH).
3. **Retail Holding Cost Sensitivity ($h = 0.20/\text{year}$)**:
   - CatBoost Mean Cost: **\$4.19**
   - LightGBM Mean Cost: **\$4.22** (0.69% difference)
   - FSH Mean Cost: **\$4.63** (MOIC provides a 9.4% cost advantage under high holding costs)

---

## 5. Supported Resume & Portfolio Claims

### Statements That Are Supported:
- *"Formulated a Model-Independent Optimal Inventory Control (MOIC) framework on 9,147 Walmart M5 time series, training LightGBM and CatBoost surrogate models to predict multi-echelon cost elements."*
- *"Built a discrete-event inventory simulation engine with strict stock-flow conservation verification, pipeline order tracking, and warm-up periods."*
- *"Demonstrated a **33.3% total cost reduction** over standard $(R=7, L=3)$ fixed heuristics while maintaining a **95.0% unit fill rate** on held-out demand."*
- *"Demonstrated that gradient-boosted surrogate models achieve <$62 unit MAE on lost sales across 686k simulation episodes."*

### Statements That Must Be Removed:
- *"CatBoost-selected inventory policies have 43% lower total cost than LightGBM-selected policies."* $\rightarrow$ **Remove**. The advantage is 0.61% and was an artifact of missing replenishment arrivals.
- *"Normalized cost reduced from 21.3 to 0.92."* $\rightarrow$ **Remove**. Cost was never normalized, and the v1 baseline was artificially crippled by non-arrival of stock.
- *"99.4% service level vs 51.6% for FSH."* $\rightarrow$ **Replace** with *"95.0% unit fill rate vs 94.3% for FSH"*.
