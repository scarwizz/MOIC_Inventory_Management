# MOIC Supply Chain Intelligence Agent

> **A Neuro-Symbolic AI decision engine that eliminates inventory guesswork — powered by LightGBM surrogate models, a LangGraph ReAct agent, and the MOIC framework.**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.61-FF4B4B?logo=streamlit)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2.11-0D1117)](https://github.com/langchain-ai/langgraph)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-blue)](https://lightgbm.readthedocs.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.10-yellow)](https://catboost.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What Is This?

The **MOIC (Model-Independent Optimal Inventory Control) Supply Chain Agent** is a dual-identity system that combines a full Machine Learning pipeline with an AI Agent orchestration layer:

1. **ML Pipeline** — Runs stock-flow conservation verified (R, s, S) inventory policy simulations over 2,058,075 SKU-policy observations across 3 causal origins from the M5 Walmart dataset, trains LightGBM and CatBoost surrogate models on 8 engineered features, and exports them as serialised predictors.
2. **AI Agent** — Wraps the surrogates inside a LangGraph ReAct agent (Llama 3.3-70b on Groq) so supply chain managers can query inventory costs in natural language and receive grounded, hallucination-proof answers.
3. **Streamlit UI** — Enterprise dark-mode dashboard that renders the full agent execution trace in real time.

---

## Table of Contents

- [ML Pipeline & Analytics Deep-Dive](#ml-pipeline--analytics-deep-dive)
  - [Problem Formulation & Dataset](#1-problem-formulation--dataset)
  - [Feature Engineering](#2-feature-engineering)
  - [Model Selection & Training](#3-model-selection--training)
  - [Evaluation Metrics & Results](#4-evaluation-metrics--results)
  - [Inference & Serving](#5-inference--serving)
- [AI Agent System Architecture](#ai-agent-system-architecture)
  - [Agent Logic & Loop](#6-agent-logic--loop)
  - [Tooling & Context Strategy](#7-tooling--context-strategy)
  - [Prompt Engineering & Safety](#8-prompt-engineering--safety)
- [ML + AI Agent Synergy](#ml--ai-agent-synergy)
  - [Interactive Workflow](#9-interactive-workflow)
  - [Mermaid Architecture Diagram](#10-end-to-end-mermaid-diagram)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Local Installation](#local-installation)
- [Reproducibility & Pipeline Execution](#reproducibility--pipeline-execution)
- [Running the App](#running-the-app)
- [Environment Variables](#environment-variables)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## ML Pipeline & Analytics Deep-Dive

### 1. Problem Formulation & Dataset

**Business Problem:**
Given a SKU's historical demand profile and an inventory policy (R, L, TSL), predict the three cost-driving outcomes — average inventory level, total lost sales, and number of orders placed — without re-running a full stochastic simulation.

**Dataset: M5 Walmart Forecasting (Kaggle)**

| Property | Value |
|----------|-------|
| Source | Walmart M5 Forecasting Accuracy Competition |
| Raw series | 30,490 item-store time series |
| Aggregation level | Level 11 (item × state): **9,147 series** |
| Total history length | 1,913 days (~5.25 years) |
| Training / Validation Origins | `train_921` (scoring d_982–d_1009), `train_1009` (scoring d_1070–d_1097), `val_1097` (scoring d_1158–d_1185) |
| Out-of-Sample Eval Origin | `eval_1550` (calib d_1186–d_1550, warmup d_1551–d_1610, scoring d_1611–d_1638) |
| Policy grid | R ∈ [1,30], L ∈ [1,30], TSL ∈ {0.90, 0.95, 0.99} → **75 policies** |
| Total Simulation Observations | 9,147 × 75 × 3 origins = **2,058,075 rows** (1,372,050 train / 686,025 val) |

**Preprocessing & Strict Causality Protocol:**
1. **Aggregation** — Item-level store sales grouped by `(item_id, state_id)` to eliminate store-level noise.
2. **Downcast** — Integer and float columns reduced to the smallest safe dtype (`int16`/`float32`) to reduce memory.
3. **Causal Demand Statistics** — Per-series ADI and CV² computed strictly over history $\le \text{history\_end\_day}$ for each origin (evaluation features strictly as of day 1550).
4. **Zero-Leakage Guarantee** — All training and validation simulation target windows finish by day 1185 or earlier. Evaluation calibration starts on day 1186. Not a single training row or feature sees day 1186+.
5. **Discrete-Event Simulation Labels** — For each SKU-policy observation across the 3 causal origins, a stock-flow conservation verified (R, s, S) discrete-event simulator produces the three regression targets over a 60-day warm-up and 28-day scoring horizon.

---

### 2. Feature Engineering

The final model uses **8 features** — 5 raw inputs and 3 engineered interaction terms:

| Feature | Formula | Domain | Interpretation |
|---------|---------|--------|---------------|
| `ADI` | total\_periods / non\_zero\_periods | [1, ∞) | Demand intermittency — higher = rarer demand |
| `CV2` | (σ / μ)² of non-zero demand | [0, ∞) | Demand size variability |
| `R` | Review period (days) | [1, 30] | How often we check stock |
| `L` | Lead time (days) | [1, 30] | Replenishment delay |
| `TSL` | Target Service Level | {0.90, 0.95, 0.99} | Risk tolerance |
| `Risk_Period` | R + L | [2, 60] | Total replenishment exposure window |
| `Policy_Interaction` | R × L | [1, 900] | Non-linear policy complexity term |
| `Demand_Profile` | ADI / CV2 (0 if CV2=0) | [0, ∞) | Demand regularity ratio |

The three engineered features are derived from the MOIC paper's theoretical framework and reduce LightGBM's prediction error by approximately 13% compared to the 5-feature baseline.

---

### 3. Model Selection & Training

**Models Evaluated:**

| Model | Objective | Strengths for this task |
|-------|-----------|------------------------|
| **LightGBM** (selected primary) | `regression_l1` (MAE) | Fastest inference, best MAE on AvgInventory and NumOrders, leaf-wise growth captures non-linear policy interactions |
| **CatBoost** (selected ensemble) | `MAE` (CPU mode) | Robust symmetric trees for policy optimisation; CPU mode avoids GPU MAE target integer bug |
| XGBoost | `reg:absoluteerror` | Evaluated but underperformed on intermittent demand series |

**Final Hyperparameters:**

```python
# LightGBM (GPU if available, CPU fallback)
lgb_params = {
    "objective":        "regression_l1",
    "learning_rate":    0.05,
    "num_leaves":       64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     1,
    "n_estimators":     500,
    "seed":             42,
}

# CatBoost (Forced CPU — avoids GPU MAE metric bug on integer targets)
catb_params = {
    "loss_function":  "MAE",
    "learning_rate":  0.05,
    "depth":          6,
    "n_estimators":   500,
    "subsample":      0.8,
    "bootstrap_type": "Bernoulli",
    "task_type":      "CPU",
    "random_seed":    42,
}
```

**Training Infrastructure & Validation Protocol:**
- Kaggle Notebook with **P100 GPU** (LightGBM GPU mode via OpenCL) / CPU fallback
- CatBoost trained on CPU (loss function `MAE`)
- Strict chronological validation: 1,372,050 training rows (`train_921`, `train_1009`) and 686,025 validation rows (`val_1097`) with non-overlapping 88-day target windows and purged boundary labels
- Grouped-SKU held-out evaluation: 80/20 SKU split (7,318 train SKUs / 1,829 unseen test SKUs)
- Three separate models per framework, one per target variable (`AvgInventory`, `LostSales`, `NumOrders`)

---

### 4. Evaluation Metrics & Results

All surrogate metrics are computed on the **chronological validation set (`val_1097`: 686,025 rows)** and the **grouped-SKU holdout (1,829 unseen SKUs)** using labels generated by the conservation-verified discrete-event simulator over an aligned 28-day scoring horizon. Full audit and fix details are documented in [`docs/AUDIT_CHANGELOG.md`](docs/AUDIT_CHANGELOG.md).

#### Surrogate Model Validation Metrics

Evaluated on the chronological validation set (`val_1097`, 686,025 rows) and grouped-SKU held-out set (1,829 unseen SKUs):

| Model | Target Metric | Chronological Val MAE | Chronological Val RMSE | Chronological $R^2$ | Held-out SKU MAE | Held-out SKU $R^2$ |
|-------|---------------|-----------------------|------------------------|---------------------|------------------|-------------------|
| **LightGBM** | `AvgInventory` | **42.38** | **169.23** | **0.2032** | 39.00 | 0.3386 |
| **LightGBM** | `LostSales` | **13.86** | **65.83** | -0.0320 | 13.25 | -0.0597 |
| **LightGBM** | `NumOrders` | **1.24** | **3.41** | **0.7809** | 1.21 | 0.7961 |
| **CatBoost** | `AvgInventory` | 43.15 | 172.14 | 0.1756 | 40.06 | 0.2922 |
| **CatBoost** | `LostSales` | 13.52 | 65.89 | -0.0341 | 12.93 | -0.0602 |
| **CatBoost** | `NumOrders` | 1.27 | 3.46 | 0.7748 | 1.23 | 0.7906 |

*(Note: Validation targets are scored over a 28-day window matching the scoring horizon, so NumOrders and LostSales scales reflect 28 days of replenishment).*

#### Clean Out-of-Sample Evaluation Results (9,147 Level 11 SKUs)

Policies were selected using surrogate predictions strictly as of day 1550 and evaluated on the hold-out scoring window (days 1611–1638) alongside heuristic baselines, a per-SKU history-tuned baseline (HTB), and an empirical hindsight oracle:

**Primary Cost Parameters ($h_{\text{annual}} = 0.01, b = \$0.25, k = \$0.50$):**

| Policy / Selection Method | Mean Total Cost ($) | Median Cost ($) | Std Cost ($) | Unit Fill Rate | Mean SKU Fill Rate | % SKUs Beating FSH | % SKUs Beating HTB |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MOIC LightGBM** | **$4.81** | **$0.55** | $16.84 | **85.94%** | 91.00% | **73.25%** | 28.77% |
| **MOIC CatBoost** | **$4.91** | **$0.55** | $17.06 | **85.64%** | 90.88% | **72.75%** | 28.58% |
| **History-Tuned Baseline (HTB)** | $3.96 | $0.56 | $14.33 | 89.52% | 92.60% | 76.54% | — |
| **Fixed Standard Heuristic (FSH)** | $4.23 | $2.02 | $9.50 | 92.14% | 94.27% | — | 23.46% |
| **Hindsight Grid Oracle (ORACLE)** | $1.34 | $0.52 | $6.07 | 97.34% | 98.20% | 85.59% | 74.38% |

*(Unit Fill Rate is defined as total units satisfied / total units demanded across the 28-day evaluation window. Hindsight Oracle represents the empirical best configuration from grid search in hindsight across the 75 policies).*

#### Paired Bootstrap 95% Confidence Intervals (Mean Cost Differences across SKUs)

- **CatBoost vs. LightGBM**: Mean Diff = $+0.0936 \pm 0.0389$ | 95% CI: $[+\$0.0559, +\$0.1338]$
- **LightGBM vs. FSH**: Mean Diff = $+0.5863 \pm 0.1845$ | 95% CI: $[+\$0.4044, +\$0.7734]$
- **LightGBM vs. HTB**: Mean Diff = $+0.8551 \pm 0.1619$ | 95% CI: $[+\$0.6962, +\$1.0201]$
- **CatBoost vs. FSH**: Mean Diff = $+0.6799 \pm 0.1893$ | 95% CI: $[+\$0.4927, +\$0.8713]$
- **CatBoost vs. HTB**: Mean Diff = $+0.9487 \pm 0.1619$ | 95% CI: $[+\$0.7920, +\$1.1158]$

#### Retail Holding Cost Sensitivity ($h_{\text{annual}} = 0.20, b = \$0.25, k = \$0.50$)

| Policy / Selection Method | Mean Total Cost ($) | Median Cost ($) | Std Cost ($) | Unit Fill Rate | Mean SKU Fill Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MOIC LightGBM** | **$5.66** | **$1.38** | $17.23 | **85.94%** | 91.00% |
| **MOIC CatBoost** | **$5.73** | **$1.38** | $17.46 | **85.64%** | 90.88% |
| **History-Tuned Baseline (HTB)** | $4.42 | $1.39 | $11.49 | 90.64% | 92.72% |
| **Fixed Standard Heuristic (FSH)** | $4.63 | $2.38 | $9.65 | 92.14% | 94.27% |
| **Hindsight Grid Oracle (ORACLE)** | $2.03 | $0.93 | $6.20 | 97.34% | 98.20% |

#### Analytical Insights: Clean Out-of-Sample Performance

1. **Median Cost Reduction vs. FSH:**  
   The Fixed Standard Heuristic ($R=7, L=3, \text{TSL}=0.95$) places recurring orders every 7 days regardless of demand velocity, accumulating excess holding buffers and ordering fees ($k=\$0.50$). The ML surrogate policies tailor review periods to each SKU's intermittency (ADI/CV²), beating FSH on **73.25% of all SKUs** and reducing median cost from **$2.02 to $0.55** (a ~73% reduction).
2. **Mean Cost Dynamics:**  
   On intermittent and highly volatile items, minor under-predictions of safety stock trigger lost-sales penalties ($b=\$0.25/\text{unit}$) during unpredicted demand surges. This inflates the right-tail cost distribution (standard deviation of $16.84 for LGBM vs $9.50 for FSH), resulting in a higher mean cost ($4.81–$4.91 vs $4.23).
3. **Surrogate vs. History-Tuned Baseline (HTB):**  
   The History-Tuned Baseline performs an empirical grid backtest strictly over the historical period of that exact series (days 1098–1550). HTB achieves the lowest causal mean cost ($3.96) and highest causal unit fill rate (89.52%), beating the ML surrogates on ~71% of SKUs. This demonstrates that while 8 cross-SKU aggregate features provide fast sub-millisecond inference for agent tool calling, per-SKU empirical history captures item-specific order-lead dynamics that summary features partially smooth out.

#### Feature Importance (LightGBM — AvgInventory model)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `Risk_Period` | Highest — validates the R+L replenishment exposure theory |
| 2 | `TSL` | Service-level directly scales safety stock |
| 3 | `ADI` | Demand frequency drives cycle stock |
| 4 | `Policy_Interaction` | R×L captures non-linear complexity |
| 5 | `CV2` | Demand variability second-order effect |
| 6 | `L` | Lead time independent effect |
| 7 | `R` | Review period independent effect |
| 8 | `Demand_Profile` | Regularity ratio — useful for lumpy SKUs |

---

### 5. Inference & Serving

**Model Loading (`inventory_tools.py`):**

```python
# Module-level cache — loads once at process startup, zero I/O per request
_MODELS_CACHE: Dict[str, Any] = load_moic_models()
```

Models are loaded lazily via `joblib.load()` with a `_ZeroPredictor` fallback so the app never crashes if files are missing.

**Prediction Path (per agent query):**

```python
# Inputs validated by Pydantic before inference
params = PolicySimulationInput(adi=adi, cv2=cv2, r=r, l=l, tsl=tsl)

# 3 engineered features computed inline (no external call)
risk_period        = params.r + params.l
policy_interaction = params.r * params.l
demand_profile     = params.adi / params.cv2 if params.cv2 != 0 else 0.0

# Single-row DataFrame -> model.predict() -> sub-millisecond latency
features = pd.DataFrame([[adi, cv2, r, l, tsl, risk_period, policy_interaction, demand_profile]],
                        columns=FEATURE_COLS)
pred = model.predict(features)[0]
```

**Latency:** ~0.5–2 ms per prediction (CPU inference, single row). Three predictions (one per target) complete well within the LLM's own network latency.

---

## AI Agent System Architecture

### 6. Agent Logic & Loop

The agent uses the **ReAct (Reasoning + Acting)** pattern, implemented with LangGraph's `StateGraph`.

**Decision Loop:**

```
User Query
    │
    ▼
[agent node] ──── LLM reasons, selects tool ────► [tools node]
    ▲                                                    │
    └──────────── tool result appended to state ────────┘
    │
    ▼ (no more tool calls)
Final natural-language response to user
```

**Implementation Details:**
- **LLM:** Configurable via `GROQ_MODEL` (defaults to `qwen/qwen3.8-27b` or `llama-3.3-70b-versatile` via Groq Cloud, temperature=0 for deterministic outputs)
- **Graph:** `StateGraph(MessagesState)` with two nodes: `agent` and `tools`
- **Routing:** `tools_condition` — if last message has `tool_calls`, route to tools; otherwise END
- **Recursion cap:** 24 steps (`GRAPH_CONFIG = {"recursion_limit": 24}`) allows rich multi-policy comparisons (up to ~12 tool round-trips)
- **Streaming:** `app.stream(mode="values")` enables real-time status updates in the Streamlit UI

---

### 7. Tooling & Context Strategy

**Three LangChain Tools — deterministic symbolic interface:**

| Order | Tool | Input Schema | What it does |
|-------|------|-------------|--------------|
| 1st | `get_sku_profile` | `sku_id: str` | Retrieves ADI and CV² for the requested SKU from the 12,201-key catalogue (9,147 state series + base items + archetypes) in $O(1)$ |
| 2nd | `simulate_policy_outcomes` | `adi, cv2, r, l, tsl, model_framework?` | Builds 8-feature vector, runs LightGBM, CatBoost, or Ensemble inference; returns AvgInventory, LostSales, NumOrders |
| 3rd | `calculate_financial_impact` | `avg_inv, lost_sales, num_orders, h, b, k` | Computes `Cost = h·I + b·LS + k·N` and returns decomposed monetary cost breakdown |

**No RAG / Vector Store:**
The system uses a **structured in-memory database** rather than RAG — demand features are deterministic and structured, making semantic vector retrieval unnecessary and slower.

**State Management:**
- Conversation history lives in `st.session_state.lc_messages` (LangChain `HumanMessage` / `AIMessage` objects)
- Full multi-turn memory: the complete message list is passed to `app.stream()` on every turn
- No external memory store required — Groq's context window handles the conversation

**SKU Catalogue Strategy:**
```python
# sku_store.py — loaded once at import time, O(1) lookups forever
_SKU_DB = {
    "FOODS_1_001_CA": {"adi": 1.22, "cv2": 0.68},  # 9,147 Level-11 state-specific entries
    "FOODS_1_001":    {"adi": 1.22, "cv2": 0.68},  # 3,049 base item fallbacks
    "SKU_FAST_MOVING": {"adi": 1.10, "cv2": 0.20},  # 5 mock archetypes always available
    # Total: 12,201 instant-lookup keys
}
```

---

### 8. Prompt Engineering & Safety

**System Prompt Design Principles:**

The system prompt in `agent_core.py` is engineered around three constraints:

**1. Tool-call sequencing (prevents hallucination):**
```
You are strictly forbidden from guessing, estimating, or calculating
any inventory metric yourself. Follow this exact workflow:
  1. Call get_sku_profile to retrieve the real ADI and CV2.
  2. Call simulate_policy_outcomes using those values (optionally pass model_framework).
  3. If costs are needed, call calculate_financial_impact.
```

**2. Multi-Policy Comparison Protocol:**
```
If the user asks to compare two policies (e.g. R=7 vs R=14) or compare two SKUs:
  - Execute profile, simulation, and cost steps for EACH scenario.
  - Synthesize a structured comparative table and clearly recommend the cost-minimizing policy.
  - STOP calling tools once all comparative scenarios are evaluated.
```

**3. Output format mandate:**
```
Present results in a structured, professional format with clear
sections for SKU profile, simulation outcomes, and cost breakdown.
```

**Safety Guardrails:**
- `recursion_limit: 24` in `GRAPH_CONFIG` — hard graph-level cap preventing infinite loops while accommodating comparisons
- Rate-limit errors caught and surfaced to the user with wait time parsed from the error message
- Pydantic V2 validation on all tool inputs — malformed agent outputs are rejected before hitting the model
- `_ZeroPredictor` fallback — agent operates safely even if model files are absent

---

## ML + AI Agent Synergy

### 9. Interactive Workflow

The ML models and AI Agent are coupled through a **deterministic tool interface**. The LLM never touches raw data or runs computations directly — it only orchestrates calls to validated, type-safe tool functions.

**Step-by-step data flow for a single query:**

```
User: "What is the total cost for HOBBIES_1_008_TX with R=7, L=3, TSL=0.95?"

Step 1 — Agent calls get_sku_profile("HOBBIES_1_008_TX")
         sku_store.py returns: {"adi": 3.41, "cv2": 1.12}

Step 2 — Agent calls simulate_policy_outcomes(adi=3.41, cv2=1.12, r=7, l=3, tsl=0.95)
         inventory_tools.py:
           - Pydantic validates inputs
           - Computes engineered features (Risk_Period=10, Policy_Interaction=21, Demand_Profile=3.04)
           - Calls lgbm_avg_inventory.predict([[3.41, 1.12, 7, 3, 0.95, 10, 21, 3.04]])
           - Returns: {avg_inventory: 28.4, lost_sales: 12.1, num_orders: 52}

Step 3 — Agent calls calculate_financial_impact(28.4, 12.1, 52, h=1.0, b=10.0, k=50.0)
         Returns: {total_cost: 2,749.4, holding: 28.4, shortage: 121.0, ordering: 2,600.0}

Step 4 — Agent synthesises a formatted Markdown response with all three sections.
```

**Why this architecture?**
- The LLM provides **flexible intent parsing** (handles varied phrasing, default parameters, SKU name variations)
- The ML model provides **deterministic, sub-millisecond predictions** (no hallucination possible)
- Pydantic acts as the **contract layer** between LLM outputs and ML inputs

---

### 10. End-to-End Mermaid Diagram

```mermaid
flowchart TD
    U([👤 Supply Chain Manager]) -->|Natural language query| ST

    subgraph UI["Streamlit Dashboard (app.py)"]
        ST[Chat Input] --> RS[Real-time Status Box]
        RS --> CO[Final Markdown Response]
    end

    ST -->|LangChain HumanMessage| AG

    subgraph AGENT["LangGraph ReAct Agent (agent_core.py)"]
        AG[Llama 3.3-70b · Groq · temp=0] -->|tool_calls in response| TC{Tool Router}
        TC -->|tools_condition| TN[ToolNode]
        TN -->|tool results appended| AG
        AG -->|no tool_calls| FIN[Final AIMessage]
    end

    TN --> T1
    TN --> T2
    TN --> T3

    subgraph TOOLS["LangChain Tools (inventory_tools.py + sku_store.py)"]
        T1["🔍 get_sku_profile(sku_id)"]
        T2["⚙️ simulate_policy_outcomes(adi,cv2,r,l,tsl)"]
        T3["💰 calculate_financial_impact(inv,ls,orders,h,b,k)"]
    end

    T1 -->|sku_id lookup O(1)| DB[(M5 Demand Features CSV\n9,147 SKUs in-memory)]
    DB -->|adi, cv2| T1

    T2 -->|8-feature vector| ML

    subgraph ML["ML Surrogate Models (models/)"]
        FE["Feature Engineering\nRisk_Period = R+L\nPolicy_Interaction = R×L\nDemand_Profile = ADI/CV2"]
        LGB["lgbm_avg_inventory.joblib\n→ AvgInventory prediction"]
        LGB2["lgbm_lost_sales.joblib\n→ LostSales prediction"]
        LGB3["lgbm_num_orders.joblib\n→ NumOrders prediction"]
        FE --> LGB
        FE --> LGB2
        FE --> LGB3
    end

    T2 --> FE
    LGB --> T2
    LGB2 --> T2
    LGB3 --> T2

    T3 -->|"C = h·I + b·LS + k·N"| COST["Cost Breakdown\nHolding · Shortage · Ordering"]
    COST --> T3

    FIN -->|streamed chunks| UI
    U -.->|"Groq Cloud API\n(external)"| AG
```

---

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| LLM | Llama 3.3-70b (Groq Cloud) | groq 0.37.1 |
| Agent Orchestration | LangGraph StateGraph + LangChain | langgraph 1.2.11 / langchain 1.3.15 |
| ML Surrogates | LightGBM + CatBoost | 4.7.0 / 1.2.10 |
| ML Validation | Scikit-learn | 1.9.0 |
| Data | pandas, NumPy, SciPy | 3.0.5 / 2.5.2 / 1.18.0 |
| Input Validation | Pydantic V2 | 2.13.4 |
| Model Serialisation | joblib | 1.5.3 |
| UI | Streamlit | 1.61.1 |
| Config | python-dotenv | 1.2.2 |
| Runtime | Python | 3.12+ |

---

## Prerequisites

- Python 3.12+
- A Groq API key — free tier at https://console.groq.com
- Pre-trained `.joblib` model files (see Model Setup below)

---

## Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/moic-supply-chain-agent.git
cd moic-supply-chain-agent

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Open .env and paste your Groq API key
```

### Model Setup

Pre-trained surrogate models are stored in `models/` (LightGBM and CatBoost):

```text
models/
├── lgbm_avg_inventory.joblib
├── lgbm_lost_sales.joblib
├── lgbm_num_orders.joblib
├── catb_AvgInventory.joblib
├── catb_LostSales.joblib
└── catb_NumOrders.joblib
```

### Reproducibility & Pipeline Execution

The experiment design enforces strict origin-based causality (`history_end_day` inclusive, zero future demand leakage into features or labels) configured in [`src/config.py`](src/config.py) using `RANDOM_SEED = 42`.

#### 1. Dataset Setup
To run benchmarks with the committed 100-SKU sample dataset:
```bash
python scripts/download_data.py
```
To run on the full Walmart M5 competition dataset, download `sales_train_validation.csv` into `Dataset/` or `data/raw/` via Kaggle CLI:
```bash
kaggle competitions download -c m5-forecasting-accuracy
```

#### 2. End-to-End Pipeline Execution
Execute the numbered scripts in sequence:
```bash
# Step 1: Aggregate to Level 11 and compute causal demand features (as of day 1550)
python scripts/01_prepare_data.py

# Step 2: Generate 2.05M causal simulation labels across 3 non-overlapping origins
python scripts/02_generate_labels.py

# Step 3: Train surrogate models via chronological validation and grouped-SKU holdout
python scripts/03_train_surrogates.py

# Step 4: Select cost-minimizing policies on aligned 28-day horizon as of day 1550
python scripts/04_select_policies.py

# Step 5: Run clean out-of-sample evaluation across 5 policies with paired bootstrap CIs
python scripts/05_evaluate.py
```

#### 3. Gitignored Files & Regeneration Guide
The following generated artifacts are excluded from Git to maintain a clean repository:
- `data/processed/*.csv`: Processed sales (`sales_train_level_11.csv`), demand features (`m5_demand_features.csv`), 2.05M simulation training records (`ml_training_data_v2_corrected.csv`), and per-SKU policy CSVs (`optimal_policies_v2_*.csv`, `evaluation_results_v2.csv`). Regenerated via `scripts/01_prepare_data.py`, `scripts/02_generate_labels.py`, `scripts/04_select_policies.py`, and `scripts/05_evaluate.py`.
- `models/*.joblib`, `models/v2/*.joblib`: Trained LightGBM and CatBoost surrogate model binaries. Regenerated via `scripts/03_train_surrogates.py`.
- `catboost_info/`: Ephemeral CatBoost training telemetry.
- `data/raw/`: Raw competition files from Kaggle.

Tracked evaluation artifacts and metadata:
- `data/sample_sales_level_11.csv`: 100-SKU sample sales dataset for testing without full M5 data.
- `models/v2/split_definition.json`: Complete chronological and grouped-SKU split definition.
- `models/v2/v2_validation_metrics.json`: Surrogate model chronological and holdout validation scores.
- `data/processed/evaluation_summary_v2.json`: Clean out-of-sample evaluation metrics, win rates, and bootstrap 95% CIs.

#### 4. Out-of-Sample Empirical Note
Out-of-sample evaluation confirms that while ML surrogate policies reduce median cost by 72.6% over fixed heuristics ($0.55 vs $2.02) and win on 73.25% of SKUs, they do **not** beat the causal History-Tuned Baseline (HTB) on mean cost ($4.81/$4.91 vs $3.96) due to right-tail lost-sales penalties on highly volatile, intermittent series.

## Running the App

Launch either the Streamlit web dashboard or the interactive terminal agent:

```bash
# Option A: Unified Launcher (Streamlit UI by default)
python main.py

# Option B: Streamlit Web UI directly
streamlit run app.py

# Option C: Interactive CLI ReAct Agent
python main.py --cli
```

The Streamlit dashboard opens at **http://localhost:8501**.

---

## Environment Variables

| Variable | Description | Default | Where to get it |
|----------|-------------|---------|-----------------|
| `GROQ_API_KEY` | API key for Groq LLM inference | *(required)* | https://console.groq.com |
| `GROQ_MODEL` | Groq chat model identifier | `qwen/qwen3.8-27b` | Groq Console (e.g. `qwen/qwen3.8-27b` or `llama-3.3-70b-versatile`) |

---

## Usage Examples

```text
Analyse SKU_FAST_MOVING with R=7, L=3, TSL=0.95. Provide the full cost breakdown.
What are the expected lost sales for SKU_LUMPY with R=14, L=5, TSL=0.90?
Compare R=7 vs R=14 for FOODS_1_001_CA with L=3 and TSL=0.95. Which minimises cost?
Run a full MOIC simulation for HOBBIES_1_008_TX with R=10, L=4, TSL=0.98.
Evaluate inventory risk for SKU_SEASONAL with R=10, L=4, TSL=0.98 using CatBoost.
```

---

## Project Structure

```text
moic-supply-chain-agent/
├── main.py                     # Unified entrypoint (Streamlit UI or CLI agent)
├── app.py                      # Streamlit enterprise dashboard & execution trace
├── agent_core.py               # LangGraph ReAct agent + multi-scenario graph logic
├── inventory_tools.py          # LangChain tools: LightGBM, CatBoost & Ensemble inference + financial breakdown
├── sku_store.py                # SKU metadata store (12,201 keys: 9,147 M5 series + fallbacks + archetypes)
├── notebooks/                  # Offline research & training
│   └── MAIN_NOTEBOOK.ipynb     # Full end-to-end offline pipeline (Data prep, simulation, GPU training)
├── models/                     # Trained .joblib files (LightGBM & CatBoost surrogates)
│   ├── lgbm_avg_inventory.joblib
│   ├── lgbm_lost_sales.joblib
│   ├── lgbm_num_orders.joblib
│   ├── catb_AvgInventory.joblib
│   ├── catb_LostSales.joblib
│   └── catb_NumOrders.joblib
├── Dataset/                    # M5 demand feature store
│   ├── m5_demand_features.csv  <- Minimum required for runtime SKU lookup
│   └── ...
├── docs/                       # Project documentation & guides (local/private)
├── requirements.txt            # Pinned dependencies
├── pyproject.toml              # Build & dependency metadata
├── .env.example                # Safe secrets template
├── .gitignore                  # Git exclusions (including private notes & models)
└── README.md
```

---

## Contributing

1. Fork the repo and create a feature branch: `git checkout -b feat/your-feature`
2. Make your changes and confirm tests pass.
3. Ensure `.env` and `.joblib` files are NOT staged (`git check-ignore -v models/*.joblib`).
4. Open a Pull Request with a clear description.

---

## License

MIT License — see LICENSE for details.

---

## References

- Greasley, A. (2023). *MOIC: Model-Independent Optimal Inventory Control*.
- Makridakis, S. et al. (2022). *M5 Accuracy Competition: Results, Findings and Conclusions*. International Journal of Forecasting.
- LangGraph ReAct Agent Pattern — LangChain Documentation.
- LightGBM: A Highly Efficient Gradient Boosting Decision Tree — NIPS 2017.
