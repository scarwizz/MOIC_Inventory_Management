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

1. **ML Pipeline** — Runs (R, s, S) inventory policy simulations over 686,025 SKU-policy observations from the M5 Walmart dataset, trains LightGBM and CatBoost surrogate models on 8 engineered features, and exports them as serialised predictors.
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
| History length | 1,913 days (~5.25 years) |
| Simulation window | Last 365 days (days 1549–1913) |
| Policy grid | R ∈ [1,30], L ∈ [1,30], TSL ∈ {0.90, 0.95, 0.99} → **75 policies** |
| Training observations | 9,147 × 75 = **686,025 rows** |

**Preprocessing Steps:**
1. **Aggregation** — Item-level store sales grouped by `(item_id, state_id)` to eliminate store noise.
2. **Downcast** — Integer and float columns reduced to the smallest safe dtype (int8/float32) to cut memory by ~60%.
3. **Demand statistics** — Per-series Average Inter-Demand Interval (ADI) and Squared Coefficient of Variation (CV²) computed over all 1,913 days.
4. **Simulation labels** — For each of the 686,025 (series, policy) pairs, a discrete-event (R, s, S) lost-sales simulation produces the three regression targets.
5. **No data leakage** — Simulation runs exclusively on days 1549–1913; ADI/CV² are computed from the full history before that window.

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
| **CatBoost** (selected ensemble) | `MAE` (CPU mode) | Superior total-cost optimisation despite higher RMSE — the "Predictor-Optimizer Paradox" |
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

**Training Infrastructure:**
- Kaggle Notebook with **P100 GPU** (LightGBM GPU mode via OpenCL)
- CatBoost trained on CPU (GPU MAE is not implemented for the metric logging path)
- 80/20 train/validation split (`random_state=42`) — 548,820 training rows, 137,205 validation rows
- Three separate models per framework, one per target variable

---

### 4. Evaluation Metrics & Results

All metrics computed on the **20% hold-out validation set (137,205 rows)**.

#### Quantitative Results

| Model | Target | MAE | RMSE | Relative MAE |
|-------|--------|-----|------|-------------|
| **LightGBM** | AvgInventory | **7.67** | **29.19** | ~0.8% of typical inventory |
| **LightGBM** | LostSales | 659.7 | 2,037.3 | ~12% of typical lost-sales |
| **LightGBM** | NumOrders | **0.0007** | **0.0065** | <0.01% — near-perfect |
| CatBoost | AvgInventory | 9.52 | 33.99 | ~1.0% of typical inventory |
| CatBoost | LostSales | 1,091.7 | 2,844.6 | ~20% of typical lost-sales |
| CatBoost | NumOrders | 66.9* | 146.2* | *Pre-fix (GPU bug) |

> **The Predictor-Optimizer Paradox (Key Finding):**
> LightGBM achieves lower RMSE/MAE on all targets, yet CatBoost generates inventory policies with a **43% lower total cost** during policy optimisation. RMSE alone is an insufficient proxy for policy quality — a core finding of the MOIC paper.

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
- **LLM:** `llama-3.3-70b-versatile` via Groq Cloud (temperature=0 for deterministic outputs)
- **Graph:** `StateGraph(MessagesState)` with two nodes: `agent` and `tools`
- **Routing:** `tools_condition` — if last message has `tool_calls`, route to tools; otherwise END
- **Recursion cap:** Hard limit of 12 steps prevents runaway tool-calling loops (~6 full tool round-trips maximum)
- **Streaming:** `app.stream(mode="values")` enables real-time status updates in the Streamlit UI

---

### 7. Tooling & Context Strategy

**Three LangChain Tools — enforced call sequence:**

| Order | Tool | Input Schema | What it does |
|-------|------|-------------|--------------|
| 1st | `get_sku_profile` | `sku_id: str` | Looks up ADI and CV² for the requested SKU from the 9,147-series M5 catalogue (O(1) in-memory dict) |
| 2nd | `simulate_policy_outcomes` | `adi, cv2, r, l, tsl` | Builds 8-feature vector, runs LightGBM inference, returns AvgInventory, LostSales, NumOrders |
| 3rd | `calculate_financial_impact` | `avg_inv, lost_sales, num_orders, h, b, k` | Computes `Cost = h·I + b·LS + k·N` and returns cost breakdown |

**No RAG / Vector Store:**
The system uses a **structured in-memory database** rather than RAG — demand features are deterministic and structured, making semantic retrieval unnecessary and slower.

**State Management:**
- Conversation history lives in `st.session_state.lc_messages` (LangChain `HumanMessage` / `AIMessage` objects)
- Full multi-turn memory: the complete message list is passed to `app.stream()` on every turn
- No external memory store required — Groq's context window handles the conversation

**SKU Catalogue Strategy:**
```python
# sku_store.py — loaded once at import time, O(1) lookups forever
_SKU_DB = {
    "FOODS_1_001_CA": {"adi": 1.23, "cv2": 0.45},
    ...  # 9,147 real M5 entries
    "SKU_FAST_MOVING": {"adi": 1.1, "cv2": 0.2},  # 5 mock archetypes always available
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
  2. Call simulate_policy_outcomes using those values.
  3. If costs are needed, call calculate_financial_impact.
```

**2. Hard termination instruction (prevents recursion):**
```
CRITICAL INSTRUCTION: Once you have received the result from
calculate_financial_impact, you MUST immediately synthesize the
final answer and STOP. Do NOT call any tools again.
```

**3. Output format mandate:**
```
Present results in a structured, professional format with clear
sections for SKU profile, simulation outcomes, and cost breakdown.
```

**Safety Guardrails:**
- `recursion_limit: 12` in `GRAPH_CONFIG` — hard graph-level cap
- Rate-limit errors caught and surfaced to the user with wait time parsed from the error message
- Pydantic V2 validation on all tool inputs — malformed agent outputs are rejected before hitting the model
- `_ZeroPredictor` fallback — agent operates safely even without model files

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

Download `trained_models.zip` from the Releases page and extract the three files into `models/`:

```
models/
+-- lgbm_avg_inventory.joblib
+-- lgbm_lost_sales.joblib
+-- lgbm_num_orders.joblib
```

To retrain from scratch on Kaggle, run `kaggle_m5_final_hybrid_pipeline.py` with the M5 dataset attached.

### Dataset Setup

Only `m5_demand_features.csv` is required to run the agent:

```
Dataset/
+-- m5_demand_features.csv       <- minimum required for SKU lookup
+-- sales_train_validation.csv   <- only needed to retrain
+-- calendar.csv                 <- only needed to retrain
+-- sell_prices.csv              <- only needed to retrain
```

---

## Running the App

```bash
streamlit run app.py
```

Opens at http://localhost:8501

---

## Environment Variables

| Variable | Description | Where to get it |
|----------|-------------|-----------------|
| `GROQ_API_KEY` | API key for Groq LLM inference | https://console.groq.com |

---

## Usage Examples

```
Analyse SKU_FAST_MOVING with R=7, L=3, TSL=0.95. Provide the full cost breakdown.
What are the expected lost sales for SKU_LUMPY with R=14, L=5, TSL=0.90?
Compare R=7 vs R=14 for FOODS_1_001_CA with L=3 and TSL=0.95. Which minimises cost?
Run a full MOIC simulation for HOBBIES_1_008_TX with R=10, L=4, TSL=0.98.
Evaluate inventory risk for SKU_SEASONAL with R=10, L=4, TSL=0.98.
```

---

## Project Structure

```
moic-supply-chain-agent/
+-- app.py                     # Streamlit enterprise dashboard
+-- agent_core.py              # LangGraph ReAct agent + graph definition
+-- inventory_tools.py         # LangChain tools: simulate + financial impact
+-- sku_store.py               # SKU metadata store (M5 catalogue + mock archetypes)
+-- simulation.py              # (R,s,S) offline simulation engine
+-- train_surrogates.py        # Local surrogate model training script
+-- MAIN_NOTEBOOK.ipynb        # Research notebook: full offline pipeline
+-- analysis.ipynb             # Exploratory data analysis
+-- models/                    # Trained .joblib files (NOT tracked by git)
|   +-- lgbm_avg_inventory.joblib
|   +-- lgbm_lost_sales.joblib
|   +-- lgbm_num_orders.joblib
+-- Dataset/                   # M5 data files (NOT tracked by git)
|   +-- m5_demand_features.csv
+-- requirements.txt
+-- pyproject.toml
+-- .env.example               # Safe secrets template -- commit this
+-- .env                       # Real secrets -- NEVER commit
+-- .gitignore
+-- README.md
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
