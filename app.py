"""
app.py — MOIC Supply Chain Intelligence Platform
Streamlit enterprise dashboard for the Neuro-Symbolic Supply Chain Decision Agent.
Run: streamlit run app.py
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# ── Page config (must be the first Streamlit call) ──────────────────────────
st.set_page_config(
    page_title="MOIC Supply Chain Agent",
    page_icon="assets/favicon.ico" if False else "📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS for enterprise SaaS look ──────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] {
    background: #0d1117;
    color: #e6edf3;
}
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* ── Top header bar ── */
.moic-header {
    background: linear-gradient(135deg, #1a2332 0%, #0d1117 100%);
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 20px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.moic-header h1 {
    margin: 0;
    font-size: 1.55rem;
    font-weight: 700;
    color: #f0f6fc;
    letter-spacing: -0.3px;
}
.moic-header p {
    margin: 4px 0 0 0;
    font-size: 0.82rem;
    color: #8b949e;
}
.badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.4px;
    margin-right: 5px;
}
.badge-green  { background: #1a3b2b; color: #3fb950; border: 1px solid #238636; }
.badge-blue   { background: #0d2644; color: #58a6ff; border: 1px solid #1f6feb; }
.badge-purple { background: #26183e; color: #a371f7; border: 1px solid #6e40c9; }

/* ── Sidebar labels ── */
.sidebar-section {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #484f58 !important;
    margin: 18px 0 6px 0;
}
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.dot-green  { background: #3fb950; box-shadow: 0 0 6px #3fb950; }
.dot-yellow { background: #d29922; }
.dot-red    { background: #f85149; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    margin-bottom: 10px;
    padding: 4px 8px;
}
/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.1) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #161b22;
    border: 1px solid #30363d !important;
    border-radius: 6px;
}
[data-testid="stExpander"] summary {
    font-size: 0.82rem;
    color: #8b949e;
}

/* ── Divider ── */
hr { border-color: #30363d !important; }

/* ── Buttons ── */
.stButton > button {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 6px !important;
    font-size: 0.78rem !important;
    text-align: left !important;
    width: 100% !important;
}
.stButton > button:hover {
    border-color: #58a6ff !important;
    color: #58a6ff !important;
    background: #0d2644 !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #f0f6fc !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached agent loader ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising MOIC engine…")
def load_agent():
    from agent_core import app, GRAPH_CONFIG
    return app, GRAPH_CONFIG


agent_app, GRAPH_CONFIG = load_agent()


# ── Session state initialisation ────────────────────────────────────────────
if "lc_messages" not in st.session_state:
    st.session_state.lc_messages = []       # LangChain message objects
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []      # {"role", "content"} for display
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 4px 0 16px 0;">
        <div style="font-size:1.1rem; font-weight:700; color:#f0f6fc; letter-spacing:-0.2px;">
            MOIC Control Panel
        </div>
        <div style="font-size:0.75rem; color:#484f58; margin-top:2px;">
            Supply Chain Decision Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">System Status</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem; line-height:2;">
        <span class="status-dot dot-green"></span><b>Groq API</b> — Connected<br>
        <span class="status-dot dot-green"></span><b>LightGBM Surrogates</b> — Loaded (3 models)<br>
        <span class="status-dot dot-green"></span><b>SKU Store</b> — 3,054 M5 series<br>
        <span class="status-dot dot-green"></span><b>LangGraph</b> — ReAct agent ready
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="sidebar-section">Model Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.8rem; color:#8b949e; line-height:2.2;">
        <span class="badge badge-blue">LLM</span> Llama 3.3-70b · Groq<br>
        <span class="badge badge-purple">ML</span> LightGBM MOIC surrogates<br>
        <span class="badge badge-green">Graph</span> LangGraph ReAct · 12-step cap<br>
        <span class="badge badge-blue">Data</span> M5 Walmart Forecasting
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="sidebar-section">Example Queries</div>', unsafe_allow_html=True)

    example_queries = [
        ("SKU Analysis", "Analyse SKU_FAST_MOVING with R=7, L=3, TSL=0.95. Provide the full cost breakdown."),
        ("Lost Sales", "What are the expected lost sales for SKU_LUMPY with R=14, L=5, TSL=0.90?"),
        ("Policy Comparison", "Compare R=7 vs R=14 for FOODS_1_001 with L=3 and TSL=0.95. Which minimises cost?"),
        ("Slow Mover", "Run a full MOIC simulation for SKU_SLOW with R=30, L=7, TSL=0.99."),
        ("Seasonal Risk", "Evaluate inventory risk for SKU_SEASONAL with R=10, L=4, TSL=0.98."),
    ]

    for label, query in example_queries:
        if st.button(f"{label}", key=query, use_container_width=True):
            st.session_state.pending_prompt = query

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Session Turns", st.session_state.turn_count)
    with col_b:
        st.metric("SKU Catalogue", "3,054")

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.lc_messages = []
        st.session_state.chat_history = []
        st.session_state.turn_count = 0
        st.rerun()


# ── Main area header ─────────────────────────────────────────────────────────
st.markdown("""
<div class="moic-header">
    <div>
        <h1>MOIC Supply Chain Intelligence</h1>
        <p>
            Neuro-symbolic decision engine powered by LightGBM surrogate models and LangGraph orchestration.
            Queries are resolved via tool-enforced simulation — no hallucination, no guessing.
        </p>
        <div style="margin-top:10px;">
            <span class="badge badge-green">LIVE</span>
            <span class="badge badge-blue">Llama 3.3-70b</span>
            <span class="badge badge-purple">MOIC Framework</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Render existing chat history ─────────────────────────────────────────────
for msg in st.session_state.chat_history:
    role = msg["role"]
    avatar = "👤" if role == "user" else "⚙"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])


# ── Agent invocation ─────────────────────────────────────────────────────────
TOOL_DISPLAY_NAMES = {
    "get_sku_profile":           "Fetching SKU demand profile (ADI, CV²)",
    "simulate_policy_outcomes":  "Running MOIC surrogate simulation",
    "calculate_financial_impact":"Computing inventory cost breakdown",
}

def run_agent(user_input: str):
    # Append to LangChain history and display
    st.session_state.lc_messages.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="⚙"):
        tool_steps: list[str] = []
        final_content = ""

        # Status container for live tool-call updates
        status_box = st.status("Running MOIC simulation pipeline…", expanded=True)

        try:
            for chunk in agent_app.stream(
                {"messages": st.session_state.lc_messages},
                config=GRAPH_CONFIG,
                stream_mode="values",
            ):
                chunk_messages = chunk.get("messages", [])
                if not chunk_messages:
                    continue

                last = chunk_messages[-1]

                # Detect tool invocations and update live status
                if hasattr(last, "tool_calls") and last.tool_calls:
                    for tc in last.tool_calls:
                        name = tc.get("name", "")
                        label = TOOL_DISPLAY_NAMES.get(name, name)
                        tool_steps.append(name)
                        status_box.write(f"**Step {len(tool_steps)}:** {label}")

                # Capture final AI text (message with content but no pending tool calls)
                if (
                    isinstance(last, AIMessage)
                    and last.content
                    and not getattr(last, "tool_calls", [])
                ):
                    final_content = last.content
                    st.session_state.lc_messages = chunk_messages

        except Exception as e:
            err_msg = str(e)
            if "recursion" in err_msg.lower():
                final_content = (
                    "> **Reasoning depth limit reached.**  \n"
                    "The agent exhausted its 12-step cap on this query. "
                    "Try rephrasing or breaking the request into smaller steps."
                )
            elif "rate_limit" in err_msg.lower() or "429" in err_msg:
                import re
                wait = re.search(r"try again in ([\w\.]+)", err_msg, re.IGNORECASE)
                wait_str = f" Please wait **{wait.group(1)}** before retrying." if wait else ""
                final_content = (
                    "> **Groq API rate limit reached.**  \n"
                    f"The daily token quota for this API key is exhausted.{wait_str}"
                )
            else:
                final_content = f"> **Error during simulation:**  \n`{err_msg}`"

        # Close status box with summary
        steps_done = len(tool_steps)
        if steps_done > 0:
            status_box.update(
                label=f"Simulation complete — {steps_done} tool{'s' if steps_done > 1 else ''} executed",
                state="complete",
                expanded=False,
            )
        else:
            status_box.update(label="Response ready", state="complete", expanded=False)

        # Render the final answer
        if final_content:
            st.markdown(final_content)
        else:
            st.warning("The agent did not produce a text response for this query.")
            final_content = "_No response generated._"

        # Optional tool trace expander (collapsed by default)
        if tool_steps:
            with st.expander("Execution trace", expanded=False):
                for i, name in enumerate(tool_steps, 1):
                    label = TOOL_DISPLAY_NAMES.get(name, name)
                    st.markdown(
                        f"`{i}.` **{label}**  "
                        f"<span style='color:#484f58;font-size:0.78rem;'>`{name}`</span>",
                        unsafe_allow_html=True,
                    )

    st.session_state.chat_history.append({"role": "assistant", "content": final_content})
    st.session_state.turn_count += 1


# ── Handle sidebar button → pending prompt ───────────────────────────────────
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    run_agent(prompt)

# ── Chat input ───────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "Enter a SKU query, policy parameters, or cost analysis request…"
)
if user_input:
    run_agent(user_input)
