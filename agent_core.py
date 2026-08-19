"""
agent_core.py
-------------
Neuro-Symbolic Supply Chain Decision Agent
Built with LangGraph + LangChain + Groq (Llama 3.3-70b)

Tool execution order enforced by the system prompt:
  1. get_sku_profile        → fetch real ADI & CV2
  2. simulate_policy_outcomes → run MOIC surrogate
  3. calculate_financial_impact → compute total cost (if costs requested)
"""

import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()  # reads GROQ_API_KEY (and any other vars) from .env

# ---------------------------------------------------------------------------
# LangGraph imports
# ---------------------------------------------------------------------------
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
from langchain_groq import ChatGroq

# ---------------------------------------------------------------------------
# Custom tools
# ---------------------------------------------------------------------------
from sku_store import get_sku_profile
from inventory_tools import simulate_policy_outcomes, calculate_financial_impact

# ---------------------------------------------------------------------------
# 1.  Tool registry
# ---------------------------------------------------------------------------
TOOLS = [get_sku_profile, simulate_policy_outcomes, calculate_financial_impact]

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS, tool_choice="auto")

# ---------------------------------------------------------------------------
# 2.  System prompt — strictly enforces tool-call sequence
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert AI Supply Chain Agent backed by the MOIC (Model-Independent Optimal Inventory Control) decision engine.
You are strictly forbidden from guessing, estimating, or calculating any inventory metric yourself.

Follow this exact workflow for every user query:
1. Call `get_sku_profile` to retrieve the real ADI and CV2 for the requested SKU.
2. Call `simulate_policy_outcomes` using the ADI and CV2 from step 1, plus the user's R, L, and TSL values. Default to R=7, L=3, TSL=0.95 if not specified.
3. If the user asks about costs or financial impact, call `calculate_financial_impact` using the simulation results from step 2.

CRITICAL INSTRUCTION: Once you have received the result from `calculate_financial_impact`, you have everything you need. You MUST immediately synthesize the final answer for the user and STOP. Do NOT call any tools again for the same query.

Present results in a structured, professional format with clear sections for SKU profile, simulation outcomes, and cost breakdown."""

# ---------------------------------------------------------------------------
# 3.  Graph nodes
# ---------------------------------------------------------------------------

def call_model(state: MessagesState) -> MessagesState:
    """Invoke the LLM with the full conversation history."""
    from langchain_core.messages import SystemMessage

    messages = state["messages"]

    # Prepend system prompt if this is the very first agent call
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(TOOLS)

# ---------------------------------------------------------------------------
# 4.  Build the StateGraph
# ---------------------------------------------------------------------------
builder = StateGraph(MessagesState)

builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")

# Route: if the last message has tool_calls → tools, otherwise → END
builder.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "tools", END: END},
)

# After tool execution, always return to the agent for next reasoning step
builder.add_edge("tools", "agent")

app = builder.compile()

# Hard cap on graph recursion — prevents infinite tool-calling loops.
# Each full round-trip (agent → tools → agent) costs 2 steps, so 12 allows
# up to 6 tool calls before the graph forcibly terminates.
GRAPH_CONFIG = {"recursion_limit": 12}

# ---------------------------------------------------------------------------
# 5.  Interactive CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    print("=" * 60)
    print("🤖  Supply Chain Decision Agent  (Llama 3.3 · Groq)")
    print("=" * 60)
    print("Example queries:")
    print("  • What is the optimal policy for SKU_FAST_MOVING with R=7, L=3, TSL=0.95?")
    print("  • Analyse SKU_LUMPY with a 14-day review period and 95% service level.")
    print("  • Compare SKU_SLOW vs SKU_TRENDING for R=10, L=5, TSL=0.90.")
    print("  Type 'quit' or 'exit' to stop.\n")

    conversation_messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        conversation_messages.append(HumanMessage(content=user_input))

        print("\nAgent: ", end="", flush=True)

        final_messages = None
        for chunk in app.stream(
            {"messages": conversation_messages},
            stream_mode="values",
        ):
            final_messages = chunk["messages"]

        if final_messages:
            last_msg = final_messages[-1]
            content = getattr(last_msg, "content", "")
            if content:
                print(content)
            # Add assistant reply to conversation history for multi-turn support
            conversation_messages = final_messages

        print()  # blank line between turns
