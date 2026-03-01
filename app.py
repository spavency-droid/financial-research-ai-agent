import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph

st.set_page_config(page_title="AI Financial Research Agent", layout="wide")
st.title("🧠 AI Financial Research Agent")

# -----------------------------
# LLM Setup
# -----------------------------
llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
    openai_api_key=st.secrets["OPENAI_API_KEY"]
)

# -----------------------------
# Agent State
# -----------------------------
class AgentState(dict):
    pass

# -----------------------------
# Router Node
# -----------------------------
def router(state):
    query = state["query"]

    prompt = f"""
    Classify the user request into one of these categories:
    - technical
    - risk

    Only return one word.
    Query: {query}
    """

    response = llm([HumanMessage(content=prompt)])
    decision = response.content.strip().lower()

    state["route"] = decision
    return state

# -----------------------------
# Fetch Data Tool
# -----------------------------
def fetch_data(state):
    symbol = state["symbol"]
    stock = yf.Ticker(symbol)
    data = stock.history(period="3mo")

    if data.empty:
        state["error"] = "Invalid symbol."
        return state

    state["data"] = data
    return state

# -----------------------------
# Technical Analysis Tool
# -----------------------------
def technical_tool(state):
    data = state["data"]

    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()

    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    data["RSI"] = 100 - (100 / (1 + rs))

    state["data"] = data
    state["analysis_type"] = "Technical Analysis"
    return state

# -----------------------------
# Risk Tool
# -----------------------------
def risk_tool(state):
    data = state["data"]

    volatility = data["Close"].pct_change().std() * (252 ** 0.5)
    state["risk_score"] = round(volatility, 4)
    state["analysis_type"] = "Risk Analysis"
    return state

# -----------------------------
# Insight Generator
# -----------------------------
def generate_insight(state):

    if state["analysis_type"] == "Technical Analysis":
        latest = state["data"].iloc[-1]
        context = f"""
        Close: {latest['Close']}
        RSI: {latest['RSI']}
        MA20: {latest['MA20']}
        MA50: {latest['MA50']}
        """

    else:
        context = f"Volatility (annualized): {state['risk_score']}"

    prompt = f"""
    You are a financial analyst.

    Provide structured insight for:
    {state['analysis_type']}

    Data:
    {context}

    Include:
    - Interpretation
    - Risk level
    - Actionable suggestion
    """

    response = llm([HumanMessage(content=prompt)])
    state["final_report"] = response.content
    return state

# -----------------------------
# Conditional Routing Logic
# -----------------------------
def route_decision(state):
    return state["route"]

# -----------------------------
# Build LangGraph
# -----------------------------
workflow = StateGraph(AgentState)

workflow.add_node("router", router)
workflow.add_node("fetch", fetch_data)
workflow.add_node("technical", technical_tool)
workflow.add_node("risk", risk_tool)
workflow.add_node("insight", generate_insight)

workflow.set_entry_point("router")

workflow.add_edge("router", "fetch")

workflow.add_conditional_edges(
    "fetch",
    route_decision,
    {
        "technical": "technical",
        "risk": "risk"
    }
)

workflow.add_edge("technical", "insight")
workflow.add_edge("risk", "insight")

agent = workflow.compile()

# -----------------------------
# Streamlit UI
# -----------------------------
query = st.text_input("Ask something (Example: Analyze RELIANCE.NS technically)")
symbol = st.text_input("Enter Stock Symbol (Example: RELIANCE.NS)")

if query and symbol:
    with st.spinner("Agent thinking..."):
        result = agent.invoke({
            "query": query,
            "symbol": symbol
        })

    if "error" in result:
        st.error(result["error"])
    else:
        st.subheader("📊 AI Report")
        st.write(result["final_report"])

        if result["analysis_type"] == "Technical Analysis":
            st.subheader("📈 Chart")
            data = result["data"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close"))
            fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA20"))
            fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="MA50"))
            st.plotly_chart(fig, use_container_width=True)
