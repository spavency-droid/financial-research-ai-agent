import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
import os

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END


# -----------------------------
# 🔐 LLM SETUP
# -----------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


# -----------------------------
# 📊 STOCK ANALYSIS FUNCTION
# -----------------------------
def analyze_stock(symbol: str):

    symbol = symbol.upper()

    if not symbol.endswith(".NS"):
        symbol += ".NS"

    data = yf.Ticker(symbol).history(period="6mo")

    if data.empty:
        return {"error": "No data found"}

    data["RSI"] = ta.momentum.RSIIndicator(data["Close"]).rsi()
    latest = data.iloc[-1]

    return {
        "symbol": symbol,
        "price": float(latest["Close"]),
        "rsi": float(latest["RSI"]),
        "data": data.tail(50)
    }


# -----------------------------
# 🧠 LANGGRAPH STATE
# -----------------------------
class State(dict):
    pass


# -----------------------------
# ROUTER NODE
# -----------------------------
def router(state: State):

    query = state["query"]

    # Simple symbol detection
    words = query.upper().split()

    symbol = None
    for w in words:
        if w.isalpha() and len(w) > 2:
            symbol = w
            break

    state["symbol"] = symbol
    return state


# -----------------------------
# ANALYSIS NODE
# -----------------------------
def analysis_node(state: State):

    symbol = state["symbol"]

    if not symbol:
        state["result"] = "Could not detect stock symbol."
        return state

    result = analyze_stock(symbol)

    if "error" in result:
        state["result"] = result["error"]
        return state

    data = result["data"]

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close"))
    st.plotly_chart(fig, use_container_width=True)

    # LLM explanation
    prompt = f"""
    Stock: {result['symbol']}
    Current Price: {result['price']}
    RSI: {result['rsi']}

    Explain clearly with short-term and long-term view.
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    state["result"] = response.content
    return state


# -----------------------------
# BUILD GRAPH
# -----------------------------
workflow = StateGraph(State)

workflow.add_node("router", router)
workflow.add_node("analysis", analysis_node)

workflow.set_entry_point("router")
workflow.add_edge("router", "analysis")
workflow.add_edge("analysis", END)

app = workflow.compile()


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("📊 AI Financial Research Agent")

query = st.text_input("Ask anything about a stock:")

if query:

    result = app.invoke({"query": query})

    st.subheader("🧠 Final Analysis")
    st.write(result["result"])
