import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(page_title="AI Financial Research Agent", layout="wide")

st.title("📊 AI Financial Research Agent")
st.markdown("LangGraph-powered intelligent stock analysis system")

# ----------------------------------
# LLM Setup
# ----------------------------------
llm = ChatOpenAI(
    temperature=0.3,
    model="gpt-3.5-turbo",
    openai_api_key=st.secrets["OPENAI_API_KEY"]
)

# ----------------------------------
# Agent State
# ----------------------------------
class AgentState(dict):
    pass

# ----------------------------------
# Tool 1: Fetch Market Data
# ----------------------------------
def fetch_data(state):
    symbol = state["symbol"]
    stock = yf.Ticker(symbol)
    data = stock.history(period="3mo")

    if data.empty:
        state["error"] = "Invalid stock symbol or no data available."
        return state

    state["data"] = data
    return state

# ----------------------------------
# Tool 2: Technical Indicators
# ----------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def technical_analysis(state):
    if "error" in state:
        return state

    data = state["data"]

    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()
    data["RSI"] = compute_rsi(data["Close"])

    state["data"] = data
    return state

# ----------------------------------
# Tool 3: AI Insight Generator
# ----------------------------------
def generate_insight(state):
    if "error" in state:
        return state

    latest = state["data"].iloc[-1]

    prompt = f"""
    Analyze the following stock data:

    Symbol: {state['symbol']}
    Latest Close Price: {latest['Close']}
    RSI: {latest['RSI']}
    MA20: {latest['MA20']}
    MA50: {latest['MA50']}

    Provide:
    1. Trend Analysis
    2. Risk Level (Low/Medium/High)
    3. Short-term Outlook
    4. Trading Suggestion (Buy/Hold/Sell with reasoning)
    """

    response = llm([HumanMessage(content=prompt)])
    state["analysis"] = response.content
    return state

# ----------------------------------
# Build LangGraph Workflow
# ----------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("fetch_data", fetch_data)
workflow.add_node("technical_analysis", technical_analysis)
workflow.add_node("generate_insight", generate_insight)

workflow.set_entry_point("fetch_data")
workflow.add_edge("fetch_data", "technical_analysis")
workflow.add_edge("technical_analysis", "generate_insight")

agent = workflow.compile()

# ----------------------------------
# Streamlit UI
# ----------------------------------
symbol = st.text_input("Enter Indian Stock Symbol (Example: RELIANCE.NS)")

if symbol:
    with st.spinner("Running AI Financial Analysis..."):
        result = agent.invoke({"symbol": symbol})

    if "error" in result:
        st.error(result["error"])
    else:
        data = result["data"]

        # Chart Section
        st.subheader("📈 Price Chart with Moving Averages")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close"))
        fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA20"))
        fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="MA50"))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price (INR)",
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

        # RSI
        st.subheader("📊 RSI Indicator")
        st.line_chart(data["RSI"])

        # AI Insight
        st.subheader("🧠 AI Generated Financial Insight")
        st.write(result["analysis"])
