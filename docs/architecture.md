# System Architecture

# Overview
This AI-Powered Financial Research Agent is built using LangGraph for agent orchestration and modular tool execution. The system processes user queries, dynamically selects tools, performs financial analysis, and generates structured insights.

# High-Level Workflow

1. User enters financial query.
2. LLM extracts intent and entities.
3. LangGraph routes execution through appropriate nodes.
4. Tools fetch stock, news, and economic data.
5. Data is processed and analyzed.
6. LLM synthesizes insights.
7. Streamlit renders interactive dashboard with:
  - Charts
  - Portfolio tracking
  - Profit/Loss metrics
  - Fundamental indicators

# Architectural Layers

# 1. Query Understanding Layer
LLM parses user input and extracts:
- Intent
- Stock symbols
- Time range

# 2. Agent Orchestration Layer (LangGraph)
- Maintains state
- Controls execution flow
- Invokes tools dynamically
- Handles retries and branching

# 3. Tool Layer
- Stock Data Tool (yfinance)
- News Retrieval Tool
- Technical Indicator Tool
- Risk Assessment Tool

# 4. Analysis Layer
- RSI
- Moving averages
- Sentiment scoring
- Risk metrics
- PE Ratio
- EPS
- Market Cap
- Fundamental health indicators

# 5. Presentation Layer
Streamlit dashboard visualizes:
- Charts
- Insights
- Risk summaries.

# 6. Portfolio Management Layer
Handles:
- Add/remove stocks
- Quantity tracking
- Average buy price
- Profit/Loss calculation
- Portfolio value aggregation
