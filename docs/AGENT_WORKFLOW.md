# AI Financial Research Agent - Workflow

## Overview
This document explains the LangGraph-based agent orchestration flow.

## Conceptual Agent Workflow

1. User submits a financial query and stock symbol via Streamlit.
2. A Router Node (LLM-powered) classifies the intent:
   - Technical Analysis
   - Risk Analysis
3. Market data is fetched using yfinance.
4. Based on routing decision:
   - Technical indicators (RSI, Moving Averages) are computed
   - OR volatility-based risk metrics are calculated
5. The Insight Generator (LLM) produces structured financial reasoning.
6. Results are visualized and displayed in the UI.
