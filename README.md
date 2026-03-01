# financial-research-ai-agent
## Overview
An AI-powered Financial Research Agent built using LangGraph for intelligent orchestration and Streamlit for interactive visualization.

The system dynamically selects tools, performs both technical and fundamental analysis, tracks portfolio performance, and generates structured financial insights through an interactive web dashboard.

## Tech Stack
- Python
- LangChain
- LangGraph
- Streamlit
- yfinance
- Plotly/Matplotlib

## System Architecture
![Architecture](docs/architecture.png)

Detailed explanation available in:
[Architecture Documentation](docs/architecture.md)

### Features
##Stock Analysis
- Real-time stock data analysis
- Technical indicators (RSI, Moving Averages)
- Fundamental metrics(PE Ratio, EPS, Market Cap)

## Conceptual Workflow
User Input → LangGraph Router → Data Fetch Tool → 
Technical Analysis Tool → Insight Generator (LLM) → 
Structured Financial Report

##Portfolio Management (v2.0)
- Add stocks with quantity
- Track holdings
- Calculate Profit/Loss
- Portfolio value aggregation

## AI Agent Capabilities
- Dynamic tool selection using LangGraph
- Query understanding via LLM
- Modular graph-based orchestration
- Risk assessment
- Sentiment analysis

## Deployment
https://financial-research-ai-agent-fbdpwwykgwz5hxbuiuzeng.streamlit.app/

##Project Evolution
- v1.0 – CLI-based Financial Research Agent
- v2.0 – Interactive Streamlit Dashboard + Portfolio Management

## Future Improvements
- Vector database integration
- RAG for financial reports
- Multi-agent collaboration
- Docker deployment
- Backtesting engine
