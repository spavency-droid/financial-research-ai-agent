# AI Financial Research Agent - Workflow

## Overview
This document explains the LangGraph-based agent orchestration flow.

User Input (Streamlit UI)
        ↓
LLM Router Node (Intent Classification)
        ↓
Market Data Fetch Tool (yfinance)
        ↓
Conditional Routing
   ├── Technical Analysis Tool
   └── Risk Analysis Tool
        ↓
LLM Insight Generator
        ↓
Structured Financial Report + Visualization
