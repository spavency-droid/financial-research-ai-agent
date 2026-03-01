!pip install -q langchain langchain-groq yfinance ta matplotlib
!pip install -U langchain langchain-community langchain-core
import os
import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt

from IPython.display import display
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

@tool
def analyze_stock(symbol: str) -> str:
    """
    Complete Technical Analysis Engine with Signals and Risk Classification
    """

    import numpy as np

    symbol = symbol.upper().strip()

    stock = yf.Ticker(symbol)
    data = stock.history(period="6mo")

    # ==========================
    # FUNDAMENTAL DATA
    # ==========================

    info = stock.info

    pe_ratio = info.get("trailingPE", "N/A")
    market_cap = info.get("marketCap", "N/A")
    eps = info.get("trailingEps", "N/A")

    if isinstance(market_cap, (int, float)):
      market_cap = f"{market_cap/1e9:.2f} Billion"

    if data.empty and not symbol.endswith(".NS"):
        symbol_ns = symbol + ".NS"
        stock = yf.Ticker(symbol_ns)
        data = stock.history(period="6mo")
        symbol = symbol_ns

    if data.empty:
        return f"No stock data found for {symbol}"

    # ==========================
    # INDICATORS
    # ==========================

    data["RSI"] = ta.momentum.RSIIndicator(data["Close"]).rsi()
    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()

    macd = ta.trend.MACD(data["Close"])
    data["MACD"] = macd.macd()
    data["MACD_SIGNAL"] = macd.macd_signal()

    latest = data.iloc[-1]

    rsi = latest["RSI"]
    ma50 = latest["MA50"]
    ma200 = latest["MA200"]
    macd_val = latest["MACD"]
    macd_signal = latest["MACD_SIGNAL"]

    # ==========================
    # SIGNAL LOGIC
    # ==========================

    signal = "HOLD"

    if rsi < 30 and ma50 > ma200:
        signal = "BUY"
    elif rsi > 70 and ma50 < ma200:
        signal = "SELL"
    elif ma50 > ma200:
        signal = "BULLISH HOLD"
    elif ma50 < ma200:
        signal = "BEARISH HOLD"

    # ==========================
    # RISK SCORE
    # ==========================

    volatility = data["Close"].pct_change().std()

    if volatility < 0.01:
        risk = "Low Risk"
    elif volatility < 0.025:
        risk = "Moderate Risk"
    else:
        risk = "High Risk"

    # ==========================
    # VISUALIZATION
    # ==========================

    # Price + MA
    plt.figure(figsize=(12,6))
    plt.plot(data["Close"], label="Close Price")
    plt.plot(data["MA50"], label="MA50")
    plt.plot(data["MA200"], label="MA200")
    plt.legend()
    plt.title(f"{symbol} Price with Moving Averages")
    plt.grid()
    plt.show()

    # RSI
    plt.figure(figsize=(12,4))
    plt.plot(data["RSI"])
    plt.axhline(70, linestyle="--")
    plt.axhline(30, linestyle="--")
    plt.title("RSI Indicator")
    plt.grid()
    plt.show()

    # MACD
    plt.figure(figsize=(12,4))
    plt.plot(data["MACD"], label="MACD")
    plt.plot(data["MACD_SIGNAL"], label="Signal Line")
    plt.legend()
    plt.title("MACD Indicator")
    plt.grid()
    plt.show()

    display(data.tail())

    return f"""
Stock: {symbol}
Current Price: {latest['Close']:.2f}

--- Technical Indicators ---
RSI: {rsi:.2f}
MA50: {ma50:.2f}
MA200: {ma200:.2f}
MACD: {macd_val:.2f}
MACD Signal: {macd_signal:.2f}

Signal: {signal}
Risk Level: {risk}

--- Fundamental Analysis ---
PE Ratio: {pe_ratio}
EPS: {eps}
Market Cap: {market_cap}
"""

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

tools = [analyze_stock]

system_prompt = """
You are a professional equity research analyst.

When analyzing a stock:

1. Explain technical indicators clearly.
2. Interpret RSI (overbought/oversold).
3. Explain moving average trend.
4. Interpret PE ratio (valuation insight).
5. Explain market cap category (Large/Mid/Small Cap).
6. If advice is requested, provide:
   - Short-term outlook
   - Long-term outlook
7. Keep explanation structured and professional.
"""
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)

company_symbol_map = {
    "INFOSYS": "INFY",
    "TCS": "TCS",
    "RELIANCE": "RELIANCE",
    "HDFC": "HDFCBANK",
    "HDFC BANK": "HDFCBANK",
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "CAPGEMINI": "CAP",
    "WELLS FARGO": "WFC",
    "TESLA": "TSLA"
}

import yfinance as yf

def get_ticker_from_name(company_name):
    try:
        search = yf.Search(company_name)
        results = search.quotes

        if not results:
            return None

        # Return first matching ticker
        return results[0]['symbol']

    except Exception as e:
        print("Ticker search failed:", e)
        return None

portfolio = {}

import re

while True:
    user_input = input("\nEnter command: ")

    if user_input.lower() == "exit":
        print("Goodbye 👋")
        break

    # ==========================
    # 📌 ADD STOCK TO PORTFOLIO
    # ==========================
    if user_input.lower().startswith("add"):
        parts = user_input.split()

        if len(parts) == 4:
            stock_symbol = parts[1].upper()
            quantity = int(parts[2])
            buy_price = float(parts[3])

            if stock_symbol in portfolio:
                old_qty = portfolio[stock_symbol]["quantity"]
                old_price = portfolio[stock_symbol]["avg_price"]

                new_qty = old_qty + quantity
                new_avg = ((old_qty * old_price) + (quantity * buy_price)) / new_qty

                portfolio[stock_symbol]["quantity"] = new_qty
                portfolio[stock_symbol]["avg_price"] = new_avg
            else:
                portfolio[stock_symbol] = {
                    "quantity": quantity,
                    "avg_price": buy_price
                }

            print(f"✅ Added {quantity} shares of {stock_symbol} at {buy_price}")
        else:
            print("Usage: add TICKER quantity buy_price")

        continue

    # ==========================
    # 📊 VIEW PORTFOLIO
    # ==========================
    if user_input.lower() == "portfolio":

        if not portfolio:
            print("⚠ Portfolio is empty.")
            continue

        print("\n========= YOUR PORTFOLIO =========")

        total_value = 0
        total_invested = 0

        for sym, data in portfolio.items():
            qty = data["quantity"]
            avg_price = data["avg_price"]

            stock = yf.Ticker(sym)
            hist = stock.history(period="5d")

            if hist.empty:
                print(f"{sym} - No data available")
                continue

            current_price = hist["Close"].iloc[-1]

            invested = qty * avg_price
            current_value = qty * current_price
            profit_loss = current_value - invested
            profit_percent = (profit_loss / invested) * 100

            total_value += current_value
            total_invested += invested

            print(f"\n📌 {sym}")
            print(f"  Shares: {qty}")
            print(f"  Avg Buy Price: {avg_price:.2f}")
            print(f"  Current Price: {current_price:.2f}")
            print(f"  Invested: {invested:.2f}")
            print(f"  Current Value: {current_value:.2f}")
            print(f"  Profit/Loss: {profit_loss:.2f} ({profit_percent:.2f}%)")
            print("-----------------------------------")

        total_profit = total_value - total_invested

        print("\n========== SUMMARY ==========")
        print(f"Total Invested: {total_invested:.2f}")
        print(f"Total Current Value: {total_value:.2f}")
        print(f"Total Profit/Loss: {total_profit:.2f}")
        print("=============================\n")

        continue

    # ==========================
    # 📈 STOCK ANALYSIS
    # ==========================
    upper_input = user_input.upper()

    symbol = company_symbol_map.get(upper_input, upper_input)

    stock = yf.Ticker(symbol)
    hist = stock.history(period="5d")

    if hist.empty:
        print("❌ Stock not found.")
        continue

    current_price = hist["Close"].iloc[-1]
    info = stock.info

    pe_ratio = info.get("trailingPE", "N/A")
    market_cap = info.get("marketCap", "N/A")
    eps = info.get("trailingEps", "N/A")

    print("\n========= STOCK ANALYSIS =========")
    print(f"Company: {symbol}")
    print(f"Current Price: {current_price:.2f}")
    print(f"PE Ratio: {pe_ratio}")
    print(f"EPS: {eps}")
    print(f"Market Cap: {market_cap}")
    print("==================================")
