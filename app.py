import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Financial Research AI Agent", layout="wide")

st.title("🌍 Financial Research AI Agent")
st.markdown("Analyze global stocks and track your portfolio in real time.")

# ==============================
# SESSION STATE FOR PORTFOLIO
# ==============================
if "portfolio" not in st.session_state:
    st.session_state.portfolio = {}

# ==============================
# SIDEBAR - PORTFOLIO SECTION
# ==============================
st.sidebar.header("📁 Portfolio Management")

with st.sidebar.form("add_stock_form"):
    st.subheader("Add Stock")
    p_symbol = st.text_input("Stock Symbol (e.g., TSLA, RELIANCE.NS)")
    p_qty = st.number_input("Quantity", min_value=1, step=1)
    p_price = st.number_input("Buy Price", min_value=0.0, step=0.1)
    submit = st.form_submit_button("Add to Portfolio")

    if submit and p_symbol:
        symbol = p_symbol.upper()

        if symbol in st.session_state.portfolio:
            old_qty = st.session_state.portfolio[symbol]["quantity"]
            old_price = st.session_state.portfolio[symbol]["avg_price"]

            new_qty = old_qty + p_qty
            new_avg = ((old_qty * old_price) + (p_qty * p_price)) / new_qty

            st.session_state.portfolio[symbol]["quantity"] = new_qty
            st.session_state.portfolio[symbol]["avg_price"] = new_avg
        else:
            st.session_state.portfolio[symbol] = {
                "quantity": p_qty,
                "avg_price": p_price
            }

        st.sidebar.success(f"{symbol} added to portfolio.")

# ==============================
# MAIN STOCK ANALYSIS SECTION
# ==============================
symbol = st.text_input("🔍 Enter Stock Symbol (e.g., TSLA, AAPL, RELIANCE.NS)")

if symbol:
    stock = yf.Ticker(symbol.upper())
    data = stock.history(period="1mo")

    if not data.empty:
        info = stock.info
        current_price = data["Close"].iloc[-1]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Current Price", f"{current_price:.2f}")
        col2.metric("PE Ratio", info.get("trailingPE", "N/A"))
        col3.metric("EPS", info.get("trailingEps", "N/A"))
        col4.metric("Market Cap", info.get("marketCap", "N/A"))

        # Price Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Close Price"
        ))

        fig.update_layout(
            title=f"{symbol.upper()} - 1 Month Price Trend",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Stock not found.")

# ==============================
# PORTFOLIO DISPLAY SECTION
# ==============================
st.subheader("📊 Portfolio Overview")

if st.session_state.portfolio:
    total_invested = 0
    total_value = 0

    for sym, data in st.session_state.portfolio.items():
        stock = yf.Ticker(sym)
        hist = stock.history(period="5d")

        if hist.empty:
            continue

        current_price = hist["Close"].iloc[-1]
        qty = data["quantity"]
        avg_price = data["avg_price"]

        invested = qty * avg_price
        current_val = qty * current_price
        profit_loss = current_val - invested

        total_invested += invested
        total_value += current_val

        st.write(f"### {sym}")
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Shares", qty)
        col2.metric("Avg Buy Price", f"{avg_price:.2f}")
        col3.metric("Current Price", f"{current_price:.2f}")
        col4.metric("P/L", f"{profit_loss:.2f}")

        st.markdown("---")

    total_pl = total_value - total_invested

    st.write("## Portfolio Summary")
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Invested", f"{total_invested:.2f}")
    col2.metric("Current Value", f"{total_value:.2f}")
    col3.metric("Total Profit/Loss", f"{total_pl:.2f}")

else:
    st.info("Portfolio is empty. Add stocks from sidebar.")
