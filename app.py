import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.title("Indian Stock Research Assistant")

symbol = st.text_input("Enter Indian Stock Symbol (Example: RELIANCE.NS)")

if symbol:
    stock = yf.Ticker(symbol)
    data = stock.history(period="1mo")

    st.write("Stock Data:")
    st.write(data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data["Close"],
        mode='lines',
        name='Close Price'
    ))

    fig.update_layout(
        title=f"{symbol} Stock Price (1 Month)",
        xaxis_title="Date",
        yaxis_title="Price (INR)"
    )

    st.plotly_chart(fig)
