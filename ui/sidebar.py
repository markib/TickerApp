import streamlit as st
from data.fetch_yfinance import fetch_yfinance_data, get_tickers
import pandas as pd

def sidebar_inputs():
    st.sidebar.header("Strategy Configuration")
    # Data selection
    tickers = get_tickers(); 
    default_ticker = "AAPL"
    # Ensure the default exists in the list
    default_index = tickers.index(default_ticker) if default_ticker in tickers else 0
    ticker = st.sidebar.selectbox("Select Ticker", tickers, index=default_index)

    # Date range selection
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))

    # Strategy selection
    strategy = st.sidebar.selectbox(
        "Select Strategy",
        ["Bollinger Bands", "SMA Crossover", "RSI Strategy"],
        index=0
    )

    # Load data (mock function - replace with your data loader)
    data = load_data(ticker, start_date, end_date)

    return data, ticker, strategy
    # stock_ticker = st.sidebar.text_input("Enter Stock Ticker", "TSLA")
    # prediction_days = st.sidebar.slider("Number of Prediction Days", 1, 30, 5)
    # sma_short_period = st.sidebar.slider("Short-Term SMA Period (e.g., SMA50)", 5, 100, 50)  # Short-term SMA
    # sma_long_period = st.sidebar.slider("Long-Term SMA Period (e.g., SMA200)", 100, 300, 200)  # Long-term SMA

    # return stock_ticker, prediction_days, sma_short_period, sma_long_period
def load_data(ticker, start_date, end_date):
    """Mock data loader - replace with your actual data loading logic"""
    # Fetch Stock Data
    data = fetch_yfinance_data(ticker, start_date=start_date, end_date=end_date)

    return data  # Your actual DataFrame loading logic here
