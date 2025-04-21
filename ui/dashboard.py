import streamlit as st
import plotly.graph_objects as go
from strategies.bollinger_bands import BollingerBandsStrategy
from utils.technical import prepare_technical_data, convert_to_backtesting_format
from backtesting import Backtest
import vectorbt as vbt
from utils.helpers import get_portfolio_stats


def display_dashboard(data):
    st.title("📊 Stock Analysis and Backtesting Dashboard")
    st.write("Select an action in the sidebar to run a prediction or backtest.")

    with st.expander("Raw Stock Data"):
        st.dataframe(data.tail(50), use_container_width=True)

    # Placeholder for charts or summaries
    st.info("You can display SMA/RSI indicators or model summaries here.")

def plot_sma_chart(data):
    fig = go.Figure()
  
    
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close Price"))
    if 'SMA_50' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name="SMA 50"))
    if 'SMA_200' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], name="SMA 200"))
    st.plotly_chart(fig, use_container_width=True)

def plot_rsi_chart(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name="RSI", line=dict(color='blue')))
    fig.add_hline(y=70, line=dict(dash='dash', color='red'))
    fig.add_hline(y=30, line=dict(dash='dash', color='green'))
    st.plotly_chart(fig, use_container_width=True)

def plot_predictions(predictions_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=predictions_df['Date'], y=predictions_df['Predicted_Close'], name="Predicted Close", line=dict(color='purple')))
    st.plotly_chart(fig, use_container_width=True)

def show_bollinger_backtest(data, ticker):
    st.subheader("📊 Bollinger Bands Strategy")

    # 1. Prepare data
    tech_data = prepare_technical_data(data, ticker)
    # print(tech_data.head())
    # 2. Get VectorBT data
    vbt_data = tech_data.rename(
        columns={"Close": "close", f"Upper_20_2": "upper", f"Lower_20_2": "lower"}
    ).vbt

    
    # Access raw data using `.values` instead of `.to_numpy()`
    close_data = vbt_data["close"].obj.values  # Using values to get raw numpy array
    lower_band = vbt_data["lower"].obj.values
    upper_band = vbt_data["upper"].obj.values

    # 3. Define signals
    entries = close_data < lower_band  # Buy when price is below lower band
    exits = close_data > upper_band  # Sell when price is above upper band

    # 4. Run backtest
    pf = vbt.Portfolio.from_signals(close_data, entries, exits, freq="1D", fees=0.001)
    
    # Get portfolio stats
    stats = get_portfolio_stats(pf)

    return pf, stats
