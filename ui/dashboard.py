import streamlit as st
import plotly.graph_objects as go
from strategies.bollinger_bands import BollingerBandsStrategy
from utils.technical import prepare_technical_data, convert_to_backtesting_format
from backtesting import Backtest
import vectorbt as vbt
from utils.helpers import get_portfolio_stats
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


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
    """Plot predictions using Plotly with enhanced styling."""
    fig = go.Figure()
    # Handle datetime conversion properly
    if "Date" in predictions_df.columns:
        dates = np.array(predictions_df["Date"].dt.to_pydatetime())
    else:
        dates = np.array(predictions_df.index.to_pydatetime())

    # Get values column
    if "Predicted_Close" in predictions_df.columns:
        values = predictions_df["Predicted_Close"]
    else:
        values = predictions_df.iloc[:, 0]
    fig.add_trace(
        go.Scatter(
            x=(
                predictions_df.index
                if "Date" not in predictions_df.columns
                else predictions_df["Date"]
            ),
            y=(
                predictions_df["Predicted_Close"]
                if "Predicted_Close" in predictions_df.columns
                else predictions_df.iloc[:, 0]
            ),
            name="Predicted Close",
            line=dict(color="purple", width=2),
            mode="lines+markers",
            marker=dict(size=8),
        )
    )
    fig.update_layout(
        title="Future Price Predictions",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_white",
    )
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


def plot_advanced_strategy(results_df):
    """
    Create an interactive visualization of the advanced trading strategy with:
    - Price and moving averages
    - Buy/sell signals
    - Position sizing
    - Model confidence
    - Volatility bands
    - Cumulative returns
    """
    # Create figure with subplots
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(
            "Price with Trading Signals",
            "Model Confidence",
            "Position Sizing",
            "Cumulative Returns",
        ),
    )

    # Convert index to datetime if not already
    if not isinstance(results_df.index, pd.DatetimeIndex):
        results_df = results_df.set_index("Date")

    # 1. Price and Signals Plot
    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=results_df["Close"],
            name="Price",
            line=dict(color="#1f77b4", width=2),
            opacity=0.8,
        ),
        row=1,
        col=1,
    )

    # Add buy/sell signals
    buy_signals = results_df[results_df["Signal"] == 1]
    sell_signals = results_df[results_df["Signal"] == -1]

    fig.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=buy_signals["Close"],
            name="Buy",
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                color="green",
                size=10,
                line=dict(width=1, color="DarkSlateGrey"),
            ),
            opacity=0.8,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=sell_signals["Close"],
            name="Sell",
            mode="markers",
            marker=dict(
                symbol="triangle-down",
                color="red",
                size=10,
                line=dict(width=1, color="DarkSlateGrey"),
            ),
            opacity=0.8,
        ),
        row=1,
        col=1,
    )

    # Add volatility bands if available
    if "Upper_Band" in results_df.columns:
        fig.add_trace(
            go.Scatter(
                x=results_df.index,
                y=results_df["Upper_Band"],
                name="Upper Vol Band",
                line=dict(color="rgba(200,200,200,0.5)", width=1),
                opacity=0.5,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=results_df.index,
                y=results_df["Lower_Band"],
                name="Lower Vol Band",
                fill="tonexty",
                fillcolor="rgba(200,200,200,0.1)",
                line=dict(color="rgba(200,200,200,0.5)", width=1),
                opacity=0.5,
            ),
            row=1,
            col=1,
        )

    # 2. Model Confidence Plot
    if "Model_Confidence" in results_df.columns:
        fig.add_trace(
            go.Scatter(
                x=results_df.index,
                y=results_df["Model_Confidence"],
                name="Confidence",
                line=dict(color="purple", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(128,0,128,0.1)",
            ),
            row=2,
            col=1,
        )

        # Add confidence thresholds
        fig.add_hline(
            y=0.6, line=dict(color="green", width=1, dash="dot"), row=2, col=1
        )
        fig.add_hline(y=0.4, line=dict(color="red", width=1, dash="dot"), row=2, col=1)

    # 3. Position Sizing Plot
    if "PositionSize" in results_df.columns:
        fig.add_trace(
            go.Bar(
                x=results_df.index,
                y=results_df["PositionSize"],
                name="Position Size",
                marker_color=np.where(
                    results_df["Signal"] == 1,
                    "rgba(50,205,50,0.7)",  # Green for long
                    "rgba(220,20,60,0.7)",  # Red for short
                ),
            ),
            row=3,
            col=1,
        )

    # 4. Cumulative Returns Plot
    if "CumulativeReturns" in results_df.columns:
        fig.add_trace(
            go.Scatter(
                x=results_df.index,
                y=results_df["CumulativeReturns"],
                name="Strategy Returns",
                line=dict(color="#17BECF", width=2),
            ),
            row=4,
            col=1,
        )

        # Add benchmark (buy-and-hold) comparison
        buy_hold = (1 + results_df["Returns"]).cumprod()
        fig.add_trace(
            go.Scatter(
                x=results_df.index,
                y=buy_hold,
                name="Buy & Hold",
                line=dict(color="#7F7F7F", width=1.5, dash="dot"),
            ),
            row=4,
            col=1,
        )

    # Update layout
    fig.update_layout(
        height=900,
        title_text="Advanced Trading Strategy Analysis",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, b=50, t=100, pad=4),
        plot_bgcolor="rgba(240,240,240,0.8)",
    )

    # Update y-axis titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Confidence", row=2, col=1, range=[0, 1])
    fig.update_yaxes(title_text="Position Size", row=3, col=1)
    fig.update_yaxes(title_text="Cumulative Returns", row=4, col=1)

    # Add trading statistics annotation
    if "StrategyReturns" in results_df.columns:
        total_return = results_df["CumulativeReturns"].iloc[-1] - 1
        sharpe_ratio = (
            results_df["StrategyReturns"].mean()
            / results_df["StrategyReturns"].std()
            * np.sqrt(252)
        )

        fig.add_annotation(
            x=0.01,
            y=1.15,
            xref="paper",
            yref="paper",
            text=f"<b>Strategy Performance:</b> Total Return: {total_return:.2%} | Sharpe: {sharpe_ratio:.2f}",
            showarrow=False,
            font=dict(size=12),
        )

    st.plotly_chart(fig, use_container_width=True)
