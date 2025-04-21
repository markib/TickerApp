import streamlit as st
from models.linear_regression import LinearRegressionModel
from strategies.sma_crossover import SMACrossoverStrategy
from strategies.rsi_strategy import RSIStrategy  # Corrected import for RSI strategy
from strategies.bollinger_bands import BollingerBandsStrategy
from backtestings.engine import BacktestEngine
from utils.helpers import generate_date_range  # Importing generate_date_range
from data.fetch_yfinance import fetch_yfinance_data
from ui.sidebar import sidebar_inputs
from ui.dashboard import display_dashboard
import pandas as pd
from ui.dashboard import display_dashboard, plot_sma_chart, plot_rsi_chart, plot_predictions
from ui.dashboard import show_bollinger_backtest
import matplotlib.pyplot as plt
from utils.helpers import get_portfolio_stats
import plotly.graph_objects as go

# Streamlit setup
st.set_page_config(page_title="Stock Analysis Tool", layout="wide")
st.title("📈 Stock Price Prediction and Backtesting Tool")

# --- Sidebar Inputs ---
data, stock_ticker, selected_strategy  = sidebar_inputs()


# Main content
if data is not None and stock_ticker:
    if selected_strategy == "Bollinger Bands":
        results,stats = show_bollinger_backtest(data, stock_ticker)

        # Convert the stats Series to a DataFrame
        stats_df = stats.to_frame(name="Value")
        # Display the stats in Streamlit
        with st.expander("Advanced Analytics"):
            st.dataframe(
                stats_df.loc[
                    [
                        "Start",
                        "End",
                        "Period",
                        "Start Value",
                        "End Value",
                        "Total Return [%]",
                        "Benchmark Return [%]",
                        "Max Gross Exposure [%]",
                        "Total Fees Paid",
                        "Max Drawdown [%]",
                        "Max Drawdown Duration",
                        "Total Trades",
                        "Total Closed Trades",
                        "Total Open Trades",
                        "Open Trade PnL",
                        "Win Rate [%]",
                        "Best Trade [%]",
                        "Worst Trade [%]",
                        "Avg Winning Trade [%]",
                        "Avg Losing Trade [%]",
                        "Avg Winning Trade Duration",
                        "Avg Losing Trade Duration",
                        "Profit Factor",
                        "Expectancy",
                        "Sharpe Ratio",
                        "Calmar Ratio",
                        "Omega Ratio",
                        "Sortino Ratio",
                    ]
                ]
            )
            st.write("Trade Duration Analysis")
            durations = (
                results.trades.records["exit_idx"] - results.trades.records["entry_idx"]
            )
            st.bar_chart(durations)

            st.write("Return Distribution")

            returns_pct = results.trades.records["return"] * 100

            fig, ax = plt.subplots()
            ax.hist(returns_pct, bins=20, color="skyblue", edgecolor="black")
            ax.set_title("Return Distribution (%)")
            ax.set_xlabel("Return %")
            ax.set_ylabel("Number of Trades")

            st.pyplot(fig)
    elif selected_strategy == "SMA Crossover":
        # SMA Crossover Strategy
        sma_strategy = SMACrossoverStrategy(
            data, stock_ticker, short_window=50, long_window=200
        )
        backtest_engine = BacktestEngine(sma_strategy)
        results = backtest_engine.run_backtest()

        # Extract DataFrame from results tuple
        results_df = results[0] if isinstance(results, tuple) else results
        # Display Backtest Results
        st.write("Backtest Results", results)

        # Machine Learning Predictions - Linear Regression Example
        lr_model = LinearRegressionModel(data)
        predictions = lr_model.predict_next_days()

        # Display Predictions
        st.write("Predicted Next 5 Days", predictions)

        # Plotting SMA chart
        plot_sma_chart(results_df)

    elif selected_strategy == "RSI Strategy":
        # RSI Strategy Implementation
        rsi_strategy = RSIStrategy(data, ticker=stock_ticker, window=14)
        backtest_engine = BacktestEngine(rsi_strategy)
        results = backtest_engine.run_backtest()

        # Extract DataFrame and handle results
        results_df = results[0] if isinstance(results, tuple) else results

        # Display Backtest Results using the DataFrame
        st.write("Backtest Results", results_df)

        # Show RSI Strategy Performance
        total_return = (
            results[1]
            if isinstance(results, tuple)
            else results_df["StrategyReturns"].sum()
        )
        st.metric("Strategy Total Return", f"{total_return:.2%}")

        # Machine Learning Predictions
        lr_model = LinearRegressionModel(data)
        predictions = lr_model.predict_next_days()
        st.write("Predicted Next 5 Days", predictions)

        # Display RSI Chart
        st.subheader("RSI Strategy Visualization")
        plot_rsi_chart(results_df)

        # Show trade signals on price chart
        st.subheader("Price and Signals")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=results_df.index, y=results_df["Close"], name="Price")
        )
        st.plotly_chart(fig)
