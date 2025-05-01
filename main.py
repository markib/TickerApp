import streamlit as st
from models.linear_regression import LinearRegressionModel
from strategies.sma_crossover import SMACrossoverStrategy
from strategies.rsi_strategy import RSIStrategy  # Corrected import for RSI strategy
from strategies.bollinger_bands import BollingerBandsStrategy
from strategies.TripleMAStrategy import TripleMAStrategy  # Import TripleMAStrategy
from strategies.MLEnhancedTradingStrategy import MLEnhancedTradingStrategy  # Import MLEnhancedTradingStrategy
from strategies.AdvancedEnsembleStrategy import (
    AdvancedTradingStrategy,
)  # Import AdvancedEnsembleStrategy
from backtestings.engine import BacktestEngine
from utils.helpers import generate_date_range  # Importing generate_date_range
from data.fetch_yfinance import fetch_yfinance_data
from ui.sidebar import sidebar_inputs
from ui.dashboard import display_dashboard
import pandas as pd
from ui.dashboard import display_dashboard, plot_sma_chart, plot_rsi_chart, plot_predictions , plot_advanced_strategy
from ui.dashboard import show_bollinger_backtest
import matplotlib.pyplot as plt
from matplotlib import colors
from utils.helpers import get_portfolio_stats
import plotly.graph_objects as go
import numpy as np

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

    elif selected_strategy == "Triple MA Crossover":
        try:
            # Initialize and run strategy
            triple_ma_strategy = TripleMAStrategy(
                data, stock_ticker, short_window=10, medium_window=50, long_window=200
            )
            backtest_engine = BacktestEngine(triple_ma_strategy)
            results = backtest_engine.run_backtest()

            # Extract results
            results_df = results[0] if isinstance(results, tuple) else results
            cumulative_return = results[1] if isinstance(results, tuple) else None

            # Create tabs for better organization
            tab1, tab2, tab3 = st.tabs(["Backtest Results", "Performance Metrics", "Predictions"])

            with tab1:
                st.subheader("Triple MA Crossover Strategy Results")

                # Display metrics in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Cumulative Return", f"{cumulative_return*100:.2f}%" if cumulative_return else "N/A")
                with col2:
                    total_trades = len(results_df[results_df['Signal'].diff() != 0])
                    st.metric("Total Trades", total_trades)
                with col3:
                    win_rate = (results_df[results_df['StrategyReturns'] > 0]['StrategyReturns'].count() / 
                            results_df[results_df['StrategyReturns'] != 0]['StrategyReturns'].count()) * 100
                    st.metric("Win Rate", f"{win_rate:.2f}%" if not pd.isna(win_rate) else "N/A")

                # Display the results table with formatting
                st.dataframe(
                    results_df.tail(10)
                    .style.format(
                        {
                            "Close": "{:.2f}",
                            "SMA10": "{:.2f}",
                            "SMA50": "{:.2f}",
                            "SMA200": "{:.2f}",
                            "Returns": "{:.2%}",
                            "StrategyReturns": "{:.2%}",
                        }
                    )
                    .map(
                        lambda x: (
                            "color: green"
                            if x == 1
                            else "color: red" if x == -1 else ""
                        ),
                        subset=["Signal"],
                    ),
                    height=400,
                )

            with tab2:
                st.subheader("Performance Analysis")

                # Calculate additional metrics
                if not results_df.empty:
                    # Annualized Return
                    days = len(results_df)
                    annualized_return = ((1 + cumulative_return) ** (252/days) - 1) * 100 if cumulative_return and days > 0 else 0

                    # Max Drawdown
                    cum_returns = (1 + results_df['StrategyReturns']).cumprod()
                    peak = cum_returns.cummax()
                    drawdown = (cum_returns - peak) / peak
                    max_drawdown = drawdown.min() * 100

                    # Sharpe Ratio (assuming risk-free rate = 0)
                    sharpe_ratio = results_df['StrategyReturns'].mean() / results_df['StrategyReturns'].std() * np.sqrt(252)

                    metrics_df = pd.DataFrame({
                        'Metric': ['Cumulative Return', 'Annualized Return', 'Max Drawdown', 'Sharpe Ratio', 'Win Rate'],
                        'Value': [
                            f"{cumulative_return*100:.2f}%" if cumulative_return else "N/A",
                            f"{annualized_return:.2f}%",
                            f"{max_drawdown:.2f}%",
                            f"{sharpe_ratio:.2f}",
                            f"{win_rate:.2f}%" if not pd.isna(win_rate) else "N/A"
                        ]
                    })

                    st.table(metrics_df.style.set_properties(**{'text-align': 'left'}))

            with tab3:
                st.subheader("Machine Learning Predictions")
                try:
                    lr_model = LinearRegressionModel(data)
                    predictions = lr_model.predict_next_days()

                    if predictions is not None and not predictions.empty:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Predicted Prices:")
                            styled_df = (
                                predictions.reset_index()
                                .style.format(
                                    {
                                        "Date": lambda x: x.strftime("%Y-%m-%d"),
                                        "Predicted_Close": "{:.2f}",
                                    }
                                )
                                .hide()
                                .map(
                                    lambda x: (
                                        "color: green"
                                        if x > 0
                                        else "color: red" if x < 0 else ""
                                    ),
                                    subset=["Predicted_Close"],
                                )
                            )
                            st.dataframe(
                                styled_df, height=min(300, 35 * len(predictions))
                            )
                        with col2:
                            plot_predictions(predictions)
                    else:
                        st.warning("No predictions generated")
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

            # Plotting
            st.subheader("Strategy Visualization")
            plot_predictions(results_df)

        except Exception as e:
            st.error(f"Error running strategy: {str(e)}")

    elif selected_strategy == "ML Enhanced Strategy":
        try:
            # Initialize and run strategy
            ml_strategy = MLEnhancedTradingStrategy(data, stock_ticker)
            results_df, metrics = ml_strategy.execute()

            # Calculate cumulative return from strategy returns
            cumulative_return = results_df['CumulativeReturns'].iloc[-1] - 1

            # Create tabs for organization
            tab1, tab2, tab3 = st.tabs(["Backtest Results", "Performance Metrics", "Model Insights"])

            with tab1:
                st.subheader("ML Enhanced Strategy Results")

                # Display metrics in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Cumulative Return", f"{cumulative_return*100:.2f}%")
                with col2:
                    total_trades = len(results_df[results_df['Signal'].diff() != 0])
                    st.metric("Total Trades", total_trades)
                with col3:
                    win_rate = (results_df[results_df['StrategyReturns'] > 0]['StrategyReturns'].count() / 
                            results_df[results_df['StrategyReturns'] != 0]['StrategyReturns'].count()) * 100
                    st.metric("Win Rate", f"{win_rate:.2f}%")

                # Display results table
                st.dataframe(
                    results_df.tail(10)
                    .style.format({
                        "Close": "{:.2f}",
                        "StrategyReturns": "{:.2%}",
                        "CumulativeReturns": "{:.2%}"
                    })
                    .map(
                        lambda x: "color: green" if x == 1 else "color: red" if x == -1 else "",
                        subset=["Signal"]
                    ),
                    height=400
                )

            with tab2:
                st.subheader("Performance Analysis")

                # Calculate performance metrics
                days = len(results_df)
                annualized_return = ((1 + cumulative_return) ** (252/days) - 1) * 100

                # Max Drawdown calculation
                cum_returns = results_df['CumulativeReturns']
                peak = cum_returns.cummax()
                drawdown = (cum_returns - peak) / peak
                max_drawdown = drawdown.min() * 100

                # Sharpe Ratio
                sharpe_ratio = results_df['StrategyReturns'].mean() / results_df['StrategyReturns'].std() * np.sqrt(252)

                metrics_df = pd.DataFrame({
                    'Metric': ['Cumulative Return', 'Annualized Return', 'Max Drawdown', 
                            'Sharpe Ratio', 'Win Rate', 'Total Trades'],
                    'Value': [
                        f"{cumulative_return*100:.2f}%",
                        f"{annualized_return:.2f}%",
                        f"{max_drawdown:.2f}%",
                        f"{sharpe_ratio:.2f}",
                        f"{win_rate:.2f}%",
                        total_trades
                    ]
                })

                st.table(metrics_df.style.set_properties(**{'text-align': 'left'}))

            with tab3:
                st.subheader("Model Insights")

                # Model performance metrics
                st.write("#### Classification Report (Test Set)")
                st.json(metrics['test_report'])

                # Feature importances
                st.write("#### Feature Importances")
                features_df = pd.DataFrame.from_dict(
                    metrics['feature_importances'], 
                    orient='index', 
                    columns=['Importance']
                ).sort_values('Importance', ascending=False)

                st.bar_chart(features_df)
                st.dataframe(features_df.style.format({'Importance': '{:.2%}'}))

            # Plotting
            st.subheader("Strategy Visualization")
            plot_predictions(results_df[['Close', 'Signal']].rename(columns={'Signal': 'Predicted_Close'}))

        except Exception as e:
            st.error(f"Error running ML Enhanced Strategy: {str(e)}")

    elif selected_strategy == "Advanced Ensemble Strategy":
        try:
            # Initialize and run strategy
            strategy = AdvancedTradingStrategy(
                data=data,
                ticker=stock_ticker,
                initial_capital=100000,
                short_window=10,
                medium_window=50,
                long_window=200,
                volatility_window=20,
                rsi_window=14,
                min_hold_days=3,
                max_position_size=0.25,
                slippage_pct=0.0005,
            )
            results, metrics, wf_results = strategy.execute()

            # print("\nWalk-Forward Validation Results:")
            # print(wf_results.groupby('model').agg({'accuracy': ['mean', 'std']}))

            # print("\nStrategy Metrics:")
            # for k, v in metrics.items():
            #     if k == 'feature_importances':
            #         print("\nFeature Importances:")
            #         for f, imp in sorted(v.items(), key=lambda x: x[1], reverse=True):
            #             print(f"{f}: {imp:.2%}")
            #     elif isinstance(v, float):
            #         print(f"{k:>25}: {v:.2%}")
            #     else:
            #         print(f"{k:>25}: {v}")

            # Check results
            # print(results.info())
            # print(metrics)

            # Handle results - check if it's a tuple or DataFrame
            if isinstance(results, tuple):
                results_df = results[0]  # First element is the results DataFrame
                performance_metrics = results[1]  # Second element is performance metrics (if exists)
            else:
                results_df = results  # Just the DataFrame
                performance_metrics = metrics  # Use the separate metrics variable
                results_df.loc[:, "Returns"] = (
                    results_df["Close"].pct_change().fillna(0)
                )
                results_df["StrategyReturns"] = results["Close"].pct_change() * results[
                    "PositionSize"
                ].shift(1)
                results_df.loc[:, "StrategyReturns"] = results_df[
                    "StrategyReturns"
                ].fillna(0)

            # Auto-scale based on 5th/95th percentiles
            returns_min = results_df["StrategyReturns"].quantile(0.05)
            returns_max = results_df["StrategyReturns"].quantile(0.95)                

            # Create tabs for organization
            tab1, tab2, tab3, tab4 = st.tabs(["Results", "Advanced Metrics", "Model Analysis", "Risk Analysis"])

            with tab1:
                st.subheader("Strategy Performance Overview")

                # Key metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Cumulative Return", f"{performance_metrics['cumulative_return']*100:.2f}%")
                with col2:
                    st.metric("Annualized Return", f"{performance_metrics['annualized_return']:.2f}%")
                with col3:
                    st.metric("Max Drawdown", f"{performance_metrics['max_drawdown']:.2f}%")
                with col4:
                    st.metric("Sharpe Ratio", f"{performance_metrics['sharpe_ratio']:.2f}")

                # Enhanced results table
                st.dataframe(
                    results_df.tail(10)
                    .style.format(
                        {
                            "Close": "{:.2f}",
                            "Upper_Band": "{:.2f}",
                            "Lower_Band": "{:.2f}",
                            "Returns": "{:.2%}",
                            "StrategyReturns": "{:.2%}",
                            "PositionSize": "{:.0%}",
                            "Volatility": "{:.2%}",
                            "Model_Confidence": "{:.2f}",
                        }
                    )
                    .map(
                        lambda x: (
                            "color: green" if x > 0 else "color: red" if x < 0 else ""
                        ),
                        subset=["Signal"],
                    )
                    .background_gradient(
                        cmap="RdYlGn",
                        subset=["StrategyReturns"],
                        vmin=returns_min,
                        vmax=returns_max,
                    )
                    .background_gradient(
                        cmap="Blues",
                        subset=["Model_Confidence"],
                        vmin=0,  # Low confidence
                        vmax=1,  # High confidence
                    ),
                    height=450,
                    use_container_width=True,
                )

            with tab2:
                st.subheader("Advanced Performance Metrics")

                # Create two columns for metrics
                col1, col2 = st.columns(2)

                with col1:
                    st.write("#### Risk-Adjusted Metrics")
                    risk_metrics = pd.DataFrame({
                        'Metric': ['Sortino Ratio', 'Calmar Ratio', 'Omega Ratio', 
                                'Tail Ratio', 'Profit Factor', 'Common Sense Ratio'],
                        'Value': [
                            f"{performance_metrics['sortino_ratio']:.2f}",
                            f"{performance_metrics['calmar_ratio']:.2f}",
                            f"{performance_metrics['omega_ratio']:.2f}",
                            f"{performance_metrics['tail_ratio']:.2f}",
                            f"{performance_metrics['profit_factor']:.2f}",
                            f"{performance_metrics['common_sense_ratio']:.2f}"
                        ]
                    })
                    st.table(risk_metrics.style.set_properties(**{'text-align': 'left'}))

                with col2:
                    st.write("#### Trade Analysis")
                    trade_metrics = pd.DataFrame({
                        'Metric': ['Win Rate', 'Avg Winning Trade', 'Avg Losing Trade',
                                'Best Trade', 'Worst Trade', 'Long/Short Ratio'],
                        'Value': [
                            f"{performance_metrics['win_rate']:.2f}%",
                            f"{performance_metrics['avg_winning_trade']:.2%}",
                            f"{performance_metrics['avg_losing_trade']:.2%}",
                            f"{performance_metrics['best_trade']:.2%}",
                            f"{performance_metrics['worst_trade']:.2%}",
                            f"{performance_metrics['long_short_ratio']:.2f}"
                        ]
                    })
                    st.table(trade_metrics.style.set_properties(**{'text-align': 'left'}))

                    # Win/Loss Distribution
                    st.write("#### Win/Loss Distribution")
                    win_loss_data = pd.DataFrame({
                        'Type': ['Winning Trades', 'Losing Trades'],
                        'Count': [
                            performance_metrics['winning_trades'],
                            performance_metrics['losing_trades']
                        ]
                    })
                    st.bar_chart(win_loss_data.set_index('Type'))

            with tab3:
                st.subheader("Model Analysis")

                # Model components
                col1, col2 = st.columns(2)

                with col1:
                    st.write("#### Feature Importances")
                    features_df = pd.DataFrame.from_dict(
                        performance_metrics['feature_importances'],
                        orient='index',
                        columns=['Importance']
                    ).sort_values('Importance', ascending=False)

                    st.bar_chart(features_df)
                    st.dataframe(
                        features_df.style.format({'Importance': '{:.2%}'}),
                        height=300
                    )

                with col2:
                    st.write("#### Model Confidence")
                    st.line_chart(results_df[['Model_Confidence']].tail(100))

                    st.write("#### Ensemble Weights")
                    weights_df = pd.DataFrame.from_dict(
                        performance_metrics['ensemble_weights'],
                        orient='index',
                        columns=['Weight']
                    )
                    st.dataframe(
                        weights_df.style.format({'Weight': '{:.2%}'}),
                        height=200
                    )

            with tab4:
                st.subheader("Risk Analysis")

                # Risk metrics
                col1, col2 = st.columns(2)

                with col1:
                    st.write("#### Drawdown Analysis")
                    drawdown_data = pd.DataFrame({
                        'Metric': ['Max Drawdown', 'Avg Drawdown', 'Drawdown Duration', 
                                'Recovery Duration', 'Pain Index', 'Ulcer Index'],
                        'Value': [
                            f"{performance_metrics['max_drawdown']:.2f}%",
                            f"{performance_metrics['avg_drawdown']:.2f}%",
                            f"{performance_metrics['drawdown_duration']} days",
                            f"{performance_metrics['recovery_duration']} days",
                            f"{performance_metrics['pain_index']:.2f}",
                            f"{performance_metrics['ulcer_index']:.2f}"
                        ]
                    })
                    st.table(drawdown_data)

                with col2:
                    st.write("#### Value-at-Risk (VaR)")
                    var_data = pd.DataFrame({
                        'Confidence Level': ['95%', '99%'],
                        'Daily VaR': [
                            f"{performance_metrics['var_95']:.2%}",
                            f"{performance_metrics['var_99']:.2%}",
                            # f"{performance_metrics['var_999']:.2%}"
                        ]
                    })
                    st.table(var_data)

                    st.write("#### Expected Shortfall")
                    es_data = pd.DataFrame({
                        'Confidence Level': ['95%', '99%'],
                        'Expected Shortfall': [
                            f"{performance_metrics['es_95']:.2%}",
                            f"{performance_metrics['es_99']:.2%}"
                        ]
                    })
                    st.table(es_data)

            # Plotting
            st.subheader("Strategy Visualization")

            plot_advanced_strategy(results_df)

        except Exception as e:
            st.error(f"Error running Advanced Strategy: {str(e)}")        
