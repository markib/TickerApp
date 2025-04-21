import numpy as np
import pandas as pd
import vectorbt as vbt


def prepare_technical_data(
    df, ticker=None, bb_window=20, bb_std=2, rsi_window=14
):
    """
    Calculate and add enhanced technical indicators to the DataFrame for VectorBT (v0.27.2).

    Args:
        df (pd.DataFrame): DataFrame containing OHLCV data.
                           Can be a single ticker or a MultiIndex DataFrame.
        ticker (str, optional): If df is a MultiIndex DataFrame, specify the ticker to process.
                                Defaults to None (processes the entire DataFrame if single-indexed).
        bb_window (int, optional): Window for Bollinger Bands calculation. Defaults to 20.
        bb_std (int, optional): Number of standard deviations for Bollinger Bands. Defaults to 2.
        rsi_window (int, optional): Window for RSI calculation. Defaults to 14.

    Returns:
        pd.DataFrame: DataFrame with added technical indicators (Upper, Lower, SMA, RSI).
                      Returns a DataFrame with NaN values removed.
    """
    if isinstance(df.columns, pd.MultiIndex) and ticker:
        data = df.xs(ticker, level="Ticker", axis=1).copy()
    else:
        data = df.copy()

    # Rename columns to VectorBT convention
    vbt_data = data.rename(
        columns={
            "Close": "close",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Volume": "volume",
        }
    ).vbt

    # 1. Bollinger Bands Calculation

    close_series = vbt_data["close"].obj
    bb_bands = vbt.BBANDS.run(close_series, window=bb_window, alpha=bb_std)
    data[f"Upper_{bb_window}_{bb_std}"] = bb_bands.upper
    data[f"Lower_{bb_window}_{bb_std}"] = bb_bands.lower
    data[f"SMA_{bb_window}"] = bb_bands.middle

    # 2. RSI Calculation
    rsi = vbt.indicators.RSI.run(vbt_data["close"].obj, window=rsi_window)
    data[f"RSI_{rsi_window}"] = rsi.rsi

    # --- Potential Enhancements ---

    # 3. Moving Average Convergence Divergence (MACD)
    macd = vbt.indicators.MACD.run(vbt_data["close"].obj)
    data["MACD"] = macd.macd  # Access MACD values directly
    data["MACD_Signal"] = macd.signal  # Access MACD signal line
    data["MACD_Hist"] = macd.hist  # Access MACD histogram

    # 4. Exponential Moving Averages (EMAs)

    ema_short = vbt.MA.run(vbt_data["close"].obj, window=12, ewm=True)
    ema_long = vbt.MA.run(vbt_data["close"].obj, window=26, ewm=True)
    data["EMA_12"] = ema_short.ma  # Access EMA values directly
    data["EMA_26"] = ema_long.ma  # Access EMA values directly

    # 5. Stochastic Oscillator (%K and %D)

    stoch = vbt.STOCH .run(
        vbt_data["high"].obj, vbt_data["low"].obj, vbt_data["close"].obj, k_window=14, d_window=3
    )
    data["Stoch_K"] = stoch.percent_k  # Access %K values
    data["Stoch_D"] = stoch.percent_d  # Access %D values

    # 6. Average True Range (ATR)
    atr = vbt.indicators.ATR.run(
        vbt_data["high"].obj, vbt_data["low"].obj, vbt_data["close"].obj
    )
    data["ATR"] = atr.atr  # Access ATR values directly

    return data.dropna()


def convert_to_backtesting_format(df):
    """Convert DataFrame to VectorBT-compatible format"""
    return df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    ).vbt
