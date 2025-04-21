import pandas as pd


class RSIStrategy:
    def __init__(self, data, ticker, window=14):
        """Initialize RSI Strategy.

        Args:
            data (pd.DataFrame): Input price data
            ticker (str): Stock ticker symbol
            window (int, optional): RSI calculation window. Defaults to 14.
        """
        self.data = data
        self.window = window
        self.ticker = ticker

    def execute(self):
        df = self.data.copy()

        # Handle multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            # Extract Close price for the specific ticker
            df["Close"] = df[("Close", self.ticker)]
        else:
            if "Close" not in df.columns:
                raise ValueError("DataFrame must contain a 'Close' column")

        # Calculate RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.window, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # Generate signals
        df["Signal"] = 0
        df.loc[df["RSI"] < 30, "Signal"] = 1  # Oversold - Buy
        df.loc[df["RSI"] > 70, "Signal"] = -1  # Overbought - Sell

        return df[["Close", "RSI", "Signal"]]
