import pandas as pd

class TripleMAStrategy:
    def __init__(
        self,
        data: pd.DataFrame,
        ticker,
        short_window=10,
        medium_window=50,
        long_window=200,
    ):
        self.ticker = ticker
        if isinstance(data.columns, pd.MultiIndex):
            self.data = data[("Close", self.ticker)].dropna().to_frame(name="Close")
        else:
            self.data = data.copy()
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window

    def compute_ma(self):
        self.data["SMA10"] = self.data["Close"].rolling(window=self.short_window).mean()
        self.data["SMA50"] = (
            self.data["Close"].rolling(window=self.medium_window).mean()
        )
        self.data["SMA200"] = self.data["Close"].rolling(window=self.long_window).mean()
        self.data = self.data.dropna()

    def generate_signal(self):
        self.compute_ma()
        self.data["Signal"] = 0

        # Buy when short MA > medium MA > long MA
        self.data.loc[
            (self.data["SMA10"] > self.data["SMA50"])
            & (self.data["SMA50"] > self.data["SMA200"]),
            "Signal",
        ] = 1

        # Sell when short MA < medium MA < long MA
        self.data.loc[
            (self.data["SMA10"] < self.data["SMA50"])
            & (self.data["SMA50"] < self.data["SMA200"]),
            "Signal",
        ] = -1

        self.calculate_returns()

    def calculate_returns(self):
        self.data["Returns"] = self.data["Close"].pct_change()
        self.data["StrategyReturns"] = self.data["Returns"] * self.data["Signal"].shift(
            1
        )
        self.data.fillna(0, inplace=True)

    def execute(self):
        self.generate_signal()
        return self.data[["Close", "SMA10", "SMA50", "SMA200", "Signal"]]
