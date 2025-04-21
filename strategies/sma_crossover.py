import pandas as pd

class SMACrossoverStrategy:
    def __init__(self, data: pd.DataFrame, ticker,short_window=50, long_window=200):
        self.ticker = ticker
        # Ensure the data is a copy to avoid modifying a slice
        if isinstance(data.columns, pd.MultiIndex):
            # Get the ticker from the MultiIndex

            # Extract Close prices for the ticker
            self.data = data[("Close", self.ticker)].dropna().to_frame(name="Close")
        else:
            self.data = data.copy()
        self.short_window = short_window
        self.long_window = long_window

    def compute_sma(self):
        self.data['SMA50'] = self.data['Close'].rolling(window=self.short_window).mean()
        self.data['SMA200'] = self.data['Close'].rolling(window=self.long_window).mean()

        # Drop any remaining NaN values
        self.data = self.data.dropna(subset=["Close", "SMA50", "SMA200"])

    def generate_signal(self):
            self.compute_sma()
            # Initialize signals
            self.data['Signal'] = 0
            
            # Generate signals using .loc
            self.data.loc[self.data['SMA50'] > self.data['SMA200'], 'Signal'] = 1
            self.data.loc[self.data['SMA50'] <= self.data['SMA200'], 'Signal'] = -1
            
            # Calculate returns and strategy returns
            self.data['Returns'] = self.data['Close'].pct_change()
            self.data['StrategyReturns'] = self.data['Returns'] * self.data['Signal'].shift(1)
            
            # Replace first row NaN values with 0
            self.data['Returns'] = self.data['Returns'].fillna(0)
            self.data['StrategyReturns'] = self.data['StrategyReturns'].fillna(0)

    def execute(self):
        self.generate_signal()
        # Ensure all required columns are present
        required_columns = ['Close', 'SMA50', 'SMA200', 'Signal']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Expected: {required_columns}")
            
        return self.data[required_columns]