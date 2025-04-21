# strategies/bollinger_bands.py
import numpy as np
from backtesting import Strategy

class BollingerBandsStrategy(Strategy):
    def init(self):
        # Strategy parameters
        self.bb_length = 20
        self.bb_std = 2.0
        self.rsi_length = 14
        
        # Initialize with np.nan instead of NaN
        self.upper = self.I(lambda: np.full(len(self.data.Close), np.nan))
        self.lower = self.I(lambda: np.full(len(self.data.Close), np.nan))
        self.rsi = self.I(lambda: np.full(len(self.data.Close), np.nan))
    
    def next(self):
        if len(self.data.Close) < max(self.bb_length, self.rsi_length):
            return
        
        # Manual Bollinger Bands calculation
        close_window = self.data.Close[-self.bb_length:]
        sma = close_window.mean()
        std = close_window.std()
        
        self.upper[-1] = sma + self.bb_std * std
        self.lower[-1] = sma - self.bb_std * std
        
        # Manual RSI calculation
        delta = self.data.Close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(self.rsi_length).mean()
        avg_loss = loss.rolling(self.rsi_length).mean()
        rs = avg_gain[-1] / avg_loss[-1]
        self.rsi[-1] = 100 - (100 / (1 + rs))
        
        # Trading logic
        price = self.data.Close[-1]
        if not self.position and price < self.lower[-1] and self.rsi[-1] < 30:
            self.buy()
        elif self.position and (price > self.upper[-1] or self.rsi[-1] > 70):
            self.position.close()