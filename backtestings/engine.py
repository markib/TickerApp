class BacktestEngine:
    def __init__(self, strategy):
        self.strategy = strategy

    def run_backtest(self):
        # Ensure signals is a copy to avoid modifying a slice
        signals = self.strategy.execute().copy()
        
        # Simulate buying/selling based on strategy
        signals.loc[:, 'Returns'] = signals['Close'].pct_change()
        signals.loc[:, 'StrategyReturns'] = signals['Returns'] * signals['Signal'].shift(1)
        total_returns = signals['StrategyReturns'].sum()
        
        # Return backtest results
        return signals,total_returns