import pandas as pd
import vectorbt as vbt
import pandas as pd
from datetime import timedelta

def generate_date_range(start_date, end_date):
    """ Generate a date range for fetching data """
    return pd.date_range(start=start_date, end=end_date).strftime('%Y-%m-%d').tolist()

def calculate_percentage_change(df, column='Close'):
    """Calculate the percentage change for a given column."""
    df['Pct_Change'] = df[column].pct_change() * 100
    return df


# Helper to format timedelta into years/months/days
def format_duration(td: timedelta):
    total_days = td.days
    years = total_days // 365
    months = (total_days % 365) // 30
    days = (total_days % 365) % 30

    parts = []
    if years > 0:
        parts.append(f"{years} year{'s' if years > 1 else ''}")
    if months > 0:
        parts.append(f"{months} month{'s' if months > 1 else ''}")
    if days > 0 or not parts:
        parts.append(f"{days} day{'s' if days != 1 else ''}")

    return ", ".join(parts)


def get_portfolio_stats(pf: vbt.portfolio.base.Portfolio):
    # Define custom metrics with formatting
    custom_metrics = {
        "Annual Return [%]": lambda pf: f"{pf.annual_return() * 100:.2f}%",
        "Volatility [%]": lambda pf: f"{pf.volatility() * 100:.2f}%",
        "Sharpe Ratio": lambda pf: f"{pf.sharpe_ratio():.2f}",
        "Calmar Ratio": lambda pf: f"{pf.calmar_ratio():.2f}",
        "Max Drawdown [%]": lambda pf: f"{pf.max_drawdown() * 100:.2f}%",
        "Max Drawdown Duration": lambda pf: str(pf.max_drawdown_duration()),
        "Value at Risk [%]": lambda pf: f"{pf.value_at_risk() * 100:.2f}%",
        "CVaR [%]": lambda pf: f"{pf.conditional_value_at_risk() * 100:.2f}%",
        "Profit Factor": lambda pf: f"{pf.profit_factor():.2f}",
        "Expectancy": lambda pf: f"{pf.expectancy():.2f}",
        "Omega Ratio": lambda pf: f"{pf.omega_ratio():.2f}",
        "Sortino Ratio": lambda pf: f"{pf.sortino_ratio():.2f}",
        "Avg Trade Return [%]": lambda pf: f"{pf.trades.avg_return() * 100:.2f}%",
        "Avg Winning Trade [%]": lambda pf: f"{pf.trades.avg_win() * 100:.2f}%",
        "Avg Losing Trade [%]": lambda pf: f"{pf.trades.avg_loss() * 100:.2f}%",
        "Best Trade [%]": lambda pf: f"{pf.best_trade_return() * 100:.2f}%",
        "Worst Trade [%]": lambda pf: f"{pf.worst_trade_return() * 100:.2f}%",
        "avg_winning_trade_duration": lambda pf: format_duration(
            pf.trades.avg_win_duration()
        ),
        "avg_losing_trade_duration": lambda pf: format_duration(
            pf.trades.avg_loss_duration()
        ),
        "Win Rate [%]": lambda pf: f"{pf.win_rate() * 100:.2f}%",
        "Total Trades": lambda pf: pf.total_trades(),
        "Total Closed Trades": lambda pf: pf.total_closed_trades(),
        "Total Open Trades": lambda pf: pf.total_open_trades(),
        "Open Trade PnL": lambda pf: f"{pf.open_trade_pnl():.2f}",
    }

    stats = pf.stats(settings=dict(metrics=custom_metrics))

    # Format only desired rows as %.2f
    for label in stats.index:
        if stats.loc[label] and isinstance(stats.loc[label], (int, float)):
            if "%" in label:
                stats.loc[label] = f"{stats.loc[label]:.2f}%"
            elif "Duration" in label:
                stats.loc[label] = f"{stats.loc[label]:.0f} days"
            else:
                stats.loc[label] = f"{stats.loc[label]:.2f}"

    return stats
