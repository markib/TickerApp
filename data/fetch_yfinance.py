import yfinance as yf
import pandas as pd

def fetch_yfinance_data(ticker, start_date="2020-01-01", end_date="2025-01-01"):
    """Fetch historical stock data from Yahoo Finance with error handling."""
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            raise ValueError(f"No data found for ticker '{ticker}'.")
        stock_data['Date'] = stock_data.index
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure

def get_tickers():
   # Download the Nasdaq ticker list
    nasdaq_url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    nasdaq_df = pd.read_csv(nasdaq_url, sep="|")
    nasdaq_tickers = nasdaq_df['Symbol'].tolist()[:-1]  # Remove last row which is not a ticker

    # Download the NYSE ticker list
    nyse_url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    nyse_df = pd.read_csv(nyse_url, sep="|")
    nyse_tickers = nyse_df['ACT Symbol'].tolist()[:-1]

    # Combine both
    all_us_tickers = nasdaq_tickers + nyse_tickers

    # Save as JSON (optional)
    # import json
    # with open("us_tickers.json", "w") as f:
    #     json.dump(all_us_tickers, f)

    return all_us_tickers