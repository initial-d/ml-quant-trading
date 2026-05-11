import yfinance as yf
import pandas as pd
import os

def download_example_data():
    ticker = "AAPL"
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start="2023-01-01", end="2023-12-31")
    
    if df.empty:
        print("Failed to download data.")
        return
    
    # Flatten columns if MultiIndex (yf 0.2.x+ behavior for single ticker can sometimes be different)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.to_csv("AAPL_sample.csv")
    print("Data saved to AAPL_sample.csv")
    print(df.head())

    # Now read it back to show it works
    print("\nReading data back from CSV...")
    df_read = pd.read_csv("AAPL_sample.csv", index_col=0, parse_dates=True)
    print(df_read.head())
    
    # Try to import Panel to see if we can map it
    try:
        import torch
        from mlquant.data.panel import Panel
        import numpy as np
        
        print("\nMapping to mlquant.Panel...")
        dates = df_read.index.to_numpy()
        stocks = [ticker]
        
        # Panel expects [T, N] tensors
        # df_read columns: Open, High, Low, Close, Adj Close, Volume
        fields = {
            "open": torch.from_numpy(df_read["Open"].to_numpy(dtype=np.float32)).view(-1, 1),
            "high": torch.from_numpy(df_read["High"].to_numpy(dtype=np.float32)).view(-1, 1),
            "low": torch.from_numpy(df_read["Low"].to_numpy(dtype=np.float32)).view(-1, 1),
            "close": torch.from_numpy(df_read["Close"].to_numpy(dtype=np.float32)).view(-1, 1),
            "volume": torch.from_numpy(df_read["Volume"].to_numpy(dtype=np.float32)).view(-1, 1),
            "vwap": torch.from_numpy(((df_read["Open"] + df_read["Close"]) / 2).to_numpy(dtype=np.float32)).view(-1, 1) # Proxy
        }
        
        mask = torch.ones((len(dates), 1), dtype=torch.bool)
        
        panel = Panel.from_tensors(dates, stocks, fields, mask)
        print("Successfully created mlquant.Panel!")
        print(f"Panel info: {panel.n_dates} dates, {panel.n_stocks} stock")
        
    except ImportError as e:
        print(f"\nSkipping Panel creation: {e}")
    except Exception as e:
        print(f"\nError creating Panel: {e}")

if __name__ == "__main__":
    download_example_data()
