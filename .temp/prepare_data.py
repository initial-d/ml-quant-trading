import yfinance as yf
import pandas as pd
import numpy as np
import os

def prepare_mlquant_data():
    ticker = "AAPL"
    print(f"Downloading data for {ticker}...")
    # Using a single ticker
    df = yf.download(ticker, start="2023-01-01", end="2023-12-31")
    
    if df.empty:
        print("Failed to download data.")
        return
    
    # yfinance columns can be tricky. Let's ensure we get the values correctly.
    # We'll use .loc to be explicit.
    def get_col(name):
        if name in df.columns:
            return df[name].values
        # Try MultiIndex if applicable
        for col in df.columns:
            if isinstance(col, tuple) and name in col:
                return df[col].values
        return None

    ml_df = pd.DataFrame()
    ml_df["TRADE_DT"] = df.index.strftime("%Y%m%d")
    ml_df["S_INFO_WINDCODE"] = ticker
    
    opens = get_col("Open")
    closes = get_col("Close")
    highs = get_col("High")
    lows = get_col("Low")
    volumes = get_col("Volume")

    ml_df["open"] = opens
    ml_df["close"] = closes
    ml_df["high"] = highs
    ml_df["low"] = lows
    ml_df["volume"] = volumes
    
    # Use simple average for vwap proxy
    ml_df["vwap"] = (opens + closes + highs + lows) / 4.0
    
    # Save as tab-separated file as expected by loaders.py
    output_path = "yfinance_aapl.csv"
    ml_df.to_csv(output_path, sep="\t", index=False)
    print(f"Formatted data saved to {output_path}")
    
    # Show the formatted data
    print("\nFormatted Data Head:")
    print(ml_df.head())

if __name__ == "__main__":
    prepare_mlquant_data()
