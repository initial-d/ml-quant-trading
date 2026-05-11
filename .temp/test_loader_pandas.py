import pandas as pd
import numpy as np

def test_loader_logic():
    path = "yfinance_aapl.csv"
    sep = "\t"
    
    print(f"Simulating mlquant.data.loaders.load_ochlv_csv for {path}...")
    
    # 1. Read CSV
    df = pd.read_csv(path, sep=sep, low_memory=False)
    print("Columns found:", list(df.columns))
    
    # 2. Date conversion
    df["TRADE_DT"] = pd.to_datetime(df["TRADE_DT"].astype(str))
    
    # 3. Ticker handling (project expects 6-char codes usually, but we'll adapt)
    df["S_INFO_WINDCODE"] = df["S_INFO_WINDCODE"].astype(str)
    
    # 4. Pivot (as done in loaders.py)
    # The loader pivots each field into [Date x Stock]
    fields = ["open", "close", "high", "low", "volume", "vwap"]
    
    print("\nPivoting fields into panels:")
    for fld in fields:
        wide = df.pivot(index="TRADE_DT", columns="S_INFO_WINDCODE", values=fld)
        print(f"Field '{fld}' shape: {wide.shape}")
    
    print("\nSuccess: yfinance data is structurally compatible with the mlquant loading pipeline!")

if __name__ == "__main__":
    test_loader_logic()
