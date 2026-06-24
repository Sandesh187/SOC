

import yfinance as yf
import pandas as pd
import numpy as np



tickers = {
    'copper': 'HG=F',
    'gold': 'GC=F', 
    'aluminum': 'ALI=F'
}

def get_prices(metal='copper', days='6mo'):
    """fetch historical prices from yahoo"""
    ticker = tickers.get(metal)
    if not ticker:
        print(f"don't know ticker for {metal}")
        return None
    
    print(f"\n--- fetching {metal} ({ticker}) ---")
    try:
        data = yf.download(ticker, period=days, progress=False)
        # yfinance sometimes gives multi-index columns which is annoying
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
        
        print(f"got {len(data)} rows, from {data.index[0]} to {data.index[-1]}")
        print(f"latest close: ${data['Close'].iloc[-1]:.2f}")
        return data
    except Exception as e:
        print(f"failed: {e}")
        return None

def main():
    
    copper = get_prices('copper', '6mo')
    if copper is not None:
        print("\nfirst 3 rows:")
        print(copper.head(3))
        print("\nlast 3 rows:")
        print(copper.tail(3))
        
       
        copper.to_csv('copper_week1.csv')
        print("\nsaved to copper_week1.csv")
    
    
    gold = get_prices('gold', '1mo')
    if gold is not None:
        print(f"\ngold latest: ${gold['Close'].iloc[-1]:.2f}")

if __name__ == "__main__":
    main()