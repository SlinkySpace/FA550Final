from pathlib import Path
import pandas as pd

paths = [
    Path(r'C:\Users\jorda\OneDrive\Desktop\FA550_BTC_Capstone\data\raw\processed\trades_continuous.parquet'),
    Path(r'C:\Users\jorda\OneDrive\Desktop\FA550_BTC_Capstone\data\raw\live\trades_continuous.parquet'),
    Path(r'C:\Users\jorda\OneDrive\Desktop\FA550_BTC_Capstone\data\raw\live\processed\trades_continuous.parquet'),
    Path(r'C:\Users\jorda\OneDrive\Desktop\ClaudeContracts\btc_kalshi\data\live\processed\trades_continuous.parquet'),
]

for p in paths:
    print('\\nFILE:', p)
    if not p.exists():
        print('  MISSING')
        continue
    df = pd.read_parquet(p, columns=['ts_event', 'price', 'size', 'symbol'])
    df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True, errors='coerce')
    print('  rows:', len(df))
    print('  min:', df['ts_event'].min())
    print('  max:', df['ts_event'].max())
    print('  symbols:', sorted(df['symbol'].dropna().astype(str).unique().tolist())[:20])
    print('  symbol count:', df['symbol'].nunique())
