from pathlib import Path
import pandas as pd

roots = [
    Path(r'C:\Users\jorda\OneDrive\Desktop\FA550_BTC_Capstone'),
    Path(r'C:\Users\jorda\OneDrive\Desktop\ClaudeContracts\btc_kalshi'),
    Path(r'C:\Users\jorda\OneDrive\Desktop\FA550_BTC_Kalshi_Final'),
]

patterns = ['*.parquet']
keywords = ['trade', 'trades', 'tbbo', 'ohlcv', 'live', 'futures', 'continuous']

seen = set()

for root in roots:
    if not root.exists():
        continue
    print(f'\\nSEARCHING {root}')
    for path in root.rglob('*.parquet'):
        if path in seen:
            continue
        seen.add(path)

        low = str(path).lower()
        if not any(k in low for k in keywords):
            continue
        if 'site-packages' in low or '.venv' in low:
            continue

        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f'\\nSKIP {path}')
            print(f'  error: {e}')
            continue

        cols = list(df.columns)
        idx_name = df.index.name

        time_cols = [c for c in cols if any(k in c.lower() for k in ['ts_event','timestamp','time','date','window_start','first_timestamp','last_timestamp'])]
        price_cols = [c for c in cols if any(k in c.lower() for k in ['price','close','bid_px','ask_px','underlying'])]
        trade_cols = [c for c in cols if any(k in c.lower() for k in ['trade','size','volume'])]

        # Try to find a real datetime range
        dt_min = None
        dt_max = None
        dt_source = None

        for c in time_cols:
            s = pd.to_datetime(df[c], utc=True, errors='coerce')
            if s.notna().any():
                dt_min = s.min()
                dt_max = s.max()
                dt_source = c
                break

        if dt_min is None:
            try:
                s = pd.to_datetime(df.index, utc=True, errors='coerce')
                if pd.Series(s).notna().any():
                    dt_min = pd.Series(s).min()
                    dt_max = pd.Series(s).max()
                    dt_source = f'index ({idx_name})'
            except Exception:
                pass

        print(f'\\nFILE: {path}')
        print(f'  rows: {len(df):,}')
        print(f'  datetime source: {dt_source}')
        print(f'  min: {dt_min}')
        print(f'  max: {dt_max}')
        print(f'  price-like cols: {price_cols}')
        print(f'  trade-like cols: {trade_cols}')
        print(f'  time-like cols: {time_cols}')
        if 'symbol' in cols:
            try:
                print(f'  symbols: {sorted(df.symbol.dropna().astype(str).unique().tolist())[:20]}')
            except Exception:
                pass

print('\\nDone.')
