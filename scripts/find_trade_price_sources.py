from pathlib import Path
import pandas as pd

roots = [
    Path(r'C:\Users\jorda\OneDrive\Desktop\FA550_BTC_Capstone'),
    Path(r'C:\Users\jorda\OneDrive\Desktop\ClaudeContracts\btc_kalshi'),
    Path(r'C:\Users\jorda\OneDrive\Desktop\FA550_BTC_Kalshi_Final'),
]

patterns = [
    '*trade*.parquet',
    '*trades*.parquet',
    '*continuous*.parquet',
]

seen = set()

for root in roots:
    if not root.exists():
        continue
    print(f'\\nSEARCHING {root}')
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path in seen:
                continue
            seen.add(path)
            if 'site-packages' in str(path) or '.venv' in str(path):
                continue
            try:
                df = pd.read_parquet(path)
            except Exception as e:
                print(f'SKIP {path}: {e}')
                continue

            cols = list(df.columns)
            price_cols = [c for c in cols if any(k in c.lower() for k in ['price', 'px', 'last', 'close'])]
            trade_cols = [c for c in cols if any(k in c.lower() for k in ['trade', 'size', 'qty', 'volume'])]
            time_cols = [c for c in cols if any(k in c.lower() for k in ['time', 'ts', 'date'])]

            print('\\nFILE:', path)
            print('rows:', len(df))
            print('time-like:', time_cols)
            print('price-like:', price_cols)
            print('trade-like:', trade_cols)
            print('all cols:', cols[:40])

print('\\nDone.')
