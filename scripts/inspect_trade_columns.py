from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
WEEKS_DIR = ROOT / 'data' / 'app' / 'weeks'

for week_dir in sorted(WEEKS_DIR.iterdir()):
    if not week_dir.is_dir():
        continue
    print(f'\\nWEEK: {week_dir.name}')
    for path in sorted(week_dir.glob('*.parquet')):
        try:
            df = pd.read_parquet(path)
            cols = list(df.columns)
            trade_cols = [c for c in cols if 'trade' in c.lower()]
            time_cols = [c for c in cols if any(x in c.lower() for x in ['time', 'date', 'ts', 'window'])]
            print(f'  {path.name}: rows={len(df):,}')
            print(f'    trade-like cols: {trade_cols}')
            print(f'    time-like cols: {time_cols[:10]}')
        except Exception as e:
            print(f'  {path.name}: ERROR {e}')
    break
