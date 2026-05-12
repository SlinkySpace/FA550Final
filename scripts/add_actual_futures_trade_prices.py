from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
WEEKS_DIR = ROOT / 'data' / 'app' / 'weeks'

TRADES_PATH = Path(r'C:\Users\jorda\OneDrive\Desktop\FA550_BTC_Capstone\data\raw\processed\trades_continuous.parquet')

print('Loading raw futures trades...')
trades = pd.read_parquet(TRADES_PATH, columns=['ts_event', 'price', 'size', 'symbol'])

trades['ts_event'] = pd.to_datetime(trades['ts_event'], utc=True, errors='coerce')
trades = trades.dropna(subset=['ts_event', 'price', 'symbol'])
trades['second'] = trades['ts_event'].dt.floor('s')

print(f'Raw trade rows loaded: {len(trades):,}')

# Build one row per futures symbol per second
trades_1s = (
    trades.sort_values('ts_event')
    .groupby(['symbol', 'second'], as_index=False)
    .agg(
        futures_last_trade_price_1s=('price', 'last'),
        futures_trade_count_1s=('price', 'size'),
        futures_trade_size_1s=('size', 'sum'),
    )
)

trades_1s['has_futures_trade_1s'] = trades_1s['futures_trade_count_1s'] > 0

print(f'1-second trade rows built: {len(trades_1s):,}')
print('Trade symbols:', sorted(trades_1s['symbol'].dropna().unique().tolist()))

files_to_update = [
    'contract_evolution_1s.parquet',
    'contract_plot_sample.parquet',
]

for week_dir in sorted(WEEKS_DIR.iterdir()):
    if not week_dir.is_dir():
        continue

    print(f'\\nUpdating week: {week_dir.name}')

    for fname in files_to_update:
        path = week_dir / fname
        if not path.exists():
            print(f'  SKIP missing {fname}')
            continue

        df = pd.read_parquet(path)

        if 'second' not in df.columns:
            print(f'  SKIP {fname}: no second column')
            continue

        symbol_col = None
        for candidate in ['selected_futures_symbol', 'futures_symbol', 'symbol']:
            if candidate in df.columns:
                symbol_col = candidate
                break

        if symbol_col is None:
            print(f'  SKIP {fname}: no futures symbol column')
            continue

        # Drop existing added columns if rerunning script
        added_cols = [
            'futures_last_trade_price_1s',
            'futures_trade_count_1s',
            'futures_trade_size_1s',
            'has_futures_trade_1s',
        ]
        df = df.drop(columns=[c for c in added_cols if c in df.columns])

        df['second'] = pd.to_datetime(df['second'], utc=True, errors='coerce')

        before_rows = len(df)

        merged = df.merge(
            trades_1s,
            how='left',
            left_on=[symbol_col, 'second'],
            right_on=['symbol', 'second'],
        )

        if 'symbol' in merged.columns and symbol_col != 'symbol':
            merged = merged.drop(columns=['symbol'])

        merged['futures_trade_count_1s'] = merged['futures_trade_count_1s'].fillna(0).astype('int64')
        merged['futures_trade_size_1s'] = merged['futures_trade_size_1s'].fillna(0.0)
        merged['has_futures_trade_1s'] = merged['futures_last_trade_price_1s'].notna()

        matched = int(merged['has_futures_trade_1s'].sum())
        pct = 100 * matched / before_rows if before_rows else 0

        merged.to_parquet(path, index=False)

        print(f'  Updated {fname}: rows={before_rows:,}, actual-trade seconds={matched:,} ({pct:.2f}%)')

print('\\nDone. Weekly files now include actual futures trade price columns.')
