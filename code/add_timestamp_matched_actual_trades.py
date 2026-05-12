from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
WEEKS_DIR = ROOT / 'data' / 'app' / 'weeks'
TRADES_PATH = Path(r'C:\Users\jorda\OneDrive\Desktop\FA550_BTC_Capstone\data\raw\processed\trades_continuous.parquet')

print('Loading raw futures trades...')
trades = pd.read_parquet(TRADES_PATH, columns=['ts_event', 'price', 'size', 'symbol'])
trades['ts_event'] = pd.to_datetime(trades['ts_event'], utc=True, errors='coerce')
trades = trades.dropna(subset=['ts_event', 'price'])
trades['second'] = trades['ts_event'].dt.floor('s')

print(f'Raw trade rows loaded: {len(trades):,}')

# Build one actual futures trade row per second, across available BTC futures symbols.
# This avoids the symbol-code mismatch that caused the strict symbol join to return 0 matches.
trades_1s = (
    trades.sort_values('ts_event')
    .groupby('second', as_index=False)
    .agg(
        underlying_trade_price=('price', 'last'),
        underlying_trade_count=('price', 'size'),
        underlying_trade_size=('size', 'sum'),
        underlying_trade_symbol=('symbol', 'last'),
    )
)

trades_1s['has_underlying_trade_price'] = trades_1s['underlying_trade_price'].notna()

print(f'1-second actual trade rows built: {len(trades_1s):,}')
print(f'Trade time range: {trades_1s.second.min()} to {trades_1s.second.max()}')

files_to_update = ['contract_evolution_1s.parquet', 'contract_plot_sample.parquet']

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

        # Remove old failed trade-price columns if they exist
        drop_cols = [
            'underlying_trade_price',
            'underlying_trade_count',
            'underlying_trade_size',
            'underlying_trade_symbol',
            'has_underlying_trade_price',
            'futures_last_trade_price_1s',
            'futures_trade_count_1s',
            'futures_trade_size_1s',
            'has_futures_trade_1s',
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        df['second'] = pd.to_datetime(df['second'], utc=True, errors='coerce')
        before_rows = len(df)

        merged = df.merge(trades_1s, how='left', on='second')

        matched = int(merged['underlying_trade_price'].notna().sum())
        pct = 100 * matched / before_rows if before_rows else 0

        merged.to_parquet(path, index=False)

        print(f'  Updated {fname}: rows={before_rows:,}, actual-trade rows={matched:,} ({pct:.2f}%)')

print('\\nDone. Weekly files now include timestamp-matched actual futures trade prices.')
