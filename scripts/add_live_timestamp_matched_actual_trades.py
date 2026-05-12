from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
WEEKS_DIR = ROOT / 'data' / 'app' / 'weeks'

# Use live trades, not the old 2022-2024 processed file.
TRADES_PATH = Path(r'C:\Users\jorda\OneDrive\Desktop\FA550_BTC_Capstone\data\raw\live\processed\trades_continuous.parquet')

print('Loading raw LIVE futures trades...')
print('Trades path:', TRADES_PATH)

trades = pd.read_parquet(TRADES_PATH, columns=['ts_event', 'price', 'size', 'symbol'])
trades['ts_event'] = pd.to_datetime(trades['ts_event'], utc=True, errors='coerce')
trades = trades.dropna(subset=['ts_event', 'price'])
trades['second'] = trades['ts_event'].dt.floor('s')

print(f'Raw live trade rows loaded: {len(trades):,}')
print(f'Trade time range: {trades.ts_event.min()} to {trades.ts_event.max()}')
print('Trade symbols:', sorted(trades['symbol'].dropna().astype(str).unique().tolist()))

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

print(f'1-second actual trade rows built: {len(trades_1s):,}')

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

        drop_cols = [
            'underlying_trade_price',
            'underlying_trade_count',
            'underlying_trade_size',
            'underlying_trade_symbol',
            'has_underlying_trade_price',
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        df['second'] = pd.to_datetime(df['second'], utc=True, errors='coerce')
        before_rows = len(df)

        merged = df.merge(trades_1s, how='left', on='second')
        merged['has_underlying_trade_price'] = merged['underlying_trade_price'].notna()

        matched = int(merged['has_underlying_trade_price'].sum())
        pct = 100 * matched / before_rows if before_rows else 0

        merged.to_parquet(path, index=False)
        print(f'  Updated {fname}: rows={before_rows:,}, actual-trade rows={matched:,} ({pct:.2f}%)')

print('\\nDone. Weekly files now include live timestamp-matched actual futures trade prices.')
