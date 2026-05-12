from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
WEEKS_DIR = ROOT / 'data' / 'app' / 'weeks'

parts = []

for week_dir in sorted(WEEKS_DIR.iterdir()):
    path = week_dir / 'futures_15m_windows.parquet'
    if not path.exists():
        continue

    df = pd.read_parquet(path)
    if 'window_start' not in df.columns or 'trade_count_sum' not in df.columns:
        print(f'SKIP {week_dir.name}: missing window_start or trade_count_sum')
        continue

    temp = df[['window_start', 'trade_count_sum']].copy()
    temp['window_start'] = pd.to_datetime(temp['window_start'], errors='coerce')
    temp = temp.dropna(subset=['window_start'])
    temp['date'] = temp['window_start'].dt.date
    temp['week_id'] = week_dir.name
    parts.append(temp)

if not parts:
    raise SystemExit('No futures_15m_windows.parquet files found with trade_count_sum.')

all_windows = pd.concat(parts, ignore_index=True)

# Sum all 15-minute trade counts into daily totals
daily = all_windows.groupby('date', as_index=False)['trade_count_sum'].sum()
daily = daily.sort_values('date')

avg_daily = daily['trade_count_sum'].mean()
median_daily = daily['trade_count_sum'].median()
total_trades = daily['trade_count_sum'].sum()
num_days = len(daily)

print('\\n==============================')
print('BTC futures average trades per day')
print('Source: futures_15m_windows.parquet')
print('==============================')
print(f'Days counted: {num_days}')
print(f'Total trades: {total_trades:,.0f}')
print(f'Average trades/day: {avg_daily:,.2f}')
print(f'Median trades/day: {median_daily:,.2f}')
print(f'Min trades/day: {daily.trade_count_sum.min():,.0f}')
print(f'Max trades/day: {daily.trade_count_sum.max():,.0f}')

print('\\nDaily totals:')
print(daily.to_string(index=False))

out = ROOT / 'daily_trade_counts_from_futures_15m.csv'
daily.to_csv(out, index=False)
print(f'\\nSaved: {out}')
