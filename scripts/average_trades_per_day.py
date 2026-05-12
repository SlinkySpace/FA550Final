from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
WEEKS_DIR = ROOT / 'data' / 'app' / 'weeks'

candidate_files = [
    'futures_label_window_behavior.parquet',
    'futures_15m_windows.parquet',
    'contract_evolution_1s.parquet',
    'contract_plot_sample.parquet'
]

daily_parts = []

for week_dir in sorted(WEEKS_DIR.iterdir()):
    if not week_dir.is_dir():
        continue

    for fname in candidate_files:
        path = week_dir / fname
        if not path.exists():
            continue

        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f'SKIP read error: {path} -> {e}')
            continue

        # Find timestamp column
        time_col = None
        for c in ['timestamp', 'ts_event', 'window_start', 'time', 'datetime']:
            if c in df.columns:
                time_col = c
                break

        if time_col is None or 'trade_count' not in df.columns:
            continue

        temp = df[[time_col, 'trade_count']].copy()
        temp[time_col] = pd.to_datetime(temp[time_col], errors='coerce')
        temp = temp.dropna(subset=[time_col])
        temp['date'] = temp[time_col].dt.date

        daily = temp.groupby('date', as_index=False)['trade_count'].sum()
        daily['source_file'] = fname
        daily['week_id'] = week_dir.name
        daily_parts.append(daily)

        print(f'Using {fname} from week {week_dir.name} with {len(df):,} rows')
        break

if not daily_parts:
    print('No weekly files with both a timestamp column and trade_count were found.')
    print('Run the inspection script below to list columns in the weekly files.')
else:
    all_daily = pd.concat(daily_parts, ignore_index=True)

    # If dates appear in multiple weekly files, combine them
    daily_total = all_daily.groupby('date', as_index=False)['trade_count'].sum()
    daily_total = daily_total.sort_values('date')

    avg_trades_per_day = daily_total['trade_count'].mean()
    median_trades_per_day = daily_total['trade_count'].median()
    total_trades = daily_total['trade_count'].sum()
    num_days = len(daily_total)

    print('\\n==============================')
    print('Average trades per day')
    print('==============================')
    print(f'Days counted: {num_days}')
    print(f'Total trades: {total_trades:,.0f}')
    print(f'Average trades/day: {avg_trades_per_day:,.2f}')
    print(f'Median trades/day: {median_trades_per_day:,.2f}')

    print('\\nDaily trade counts:')
    print(daily_total.to_string(index=False))

    out = ROOT / 'daily_trade_counts.csv'
    daily_total.to_csv(out, index=False)
    print(f'\\nSaved daily results to: {out}')
