from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
WEEKS_DIR = ROOT / 'data' / 'app' / 'weeks'

files_to_check = [
    'contract_evolution_1s.parquet',
    'contract_plot_sample.parquet',
    'futures_15m_windows.parquet',
    'futures_label_window_behavior.parquet',
    'kalshi_threshold_matches.parquet'
]

keywords = ['price', 'trade', 'underlying', 'mid', 'bid', 'ask', 'last', 'close']

for week_dir in sorted(WEEKS_DIR.iterdir()):
    if not week_dir.is_dir():
        continue
    print(f'CHECKING WEEK: {week_dir.name}')
    for fname in files_to_check:
        path = week_dir / fname
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        print(f'\\nFILE: {fname}')
        print(f'rows: {len(df):,}')
        matches = [c for c in df.columns if any(k in c.lower() for k in keywords)]
        for c in matches:
            print('  ' + c)
    break
