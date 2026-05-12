from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
WEEKS_DIR = ROOT / 'data' / 'app' / 'weeks'

for week_dir in sorted(WEEKS_DIR.iterdir()):
    if not week_dir.is_dir():
        continue
    print(f'Updating aliases in {week_dir.name}')
    for fname in ['contract_evolution_1s.parquet', 'contract_plot_sample.parquet']:
        path = week_dir / fname
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if 'futures_last_trade_price_1s' in df.columns:
            df['underlying_trade_price'] = df['futures_last_trade_price_1s']
        if 'futures_trade_count_1s' in df.columns:
            df['underlying_trade_count'] = df['futures_trade_count_1s']
        if 'futures_trade_size_1s' in df.columns:
            df['underlying_trade_size'] = df['futures_trade_size_1s']
        df.to_parquet(path, index=False)
        nonnull = int(df['underlying_trade_price'].notna().sum()) if 'underlying_trade_price' in df.columns else 0
        print(f'  {fname}: non-null underlying_trade_price = {nonnull:,}')

print('Done.')
