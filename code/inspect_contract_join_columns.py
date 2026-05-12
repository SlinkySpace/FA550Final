from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
week = sorted((ROOT / 'data' / 'app' / 'weeks').iterdir())[0]

for fname in ['contract_evolution_1s.parquet', 'contract_plot_sample.parquet']:
    path = week / fname
    df = pd.read_parquet(path)
    print('\\nFILE:', path)
    print('rows:', len(df))
    print('columns:')
    for c in df.columns:
        print(' ', c)
    print('\\nfirst row:')
    print(df.head(1).T.to_string())
