from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

root = Path.cwd()
weeks_dir = root / 'data' / 'app' / 'weeks'
out_csv = root / 'data' / 'app' / 'week_index.csv'

required = {
    'contract_summary.parquet',
    'contract_evolution_1s.parquet',
    'contract_plot_sample.parquet',
    'sensitivity_bins.parquet',
    'threshold_heatmap.parquet',
    'contract_decision_times.parquet',
}

rows = []

for week_dir in sorted(weeks_dir.iterdir()):
    if not week_dir.is_dir():
        continue

    files = {p.name for p in week_dir.iterdir() if p.is_file()}
    if not required.issubset(files):
        continue

    week_id = week_dir.name
    try:
        start_dt = datetime.strptime(week_id, '%Y-%m-%d')
        end_dt = start_dt + timedelta(days=6)
        week_label = start_dt.strftime('%b %d, %Y') + ' - ' + end_dt.strftime('%b %d, %Y')
    except ValueError:
        week_label = week_id

    rows.append({
        'week_id': week_id,
        'week_label': week_label,
        'week_start': week_id,
        'week_path': f'data/app/weeks/{week_id}',
    })

df = pd.DataFrame(rows)
df.to_csv(out_csv, index=False)

print(f'Created {out_csv}')
print(df.head())
print(f'Total complete weeks: {len(df)}')
