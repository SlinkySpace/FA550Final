# User Guide: BTC Futures and Kalshi Contract Behavior Dashboard

## Getting Started

Open the deployed Streamlit app, or run it locally from the project folder.

Local run command:

streamlit run .\app\app.py

The dashboard loads prepared weekly files, so the first load may take a few seconds.

## Main Workflow

1. Select an analysis week from the sidebar.
2. Select a Kalshi contract.
3. Move through the tabs to inspect contract behavior, futures microstructure, decision timing, threshold risk, sensitivity, and raw data previews.

## Sidebar Controls

### Analysis Week

Loads one prepared weekly dataset at a time. This keeps the dashboard responsive and prevents the app from loading the full high-frequency dataset at once.

### Contract

Selects one Kalshi BTC contract for the Contract Evolution page.

### Time-to-Close Bucket

Filters views by how much time remains before contract close.

### Moneyness Bucket

Filters views by distance from the contract threshold.

### Data Source Period

Filters rows by source period when that field is available.

### Adaptive Contract Price Axis

Zooms selected contract price charts to the observed range instead of always forcing a 0-100 cent axis.

### Show Trade-Second Markers

Adds markers for contract trade seconds when available. This can clutter the chart, so it is optional.

## Dashboard Tabs

### Overview

Shows selected-week summary metrics, contract outcome distribution, BTC futures quote midpoint path, and decision-time distribution.

### Contract Evolution

Shows one selected Kalshi contract over time, including YES/NO prices, BTC futures quote midpoint, decision marker, close marker, and short-horizon YES price movement.

### Futures Microstructure

Shows futures-side quote midpoint, spread, trade activity, and local volatility.

### Decision Time

Shows how many minutes before close contracts became effectively decided.

### Threshold Risk

Shows distance-from-threshold and moneyness-style views. For binary contracts, moneyness is best interpreted as distance from the decision boundary, not growing intrinsic value.

### Sensitivity

Shows the relationship between short-horizon BTC futures quote movement and short-horizon Kalshi YES price movement.

### Data Preview

Provides audit tables showing which weekly files and columns are being loaded.

## Underlying Reference

The dashboard uses BTC futures quote midpoint as the underlying reference:

quote_mid = (best_bid + best_ask) / 2

Actual futures trade prices were tested but removed from the final app because the available actual-trade file did not cover the full dashboard range. Quote midpoint is used because it is consistent across all included weekly files.

## Troubleshooting

If charts do not load:

- Refresh the browser.
- Use Chrome or Edge.
- Wait for the selected week to finish loading.
- Make sure the weekly data files exist under data/app/weeks/.
- If running locally, make sure dependencies are installed with:

pip install -r requirements.txt

If the app cannot find data:

- Confirm that data/app/week_index.csv exists.
- Confirm that data/app/weeks/ contains weekly folders.
- Confirm that each weekly folder contains the required Parquet files.
