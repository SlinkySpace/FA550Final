# Data README

This folder contains the dashboard-ready data files used by the FA550 BTC/Kalshi capstone project.

## data/app/weeks/

Each subfolder contains one week of precomputed dashboard data. The Streamlit app loads one selected week at a time for performance.

Weekly folders may include files such as:

- contract_summary.parquet
- contract_evolution_1s.parquet
- contract_plot_sample.parquet
- sensitivity_bins.parquet
- threshold_heatmap.parquet
- contract_decision_times.parquet
- summary metric CSV files

These files support the dashboard pages for contract evolution, futures microstructure, decision timing, threshold risk, sensitivity, and data preview.

## Notes

The project focuses on BTC futures microstructure and Kalshi BTC binary contract price behavior.

Trading strategy execution and trading-strategy evaluation are intentionally not part of this assignment.
