from pathlib import Path
import polars as pl

# Run this script from the FA550_BTC_Capstone folder.
# Example:
# PS C:\Users\jorda\OneDrive\Desktop\FA550_BTC_Capstone> python .\inspect_data.py

BASE = Path.cwd()

FILES = {
    "microstructure_1s_mvp": BASE / "data/raw/live/processed/microstructure_1s_mvp.parquet",
    "live_tbbo_processed": BASE / "data/raw/live/processed/tbbo_continuous.parquet",
    "live_trades_processed": BASE / "data/raw/live/processed/trades_continuous.parquet",
    "live_ohlcv_1s_processed": BASE / "data/raw/live/processed/ohlcv-1s_continuous.parquet",
    "live_ohlcv_1m_processed": BASE / "data/raw/live/processed/ohlcv-1m_continuous.parquet",

    # These duplicate files may also exist directly under raw/live.
    "live_tbbo_direct": BASE / "data/raw/live/tbbo_continuous.parquet",
    "live_trades_direct": BASE / "data/raw/live/trades_continuous.parquet",
    "live_ohlcv_1s_direct": BASE / "data/raw/live/ohlcv-1s_continuous.parquet",
    "live_ohlcv_1m_direct": BASE / "data/raw/live/ohlcv-1m_continuous.parquet",

    "kalshi_markets": BASE / "data/raw/kalshi/kxbtc15m_markets.parquet",
    "kalshi_analysis": BASE / "data/raw/kalshi/kxbtc15m_analysis.parquet",
    "kalshi_markets_extra": BASE / "data/raw/kalshi/markets.parquet",
    "kalshi_binary_markets": BASE / "data/raw/kalshi/binary_markets.parquet",

    "polymarket_markets": BASE / "data/raw/polymarket/markets.parquet",
    "polymarket_price_history": BASE / "data/raw/polymarket/price_history.parquet",
    "polymarket_by_minute": BASE / "data/raw/polymarket/by_minute.parquet",
    "polymarket_drift": BASE / "data/raw/polymarket/drift.parquet",

    # General processed files. These may be trading/model files, but we inspect them
    # so we know what is useful for price-behavior visuals.
    "processed_features": BASE / "data/raw/processed/features.parquet",
    "processed_features_clean": BASE / "data/raw/processed/features_clean.parquet",
    "processed_features_enriched": BASE / "data/raw/processed/features_enriched.parquet",
    "processed_labels": BASE / "data/raw/processed/labels.parquet",
    "processed_tbbo": BASE / "data/raw/processed/tbbo_continuous.parquet",
    "processed_trades": BASE / "data/raw/processed/trades_continuous.parquet",
}


def safe_collect(lf, description):
    try:
        return lf.collect()
    except Exception as e:
        print(f"Could not collect {description}: {e}")
        return None


def inspect_parquet(name, path):
    print("\n" + "=" * 90)
    print(name)
    print(path)

    if not path.exists():
        print("MISSING")
        return

    try:
        lf = pl.scan_parquet(path)
        schema = lf.collect_schema()
    except Exception as e:
        print(f"Could not scan parquet file: {e}")
        return

    print("\nCOLUMNS:")
    for col, dtype in schema.items():
        print(f"  {col}: {dtype}")

    print("\nSHAPE:")
    row_count = safe_collect(
        lf.select(pl.len().alias("rows")),
        "row count",
    )
    if row_count is not None:
        print(row_count)

    print("\nHEAD:")
    head = safe_collect(lf.head(5), "head")
    if head is not None:
        print(head)

    possible_time_cols = [
        "timestamp",
        "ts_event",
        "ts_recv",
        "time",
        "datetime",
        "date",
        "created_time",
        "close_time",
        "expiration_time",
        "expiration_ts",
        "open_time",
        "event_time",
        "market_open_time",
        "market_close_time",
        "settlement_time",
    ]

    time_cols = [c for c in possible_time_cols if c in schema.names()]

    if time_cols:
        print("\nTIME COLUMN RANGES:")
        for col_name in time_cols:
            try:
                out = lf.select(
                    pl.col(col_name).min().alias(f"{col_name}_min"),
                    pl.col(col_name).max().alias(f"{col_name}_max"),
                ).collect()
                print(out)
            except Exception as e:
                print(f"Could not summarize {col_name}: {e}")

    possible_symbol_cols = [
        "symbol",
        "ticker",
        "market_ticker",
        "event_ticker",
        "contract_ticker",
        "series_ticker",
        "instrument_id",
    ]

    symbol_cols = [c for c in possible_symbol_cols if c in schema.names()]

    if symbol_cols:
        print("\nSYMBOL / TICKER SAMPLE VALUES:")
        for col_name in symbol_cols:
            try:
                sample = (
                    lf.select(pl.col(col_name))
                    .drop_nulls()
                    .unique()
                    .head(10)
                    .collect()
                )
                print(f"\n{col_name}:")
                print(sample)
            except Exception as e:
                print(f"Could not sample {col_name}: {e}")

    possible_numeric_cols = [
        "best_bid",
        "best_ask",
        "bid",
        "ask",
        "price",
        "yes_bid",
        "yes_ask",
        "no_bid",
        "no_ask",
        "yes_price",
        "no_price",
        "mid_price",
        "spread",
        "spread_bps",
        "trade_count",
        "trade_volume",
        "volume",
        "rv_60s",
        "fwd_return_5s",
        "abs_fwd_return_5s",
    ]

    numeric_cols = [c for c in possible_numeric_cols if c in schema.names()]

    if numeric_cols:
        print("\nNUMERIC SUMMARY FOR KEY COLUMNS:")
        exprs = []
        for col_name in numeric_cols:
            exprs.extend(
                [
                    pl.col(col_name).min().alias(f"{col_name}_min"),
                    pl.col(col_name).max().alias(f"{col_name}_max"),
                    pl.col(col_name).mean().alias(f"{col_name}_mean"),
                ]
            )

        try:
            summary = lf.select(exprs).collect()
            print(summary)
        except Exception as e:
            print(f"Could not create numeric summary: {e}")


def main():
    print("BASE DIRECTORY:")
    print(BASE)

    print("\nChecking expected files...")
    for name, path in FILES.items():
        status = "FOUND" if path.exists() else "MISSING"
        print(f"{status:8} {name:30} {path}")

    for name, path in FILES.items():
        inspect_parquet(name, path)


if __name__ == "__main__":
    main()