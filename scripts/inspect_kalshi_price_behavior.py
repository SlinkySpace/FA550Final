from pathlib import Path
import sys
import polars as pl

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
pl.Config.set_tbl_rows(25)
pl.Config.set_tbl_cols(25)

BASE = Path.cwd()

FILES = {
    "modeling_combined_1s": BASE / "data/raw/modeling_data_combined/kalshi_model_1s_combined.parquet",

    "old_grid_1s": BASE / "data/raw/kalshi_1s/kalshi_btc15m_grid_1s.parquet",
    "old_observed_1s": BASE / "data/raw/kalshi_1s/kalshi_btc15m_observed_1s.parquet",

    "april_may_grid_1s": BASE / "data/raw/kalshi_1s_april_may/kalshi_btc15m_grid_1s.parquet",
    "april_may_observed_1s": BASE / "data/raw/kalshi_1s_april_may/kalshi_btc15m_observed_1s.parquet",

    "old_decision_times": BASE / "data/raw/kalshi_decision_time/kalshi_contract_decision_times.parquet",
    "april_may_decision_times": BASE / "data/raw/kalshi_decision_time_april_may/kalshi_contract_decision_times.parquet",
}


def print_section(title):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def safe_collect(lf, label):
    try:
        return lf.collect()
    except Exception as e:
        print(f"Could not collect {label}: {e}")
        return None


def inspect_file(name, path):
    print_section(name)
    print(path)

    if not path.exists():
        print("MISSING")
        return

    try:
        lf = pl.scan_parquet(path)
        schema = lf.collect_schema()
    except Exception as e:
        print(f"Could not scan file: {e}")
        return

    print("\nCOLUMNS:")
    for col, dtype in schema.items():
        print(f"  {col}: {dtype}")

    print("\nROW COUNT:")
    print(safe_collect(lf.select(pl.len().alias("rows")), "row count"))

    print("\nHEAD:")
    print(safe_collect(lf.head(10), "head"))

    names = schema.names()

    time_cols = [
        c for c in names
        if any(s in c.lower() for s in ["time", "timestamp", "ts", "date", "second"])
    ]

    if time_cols:
        print("\nTIME-LIKE COLUMN RANGES:")
        for c in time_cols:
            try:
                out = lf.select(
                    pl.col(c).min().alias("min"),
                    pl.col(c).max().alias("max"),
                ).collect()
                print(f"\n{c}:")
                print(out)
            except Exception as e:
                print(f"{c}: could not summarize: {e}")

    id_cols = [
        c for c in names
        if any(s in c.lower() for s in ["ticker", "market", "contract", "event", "symbol"])
    ]

    if id_cols:
        print("\nID / TICKER SAMPLE VALUES:")
        for c in id_cols:
            try:
                out = (
                    lf.select(pl.col(c))
                    .drop_nulls()
                    .unique()
                    .head(12)
                    .collect()
                )
                print(f"\n{c}:")
                print(out)
            except Exception as e:
                print(f"{c}: could not sample: {e}")

    price_cols = [
        c for c in names
        if any(s in c.lower() for s in [
            "price", "bid", "ask", "mid", "yes", "no", "spread",
            "threshold", "strike", "distance", "underlying", "futures"
        ])
    ]

    numeric_price_cols = []
    for c in price_cols:
        try:
            if schema[c].is_numeric():
                numeric_price_cols.append(c)
        except Exception:
            pass

    if numeric_price_cols:
        print("\nNUMERIC PRICE / DISTANCE SUMMARY:")
        exprs = []
        for c in numeric_price_cols[:40]:
            exprs.extend([
                pl.col(c).min().alias(f"{c}_min"),
                pl.col(c).max().alias(f"{c}_max"),
                pl.col(c).mean().alias(f"{c}_mean"),
            ])

        try:
            print(lf.select(exprs).collect())
        except Exception as e:
            print(f"Could not summarize numeric price cols: {e}")

    # Useful quick checks for dashboard feasibility.
    likely_contract_cols = [c for c in names if c.lower() in ["ticker", "contract_ticker", "market_ticker"]]
    likely_time_cols = [c for c in names if c.lower() in ["timestamp", "ts", "time", "second", "window_second"]]

    if likely_contract_cols and time_cols:
        print("\nPOTENTIAL DASHBOARD KEYS:")
        print(f"Contract-like columns: {likely_contract_cols}")
        print(f"Time-like columns: {time_cols[:10]}")


def main():
    print("BASE:")
    print(BASE)

    print("\nFILES:")
    for name, path in FILES.items():
        status = "FOUND" if path.exists() else "MISSING"
        print(f"{status:8} {name:28} {path}")

    for name, path in FILES.items():
        inspect_file(name, path)


if __name__ == "__main__":
    main()