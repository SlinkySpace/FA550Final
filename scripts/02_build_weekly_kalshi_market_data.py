from pathlib import Path
import polars as pl

# Build weekly Kalshi/futures app-ready files.
# Run from FA550_BTC_Capstone root:
# python .\code\02_build_weekly_kalshi_market_data.py

BASE = Path(__file__).resolve().parents[1]

FUTURES_PATH = BASE / "data/raw/live/processed/microstructure_1s_mvp.parquet"
KALSHI_ANALYSIS_PATH = BASE / "data/raw/kalshi/kxbtc15m_analysis.parquet"
LABELS_LIVE_PATH = BASE / "data/raw/live/processed/labels_live.parquet"

APP_DIR = BASE / "data/app"
WEEKS_DIR = APP_DIR / "weeks"
WEEK_INDEX_PATH = APP_DIR / "week_index.csv"

APP_DIR.mkdir(parents=True, exist_ok=True)
WEEKS_DIR.mkdir(parents=True, exist_ok=True)


def clean_kalshi_analysis() -> pl.DataFrame:
    if not KALSHI_ANALYSIS_PATH.exists():
        raise FileNotFoundError(f"Missing {KALSHI_ANALYSIS_PATH}")

    print(f"Reading Kalshi analysis: {KALSHI_ANALYSIS_PATH}")

    df = (
        pl.scan_parquet(KALSHI_ANALYSIS_PATH)
        .select(
            [
                "ticker",
                "event_ticker",
                "title",
                "subtitle",
                "open_time",
                "close_time",
                "floor_strike",
                "strike_type",
                "result_binary",
                "last_price",
                "yes_ask",
                "yes_bid",
                "spread",
                "volume",
                "open_interest",
                "liquidity",
                "session",
                "price_bucket",
            ]
        )
        .filter(pl.col("close_time").is_not_null())
        .with_columns(
            [
                pl.col("open_time").dt.cast_time_unit("us"),
                pl.col("close_time").dt.cast_time_unit("us"),
            ]
        )
        .with_columns(
            [
                # Market prices are stored in dollars. Cents are easier for visuals.
                (pl.col("last_price") * 100).alias("last_price_cents"),
                (pl.col("yes_ask") * 100).alias("yes_ask_cents"),
                (pl.col("yes_bid") * 100).alias("yes_bid_cents"),
                (pl.col("spread") * 100).alias("spread_cents"),
                ((pl.col("yes_ask") + pl.col("yes_bid")) / 2 * 100).alias("yes_mid_cents"),
                pl.col("close_time").dt.truncate("1w").alias("week_start"),
                pl.col("close_time").dt.strftime("%Y-%m-%d").alias("close_date"),
            ]
        )
        .collect()
    )

    return df


def clean_labels_live() -> pl.DataFrame:
    if not LABELS_LIVE_PATH.exists():
        raise FileNotFoundError(f"Missing {LABELS_LIVE_PATH}")

    print(f"Reading labels: {LABELS_LIVE_PATH}")

    df = (
        pl.scan_parquet(LABELS_LIVE_PATH)
        .select(
            [
                "window_start",
                "window_end",
                "decision_point",
                "close_start",
                "decision_price",
                "close_price",
                "has_gap",
                "price_change",
                "price_change_pct",
                "label",
            ]
        )
        .with_columns(
            [
                pl.col("window_start").dt.cast_time_unit("us"),
                pl.col("window_end").dt.cast_time_unit("us"),
                pl.col("decision_point").dt.cast_time_unit("us"),
                pl.col("close_start").dt.cast_time_unit("us"),
            ]
        )
        .with_columns(
            [
                pl.col("window_start").dt.truncate("1w").alias("week_start"),
                ((pl.col("window_end") - pl.col("window_start")).dt.total_seconds()).alias("window_seconds"),
                ((pl.col("window_end") - pl.col("decision_point")).dt.total_seconds()).alias("seconds_from_decision_to_end"),
                ((pl.col("window_end") - pl.col("close_start")).dt.total_seconds()).alias("seconds_from_close_start_to_end"),
            ]
        )
        .collect()
    )

    return df


def build_futures_15m_windows() -> pl.DataFrame:
    if not FUTURES_PATH.exists():
        raise FileNotFoundError(f"Missing {FUTURES_PATH}")

    print(f"Building 15m futures windows from: {FUTURES_PATH}")

    df = (
        pl.scan_parquet(FUTURES_PATH)
        .select(
            [
                "timestamp",
                "symbol",
                "mid_price",
                "spread",
                "spread_bps",
                "trade_count",
                "trade_volume",
                "rv_60s",
                "vol_regime",
                "event_large_move",
            ]
        )
        .filter(pl.col("timestamp").is_not_null())
        .filter(pl.col("mid_price").is_not_null())
        .with_columns(
            [
                pl.col("timestamp").dt.cast_time_unit("us").alias("timestamp"),
            ]
        )
        .with_columns(
            [
                pl.col("timestamp").dt.truncate("15m").alias("window_start"),
                pl.col("timestamp").dt.truncate("1w").alias("week_start"),
            ]
        )
        .group_by(["window_start", "symbol", "week_start"])
        .agg(
            [
                pl.col("timestamp").min().alias("first_timestamp"),
                pl.col("timestamp").max().alias("last_timestamp"),
                pl.col("mid_price").first().alias("futures_open"),
                pl.col("mid_price").last().alias("futures_close"),
                pl.col("mid_price").max().alias("futures_high"),
                pl.col("mid_price").min().alias("futures_low"),
                pl.col("spread").mean().alias("spread_mean"),
                pl.col("spread_bps").mean().alias("spread_bps_mean"),
                pl.col("trade_count").sum().alias("trade_count_sum"),
                pl.col("trade_volume").sum().alias("trade_volume_sum"),
                pl.col("rv_60s").mean().alias("rv_60s_mean"),
                pl.col("event_large_move").sum().alias("large_move_rows"),
                pl.col("vol_regime").drop_nulls().last().alias("vol_regime"),
            ]
        )
        .with_columns(
            [
                (pl.col("futures_close") - pl.col("futures_open")).alias("futures_change"),
                (
                    (pl.col("futures_close") - pl.col("futures_open"))
                    / pl.col("futures_open")
                    * 100
                ).alias("futures_return_pct"),
            ]
        )
        .sort(["week_start", "window_start", "symbol"])
        .collect()
    )

    return df


def choose_primary_symbol_by_week(futures_15m: pl.DataFrame) -> pl.DataFrame:
    # The futures contract rolls across BTCZ5, BTCF6, BTCG6, BTCH6.
    # Because the dashboard only loads one week at a time, choose the most populated
    # futures symbol separately for each week.
    symbol_counts = (
        futures_15m
        .group_by(["week_start", "symbol"])
        .agg(pl.len().alias("rows"))
        .sort(["week_start", "rows"], descending=[False, True])
    )

    print("\nFutures symbols by week:")
    print(symbol_counts)

    dominant = (
        symbol_counts
        .group_by("week_start")
        .agg(
            [
                pl.col("symbol").first().alias("dominant_symbol"),
                pl.col("rows").first().alias("dominant_symbol_rows"),
            ]
        )
    )

    print("\nDominant futures symbol by week:")
    print(dominant.sort("week_start"))

    out = (
        futures_15m
        .join(dominant, on="week_start", how="left")
        .filter(pl.col("symbol") == pl.col("dominant_symbol"))
        .drop(["dominant_symbol", "dominant_symbol_rows"])
    )

    return out


def build_window_behavior(labels: pl.DataFrame, futures_15m_primary: pl.DataFrame) -> pl.DataFrame:
    out = (
        labels.join(
            futures_15m_primary,
            on=["window_start", "week_start"],
            how="left",
        )
        .with_columns(
            [
                pl.when(pl.col("label") == 1)
                .then(pl.lit("Up / YES-like"))
                .otherwise(pl.lit("Down / NO-like"))
                .alias("window_direction"),

                pl.when(pl.col("price_change") >= 0)
                .then(pl.lit("Up"))
                .otherwise(pl.lit("Down"))
                .alias("futures_direction"),
            ]
        )
        .sort("window_start")
    )

    return out


def build_market_threshold_matches(
    kalshi: pl.DataFrame,
    futures_15m_primary: pl.DataFrame,
) -> pl.DataFrame:
    # Approximate each Kalshi market's close with the futures 15m window ending at close_time.
    # Since futures_15m window_start is the beginning of the 15m period, use close_time - 15m.
    kalshi_keyed = kalshi.with_columns(
        [
            (pl.col("close_time") - pl.duration(minutes=15))
            .dt.cast_time_unit("us")
            .dt.truncate("15m")
            .alias("window_start")
        ]
    )

    matched = (
        kalshi_keyed.join(
            futures_15m_primary,
            on=["window_start", "week_start"],
            how="left",
        )
        .with_columns(
            [
                (pl.col("futures_close") - pl.col("floor_strike")).alias("distance_to_threshold"),
                (pl.col("futures_close") - pl.col("floor_strike")).abs().alias("abs_distance_to_threshold"),

                pl.when(pl.col("floor_strike").is_not_null() & pl.col("futures_close").is_not_null())
                .then((pl.col("futures_close") - pl.col("floor_strike")) / pl.col("futures_close") * 10_000)
                .otherwise(None)
                .alias("distance_to_threshold_bps"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("abs_distance_to_threshold") <= 25).then(pl.lit("±$25"))
                .when(pl.col("abs_distance_to_threshold") <= 50).then(pl.lit("±$50"))
                .when(pl.col("abs_distance_to_threshold") <= 100).then(pl.lit("±$100"))
                .when(pl.col("abs_distance_to_threshold") <= 250).then(pl.lit("±$250"))
                .otherwise(pl.lit("Outside ±$250"))
                .alias("threshold_band"),
            ]
        )
        .sort(["close_time", "ticker"])
    )

    return matched


def update_or_create_week_index(
    kalshi: pl.DataFrame,
    labels: pl.DataFrame,
    futures_15m: pl.DataFrame,
) -> pl.DataFrame:
    kalshi_idx = (
        kalshi.group_by("week_start")
        .agg(
            [
                pl.col("close_time").min().alias("kalshi_start"),
                pl.col("close_time").max().alias("kalshi_end"),
                pl.len().alias("n_kalshi_markets"),
                pl.col("event_ticker").n_unique().alias("n_kalshi_events"),
            ]
        )
    )

    labels_idx = (
        labels.group_by("week_start")
        .agg(
            [
                pl.len().alias("n_label_windows"),
                pl.col("price_change").abs().mean().alias("avg_abs_15m_futures_change"),
            ]
        )
    )

    futures_idx = (
        futures_15m.group_by("week_start")
        .agg(
            [
                pl.len().alias("n_futures_15m_rows"),
                pl.col("large_move_rows").sum().alias("n_large_move_rows_15m"),
            ]
        )
    )

    idx = (
        kalshi_idx
        .join(labels_idx, on="week_start", how="full", coalesce=True)
        .join(futures_idx, on="week_start", how="full", coalesce=True)
        .sort("week_start")
        .with_columns(
            [
                pl.col("week_start").dt.strftime("%Y-%m-%d").alias("week_id"),
                (
                    pl.col("week_start").dt.strftime("%b %d, %Y")
                    + " to "
                    + (pl.col("week_start") + pl.duration(days=6)).dt.strftime("%b %d, %Y")
                ).alias("week_label"),
            ]
        )
    )

    idx.write_csv(WEEK_INDEX_PATH)
    print(f"\nWrote week index: {WEEK_INDEX_PATH}")
    print(idx)

    return idx


def write_weekly_files(
    kalshi: pl.DataFrame,
    labels: pl.DataFrame,
    futures_15m: pl.DataFrame,
    window_behavior: pl.DataFrame,
    market_thresholds: pl.DataFrame,
    week_index: pl.DataFrame,
):
    for row in week_index.iter_rows(named=True):
        week_start = row["week_start"]
        week_id = row["week_id"]

        week_dir = WEEKS_DIR / week_id
        week_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nWriting week {week_id}...")

        kalshi_week = kalshi.filter(pl.col("week_start") == week_start)
        labels_week = labels.filter(pl.col("week_start") == week_start)
        futures_week = futures_15m.filter(pl.col("week_start") == week_start)
        behavior_week = window_behavior.filter(pl.col("week_start") == week_start)
        thresholds_week = market_thresholds.filter(pl.col("week_start") == week_start)

        kalshi_week.write_parquet(week_dir / "kalshi_markets.parquet")
        labels_week.write_parquet(week_dir / "labels_15m.parquet")
        futures_week.write_parquet(week_dir / "futures_15m_windows.parquet")
        behavior_week.write_parquet(week_dir / "futures_label_window_behavior.parquet")
        thresholds_week.write_parquet(week_dir / "kalshi_threshold_matches.parquet")

        avg_abs_change = None
        if labels_week.height > 0:
            avg_abs_change = labels_week.select(pl.col("price_change").abs().mean()).item()

        near_threshold_100 = thresholds_week.filter(
            pl.col("abs_distance_to_threshold") <= 100
        ).height

        summary = pl.DataFrame(
            {
                "metric": [
                    "kalshi_markets",
                    "label_windows",
                    "futures_15m_rows",
                    "threshold_matches",
                    "near_threshold_100_count",
                    "avg_abs_15m_futures_change",
                ],
                "value": [
                    float(kalshi_week.height),
                    float(labels_week.height),
                    float(futures_week.height),
                    float(thresholds_week.height),
                    float(near_threshold_100),
                    float(avg_abs_change) if avg_abs_change is not None else None,
                ],
            }
        )

        summary.write_csv(week_dir / "summary_metrics.csv")

        print(f"  kalshi_markets: {kalshi_week.height}")
        print(f"  labels_15m: {labels_week.height}")
        print(f"  futures_15m_windows: {futures_week.height}")
        print(f"  threshold matches: {thresholds_week.height}")
        print(f"  near threshold ±$100: {near_threshold_100}")


def main():
    kalshi = clean_kalshi_analysis()
    labels = clean_labels_live()
    futures_15m = build_futures_15m_windows()
    futures_15m_primary = choose_primary_symbol_by_week(futures_15m)

    window_behavior = build_window_behavior(labels, futures_15m_primary)
    market_thresholds = build_market_threshold_matches(kalshi, futures_15m_primary)

    week_index = update_or_create_week_index(
        kalshi=kalshi,
        labels=labels,
        futures_15m=futures_15m_primary,
    )

    write_weekly_files(
        kalshi=kalshi,
        labels=labels,
        futures_15m=futures_15m_primary,
        window_behavior=window_behavior,
        market_thresholds=market_thresholds,
        week_index=week_index,
    )

    print("\nDone.")
    print("Weekly files written under:")
    print(WEEKS_DIR)


if __name__ == "__main__":
    main()