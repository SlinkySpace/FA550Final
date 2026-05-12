from pathlib import Path
import polars as pl

# Build weekly dashboard-ready contract behavior files from the combined 1s Kalshi/futures table.
# Run from FA550_BTC_Capstone root:
# python .\code\03_build_weekly_contract_behavior.py

BASE = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE / "data/raw/modeling_data_combined/kalshi_model_1s_combined.parquet"
APP_DIR = BASE / "data/app"
WEEKS_DIR = APP_DIR / "weeks"
WEEK_INDEX_PATH = APP_DIR / "week_index_contract_behavior.csv"

APP_DIR.mkdir(parents=True, exist_ok=True)
WEEKS_DIR.mkdir(parents=True, exist_ok=True)

MAX_SCATTER_ROWS_PER_WEEK = 150_000


KEEP_COLUMNS = [
    "ticker",
    "second",
    "relative_second",
    "seconds_to_close",
    "minutes_to_close",
    "contract_open_time",
    "contract_close_time",
    "selected_futures_symbol",
    "futures_symbol",
    "underlying_source",
    "underlying_value",
    "reference_underlying",
    "underlying_diff",
    "underlying_return_from_open",
    "underlying_bps_from_open",
    "underlying_return_1s",
    "log_moneyness",
    "z_moneyness",
    "z_bps",
    "abs_z_bps",
    "time_to_close_seconds",
    "time_to_close_frac_15m",
    "yes_state_cents",
    "no_state_cents",
    "kalshi_yes_price",
    "kalshi_no_price",
    "kalshi_yes_price_clipped",
    "trade_count",
    "trade_size",
    "has_trade_this_second",
    "inferred_winner",
    "outcome_yes",
    "decision_found",
    "decision_time",
    "minutes_before_close_at_decision",
    "is_after_decision",
    "has_underlying",
    "underlying_path_complete",
    "has_futures_observation",
    "seconds_since_last_futures_update",
    "is_futures_stale",
    "best_bid",
    "best_ask",
    "spread",
    "spread_bps",
    "is_valid_spread",
    "data_source_period",
]


def load_base_lazy() -> pl.LazyFrame:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing file: {MODEL_PATH}")

    print(f"Reading combined modeling table: {MODEL_PATH}")

    lf = (
        pl.scan_parquet(MODEL_PATH)
        .select(KEEP_COLUMNS)
        .with_columns(
            [
                pl.col("second").dt.cast_time_unit("us").alias("second"),
                pl.col("contract_open_time").dt.cast_time_unit("us").alias("contract_open_time"),
                pl.col("contract_close_time").dt.cast_time_unit("us").alias("contract_close_time"),
                pl.col("decision_time").dt.cast_time_unit("us").alias("decision_time"),
            ]
        )
        .with_columns(
            [
                pl.col("contract_close_time").dt.truncate("1w").alias("week_start"),

                # More readable contract price columns.
                pl.col("kalshi_yes_price").alias("yes_price_cents"),
                pl.col("kalshi_no_price").alias("no_price_cents"),

                # Empirical contract price movement.
                (
                    pl.col("kalshi_yes_price")
                    - pl.col("kalshi_yes_price").shift(1).over("ticker")
                ).alias("yes_price_change_1s"),

                (
                    pl.col("kalshi_yes_price")
                    - pl.col("kalshi_yes_price").shift(5).over("ticker")
                ).alias("yes_price_change_5s"),

                (
                    pl.col("underlying_value")
                    - pl.col("underlying_value").shift(1).over("ticker")
                ).alias("underlying_change_1s"),

                (
                    pl.col("underlying_value")
                    - pl.col("underlying_value").shift(5).over("ticker")
                ).alias("underlying_change_5s"),
            ]
        )
        .with_columns(
            [
                pl.col("yes_price_change_5s").abs().alias("abs_yes_price_change_5s"),
                pl.col("underlying_change_5s").abs().alias("abs_underlying_change_5s"),

                pl.when(pl.col("abs_z_bps") <= 5).then(pl.lit("Very near"))
                .when(pl.col("abs_z_bps") <= 10).then(pl.lit("Near"))
                .when(pl.col("abs_z_bps") <= 25).then(pl.lit("Moderate"))
                .when(pl.col("abs_z_bps") <= 50).then(pl.lit("Far"))
                .otherwise(pl.lit("Very far"))
                .alias("moneyness_bucket"),

                pl.when(pl.col("minutes_to_close") <= 1).then(pl.lit("0-1m"))
                .when(pl.col("minutes_to_close") <= 2).then(pl.lit("1-2m"))
                .when(pl.col("minutes_to_close") <= 3).then(pl.lit("2-3m"))
                .when(pl.col("minutes_to_close") <= 5).then(pl.lit("3-5m"))
                .when(pl.col("minutes_to_close") <= 7).then(pl.lit("5-7m"))
                .when(pl.col("minutes_to_close") <= 10).then(pl.lit("7-10m"))
                .otherwise(pl.lit("10-15m"))
                .alias("time_to_close_bucket"),

                pl.when(pl.col("kalshi_yes_price").is_between(20, 80))
                .then(pl.lit("Uncertain 20-80c"))
                .otherwise(pl.lit("Mostly decided"))
                .alias("uncertainty_bucket"),
            ]
        )
    )

    return lf


def build_week_index(lf: pl.LazyFrame) -> pl.DataFrame:
    print("Building contract-behavior week index...")

    idx = (
        lf.group_by("week_start")
        .agg(
            [
                pl.col("second").min().alias("start_time"),
                pl.col("second").max().alias("end_time"),
                pl.len().alias("n_rows"),
                pl.col("ticker").n_unique().alias("n_contracts"),
                pl.col("has_trade_this_second").sum().alias("trade_seconds"),
                pl.col("decision_found").sum().alias("decision_rows"),
                pl.col("data_source_period").drop_nulls().n_unique().alias("n_source_periods"),
            ]
        )
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
        .collect()
    )

    idx.write_csv(WEEK_INDEX_PATH)
    print(f"Wrote {WEEK_INDEX_PATH}")
    print(idx)

    return idx


def make_decision_times(week_df: pl.DataFrame) -> pl.DataFrame:
    cols = [
        "ticker",
        "contract_open_time",
        "contract_close_time",
        "inferred_winner",
        "outcome_yes",
        "decision_found",
        "decision_time",
        "minutes_before_close_at_decision",
        "data_source_period",
    ]

    available = [c for c in cols if c in week_df.columns]

    decision = (
        week_df
        .select(available)
        .unique(subset=["ticker"])
        .sort("contract_close_time")
    )

    return decision


def make_contract_summary(week_df: pl.DataFrame) -> pl.DataFrame:
    summary = (
        week_df
        .group_by("ticker")
        .agg(
            [
                pl.col("contract_open_time").first().alias("contract_open_time"),
                pl.col("contract_close_time").first().alias("contract_close_time"),
                pl.col("inferred_winner").first().alias("inferred_winner"),
                pl.col("outcome_yes").first().alias("outcome_yes"),
                pl.col("decision_found").first().alias("decision_found"),
                pl.col("decision_time").first().alias("decision_time"),
                pl.col("minutes_before_close_at_decision").first().alias("minutes_before_close_at_decision"),
                pl.col("selected_futures_symbol").first().alias("selected_futures_symbol"),
                pl.col("underlying_value").first().alias("underlying_open"),
                pl.col("underlying_value").last().alias("underlying_close"),
                pl.col("kalshi_yes_price").first().alias("yes_open_cents"),
                pl.col("kalshi_yes_price").last().alias("yes_close_cents"),
                pl.col("kalshi_yes_price").max().alias("yes_max_cents"),
                pl.col("kalshi_yes_price").min().alias("yes_min_cents"),
                pl.col("trade_count").sum().alias("total_trade_count"),
                pl.col("trade_size").sum().alias("total_trade_size"),
                pl.col("abs_z_bps").min().alias("min_abs_z_bps"),
                pl.col("abs_z_bps").mean().alias("mean_abs_z_bps"),
            ]
        )
        .with_columns(
            [
                (pl.col("underlying_close") - pl.col("underlying_open")).alias("underlying_change"),
                (pl.col("yes_close_cents") - pl.col("yes_open_cents")).alias("yes_price_change"),
            ]
        )
        .sort("contract_close_time")
    )

    return summary


def make_sensitivity_bins(week_df: pl.DataFrame) -> pl.DataFrame:
    bins = (
        week_df
        .filter(pl.col("yes_price_change_5s").is_not_null())
        .filter(pl.col("underlying_change_5s").is_not_null())
        .group_by(["time_to_close_bucket", "moneyness_bucket"])
        .agg(
            [
                pl.len().alias("n_rows"),
                pl.col("abs_yes_price_change_5s").mean().alias("mean_abs_yes_price_change_5s"),
                pl.col("yes_price_change_5s").mean().alias("mean_yes_price_change_5s"),
                pl.col("underlying_change_5s").mean().alias("mean_underlying_change_5s"),
                pl.col("abs_underlying_change_5s").mean().alias("mean_abs_underlying_change_5s"),
                pl.col("has_trade_this_second").sum().alias("trade_seconds"),
            ]
        )
        .sort(["time_to_close_bucket", "moneyness_bucket"])
    )

    return bins


def make_threshold_heatmap(week_df: pl.DataFrame) -> pl.DataFrame:
    # Heatmap data: time-to-close bucket x moneyness bucket -> average absolute contract movement.
    heatmap = (
        week_df
        .filter(pl.col("abs_yes_price_change_5s").is_not_null())
        .group_by(["time_to_close_bucket", "moneyness_bucket"])
        .agg(
            [
                pl.len().alias("n_rows"),
                pl.col("abs_yes_price_change_5s").mean().alias("mean_abs_yes_price_change_5s"),
                pl.col("kalshi_yes_price").mean().alias("mean_yes_price_cents"),
                pl.col("abs_z_bps").mean().alias("mean_abs_z_bps"),
            ]
        )
        .sort(["time_to_close_bucket", "moneyness_bucket"])
    )

    return heatmap


def make_event_window_response(week_df: pl.DataFrame) -> pl.DataFrame:
    # For now, use contract-relative time from 15m before close to close.
    # This supports average contract path, uncertainty curve, and underlying path.
    event = (
        week_df
        .group_by(["relative_second", "time_to_close_bucket", "moneyness_bucket"])
        .agg(
            [
                pl.len().alias("n_rows"),
                pl.col("underlying_diff").mean().alias("mean_underlying_diff"),
                pl.col("underlying_bps_from_open").mean().alias("mean_underlying_bps_from_open"),
                pl.col("kalshi_yes_price").mean().alias("mean_yes_price_cents"),
                pl.col("kalshi_no_price").mean().alias("mean_no_price_cents"),
                pl.col("abs_z_bps").mean().alias("mean_abs_z_bps"),
                pl.col("has_trade_this_second").sum().alias("trade_seconds"),
            ]
        )
        .sort(["relative_second", "time_to_close_bucket", "moneyness_bucket"])
    )

    return event


def make_plot_sample(week_df: pl.DataFrame) -> pl.DataFrame:
    if week_df.height <= MAX_SCATTER_ROWS_PER_WEEK:
        return week_df

    frac = MAX_SCATTER_ROWS_PER_WEEK / week_df.height
    return week_df.sample(fraction=frac, seed=42)


def write_week_files(lf: pl.LazyFrame, week_index: pl.DataFrame):
    for row in week_index.iter_rows(named=True):
        week_start = row["week_start"]
        week_id = row["week_id"]

        week_dir = WEEKS_DIR / week_id
        week_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nBuilding contract behavior for {week_id}...")

        week_df = (
            lf.filter(pl.col("week_start") == week_start)
            .sort(["ticker", "second"])
            .collect()
        )

        if week_df.is_empty():
            print("  no rows")
            continue

        # Core files.
        contract_summary = make_contract_summary(week_df)
        decision_times = make_decision_times(week_df)
        sensitivity_bins = make_sensitivity_bins(week_df)
        threshold_heatmap = make_threshold_heatmap(week_df)
        event_window_response = make_event_window_response(week_df)
        plot_sample = make_plot_sample(week_df)

        # Write app-ready weekly files.
        contract_summary.write_parquet(week_dir / "contract_summary.parquet")
        decision_times.write_parquet(week_dir / "contract_decision_times.parquet")
        sensitivity_bins.write_parquet(week_dir / "sensitivity_bins.parquet")
        threshold_heatmap.write_parquet(week_dir / "threshold_heatmap.parquet")
        event_window_response.write_parquet(week_dir / "contract_event_window_response.parquet")
        plot_sample.write_parquet(week_dir / "contract_plot_sample.parquet")

        # For contract evolution page, keep full 1s data but selected columns only.
        evolution_cols = [
            "ticker",
            "second",
            "relative_second",
            "seconds_to_close",
            "minutes_to_close",
            "contract_open_time",
            "contract_close_time",
            "selected_futures_symbol",
            "underlying_value",
            "reference_underlying",
            "underlying_diff",
            "underlying_return_1s",
            "z_bps",
            "abs_z_bps",
            "moneyness_bucket",
            "yes_price_cents",
            "no_price_cents",
            "yes_price_change_1s",
            "yes_price_change_5s",
            "underlying_change_1s",
            "underlying_change_5s",
            "has_trade_this_second",
            "trade_count",
            "trade_size",
            "inferred_winner",
            "outcome_yes",
            "decision_found",
            "decision_time",
            "minutes_before_close_at_decision",
            "is_after_decision",
            "best_bid",
            "best_ask",
            "spread",
            "spread_bps",
            "time_to_close_bucket",
            "uncertainty_bucket",
            "data_source_period",
        ]

        available_evolution_cols = [c for c in evolution_cols if c in week_df.columns]
        week_df.select(available_evolution_cols).write_parquet(
            week_dir / "contract_evolution_1s.parquet"
        )

        # Summary metrics for dashboard cards.
        metrics = pl.DataFrame(
            {
                "metric": [
                    "contract_behavior_rows",
                    "contracts",
                    "trade_seconds",
                    "decision_contracts",
                    "avg_minutes_before_decision",
                    "near_moneyness_rows_abs_z_bps_10",
                    "uncertain_20_80_rows",
                ],
                "value": [
                    float(week_df.height),
                    float(week_df["ticker"].n_unique()),
                    float(week_df["has_trade_this_second"].sum()),
                    float(decision_times.filter(pl.col("decision_found") == True).height),
                    float(
                        decision_times
                        .filter(pl.col("decision_found") == True)
                        .select(pl.col("minutes_before_close_at_decision").mean())
                        .item()
                    ) if decision_times.filter(pl.col("decision_found") == True).height > 0 else None,
                    float(week_df.filter(pl.col("abs_z_bps") <= 10).height),
                    float(week_df.filter(pl.col("kalshi_yes_price").is_between(20, 80)).height),
                ],
            }
        )

        metrics.write_csv(week_dir / "contract_behavior_summary_metrics.csv")

        print(f"  rows: {week_df.height:,}")
        print(f"  contracts: {week_df['ticker'].n_unique():,}")
        print(f"  wrote weekly contract behavior files")


def main():
    lf = load_base_lazy()
    week_index = build_week_index(lf)
    write_week_files(lf, week_index)

    print("\nDone.")
    print("Weekly contract behavior files written under:")
    print(WEEKS_DIR)


if __name__ == "__main__":
    main()