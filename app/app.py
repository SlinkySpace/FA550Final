from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# FA550 BTC Capstone Dashboard
# BTC Futures and Prediction Contract Price Behavior
# ============================================================

BASE = Path(__file__).resolve().parents[1]
APP_DATA = BASE / "data" / "app"
WEEKS_DIR = APP_DATA / "weeks"

WEEK_INDEX_CONTRACT_PATH = APP_DATA / "week_index_contract_behavior.csv"
WEEK_INDEX_MARKET_PATH = APP_DATA / "week_index.csv"

st.set_page_config(
    page_title="BTC Futures & Prediction Contract Behavior",
    layout="wide",
)


# ============================================================
# Helpers
# ============================================================

@st.cache_data(show_spinner=False)
def load_week_index() -> pd.DataFrame:
    if WEEK_INDEX_CONTRACT_PATH.exists():
        df = pd.read_csv(WEEK_INDEX_CONTRACT_PATH)
    elif WEEK_INDEX_MARKET_PATH.exists():
        df = pd.read_csv(WEEK_INDEX_MARKET_PATH)
    else:
        st.error("Missing week index file. Run the weekly build scripts first.")
        st.stop()

    if "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(df["week_start"], utc=True, errors="coerce")

    df["display"] = df["week_label"] + "  (" + df["week_id"] + ")"
    return df


@st.cache_data(show_spinner=False)
def load_week_file(week_id: str, filename: str) -> pd.DataFrame:
    path = WEEKS_DIR / week_id / filename

    if not path.exists():
        return pd.DataFrame()

    if filename.endswith(".csv"):
        return pd.read_csv(path)

    return pd.read_parquet(path)


def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    datetime_cols = {
        "second",
        "window_start",
        "window_end",
        "decision_point",
        "close_start",
        "contract_open_time",
        "contract_close_time",
        "decision_time",
        "open_time",
        "close_time",
        "first_timestamp",
        "last_timestamp",
        "kalshi_start",
        "kalshi_end",
        "start_time",
        "end_time",
    }

    for col in df.columns:
        if col in datetime_cols:
            try:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            except Exception:
                pass

    return df


def get_metric(summary: pd.DataFrame, metric: str, default=np.nan):
    if summary.empty or "metric" not in summary.columns or "value" not in summary.columns:
        return default

    row = summary.loc[summary["metric"] == metric, "value"]

    if row.empty:
        return default

    return row.iloc[0]


def fmt_int(x):
    try:
        if pd.isna(x):
            return "0"
        return f"{int(float(x)):,}"
    except Exception:
        return "0"


def fmt_float(x, digits=2):
    try:
        if pd.isna(x):
            return "—"
        return f"{float(x):,.{digits}f}"
    except Exception:
        return "—"


def safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def order_time_buckets(df: pd.DataFrame, col="time_to_close_bucket") -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df

    order = ["0-1m", "1-2m", "2-3m", "3-5m", "5-7m", "7-10m", "10-15m"]
    df = df.copy()
    df[col] = df[col].astype(str)
    df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df.sort_values(col)


def order_moneyness_buckets(df: pd.DataFrame, col="moneyness_bucket") -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df

    order = ["Very near", "Near", "Moderate", "Far", "Very far"]
    df = df.copy()
    df[col] = df[col].astype(str)
    df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df.sort_values(col)


def needs_price_multiplier(series: pd.Series) -> bool:
    vals = pd.to_numeric(series, errors="coerce").dropna()

    if vals.empty:
        return False

    return vals.abs().quantile(0.99) <= 1.5


def add_price_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    price_source_cols = [
        "yes_price_cents",
        "no_price_cents",
        "yes_open_cents",
        "yes_close_cents",
        "yes_max_cents",
        "yes_min_cents",
        "mean_yes_price_cents",
        "mean_no_price_cents",
    ]

    change_source_cols = [
        "yes_price_change",
        "yes_price_change_1s",
        "yes_price_change_5s",
        "abs_yes_price_change_5s",
        "mean_yes_price_change_5s",
        "mean_abs_yes_price_change_5s",
    ]

    multiplier = 1.0

    for col in price_source_cols + change_source_cols:
        if col in df.columns and needs_price_multiplier(df[col]):
            multiplier = 100.0
            break

    for col in price_source_cols + change_source_cols:
        if col in df.columns:
            df[col + "_display"] = pd.to_numeric(df[col], errors="coerce") * multiplier

    return df


def add_underlying_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    if "underlying_value" in df.columns:
        df["underlying_modeling"] = pd.to_numeric(df["underlying_value"], errors="coerce")

    trade_price_candidates = [
        "underlying_trade_price",
        "futures_trade_price",
        "trade_price",
        "last_trade_price",
        "trade_px",
        "last_price",
        "last_trade",
        "trade",
    ]

    for col in trade_price_candidates:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().any():
                df["underlying_trade_price"] = vals
                break

    if "best_bid" in df.columns and "best_ask" in df.columns:
        bid = pd.to_numeric(df["best_bid"], errors="coerce")
        ask = pd.to_numeric(df["best_ask"], errors="coerce")

        valid = bid.notna() & ask.notna() & (bid > 0) & (ask > 0) & (ask >= bid)
        df["underlying_quote_mid"] = np.where(valid, (bid + ask) / 2.0, np.nan)

        if "quote_spread" not in df.columns:
            df["quote_spread"] = np.where(valid, ask - bid, np.nan)

    return df


def get_underlying_plot_col(df: pd.DataFrame, selected_source: str) -> tuple[str | None, str]:
    if df.empty:
        return None, "No underlying data available"

    if selected_source == "Futures trade price":
        if "underlying_trade_price" in df.columns and df["underlying_trade_price"].notna().any():
            return "underlying_trade_price", "Futures trade price"
        if "underlying_quote_mid" in df.columns and df["underlying_quote_mid"].notna().any():
            return "underlying_quote_mid", "Quote mid"
        if "underlying_modeling" in df.columns and df["underlying_modeling"].notna().any():
            return "underlying_modeling", "Modeling underlying"

    if selected_source == "Quote mid from best_bid / best_ask":
        if "underlying_quote_mid" in df.columns and df["underlying_quote_mid"].notna().any():
            return "underlying_quote_mid", "Quote mid"
        if "underlying_trade_price" in df.columns and df["underlying_trade_price"].notna().any():
            return "underlying_trade_price", "Futures trade price"
        if "underlying_modeling" in df.columns and df["underlying_modeling"].notna().any():
            return "underlying_modeling", "Modeling underlying"

    if "underlying_modeling" in df.columns and df["underlying_modeling"].notna().any():
        return "underlying_modeling", "Modeling underlying"

    if "underlying_value" in df.columns and df["underlying_value"].notna().any():
        return "underlying_value", "Underlying value"

    return None, "No underlying data available"


def get_selected_contract_decision_time(
    selected_ticker: str,
    decision_times: pd.DataFrame,
    contract_df: pd.DataFrame,
):
    if not selected_ticker or decision_times.empty or contract_df.empty:
        return None

    if "ticker" not in decision_times.columns:
        return None

    rows = decision_times[decision_times["ticker"] == selected_ticker].copy()

    if rows.empty:
        return None

    if "decision_found" in rows.columns:
        rows = rows[rows["decision_found"] == True]

    if rows.empty:
        return None

    row = rows.iloc[0]

    if "minutes_before_close_at_decision" not in row.index:
        return None

    minutes_before = row["minutes_before_close_at_decision"]

    if pd.isna(minutes_before):
        return None

    close_time = None

    if "contract_close_time" in contract_df.columns and contract_df["contract_close_time"].notna().any():
        close_time = pd.to_datetime(contract_df["contract_close_time"].dropna().iloc[0], utc=True)

    if close_time is None or pd.isna(close_time):
        return None

    return close_time - pd.Timedelta(minutes=float(minutes_before))


def add_decision_close_shapes(
    fig,
    contract_df: pd.DataFrame,
    show_decision=True,
    decision_marker_time=None,
):
    if contract_df.empty:
        return fig

    close_time = None

    if "contract_close_time" in contract_df.columns and contract_df["contract_close_time"].notna().any():
        close_time = pd.to_datetime(contract_df["contract_close_time"].dropna().iloc[0], utc=True)

    if show_decision and decision_marker_time is not None and not pd.isna(decision_marker_time):
        decision_x = pd.to_datetime(decision_marker_time, utc=True).isoformat()

        fig.add_shape(
            type="line",
            x0=decision_x,
            x1=decision_x,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(dash="dash", color="orange", width=2),
        )

        fig.add_annotation(
            x=decision_x,
            y=1,
            xref="x",
            yref="paper",
            text="Decision marker",
            showarrow=False,
            yanchor="bottom",
            font=dict(color="orange"),
        )

    if close_time is not None:
        close_x = close_time.isoformat()

        fig.add_shape(
            type="line",
            x0=close_x,
            x1=close_x,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(dash="dot", color="red", width=2),
        )

        fig.add_annotation(
            x=close_x,
            y=1,
            xref="x",
            yref="paper",
            text="Close",
            showarrow=False,
            yanchor="bottom",
            font=dict(color="red"),
        )

        start_7 = (close_time - pd.Timedelta(minutes=7)).isoformat()
        start_2 = (close_time - pd.Timedelta(minutes=2)).isoformat()

        fig.add_shape(
            type="rect",
            x0=start_7,
            x1=start_2,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            fillcolor="gray",
            opacity=0.15,
            line_width=0,
            layer="below",
        )

        fig.add_annotation(
            x=start_7,
            y=0.98,
            xref="x",
            yref="paper",
            text="2-7 min window",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(color="gray"),
        )

    return fig


def adaptive_price_axis(df: pd.DataFrame, cols: list[str], full_axis: bool = False):
    if full_axis:
        return dict(range=[0, 100])

    vals = []

    for col in cols:
        if col in df.columns:
            vals.append(pd.to_numeric(df[col], errors="coerce"))

    if not vals:
        return dict(range=[0, 100])

    price_values = pd.concat(vals, axis=0).dropna()

    if price_values.empty:
        return dict(range=[0, 100])

    y_min = float(price_values.min())
    y_max = float(price_values.max())

    if y_min == y_max:
        pad = 2.0
    else:
        pad = max(2.0, (y_max - y_min) * 0.20)

    y_low = max(0.0, y_min - pad)
    y_high = min(100.0, y_max + pad)

    if y_high - y_low < 5:
        center = (y_high + y_low) / 2
        y_low = max(0.0, center - 2.5)
        y_high = min(100.0, center + 2.5)

    return dict(range=[y_low, y_high])


def get_moneyness_plot_spec(df: pd.DataFrame):
    if df.empty:
        return None, None, None, None

    dollar_candidates = [
        "distance_from_threshold_dollars",
        "threshold_distance_dollars",
        "underlying_distance_to_threshold",
        "distance_to_threshold",
        "threshold_distance",
    ]

    for col in dollar_candidates:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            return (
                col,
                "Distance from threshold ($)",
                50.0,
                "Near-threshold zone: ±$50",
            )

    if "z_bps" in df.columns and pd.to_numeric(df["z_bps"], errors="coerce").notna().any():
        return (
            "z_bps",
            "Distance from threshold (bps)",
            10.0,
            "Near-threshold zone: ±10 bps",
        )

    if "abs_z_bps" in df.columns and pd.to_numeric(df["abs_z_bps"], errors="coerce").notna().any():
        return (
            "abs_z_bps",
            "Absolute distance from threshold (bps)",
            10.0,
            "Near-threshold zone: 0 to 10 bps",
        )

    return None, None, None, None


def make_microstructure_frame(df: pd.DataFrame, underlying_source_choice: str) -> tuple[pd.DataFrame, str]:
    if df.empty:
        return pd.DataFrame(), "No data"

    work = df.copy()
    work = work.sort_values("second")

    underlying_col, underlying_label = get_underlying_plot_col(work, underlying_source_choice)

    agg_dict = {}

    if underlying_col is not None:
        agg_dict[underlying_col] = "mean"

    for col in ["spread", "spread_bps", "quote_spread", "trade_count", "trade_size", "trade_volume"]:
        if col in work.columns:
            agg_dict[col] = "mean" if col in ["spread", "spread_bps", "quote_spread"] else "sum"

    if not agg_dict:
        return pd.DataFrame(), "No usable microstructure fields"

    micro = work.groupby("second", as_index=False).agg(agg_dict)
    micro = micro.sort_values("second")

    if underlying_col is not None and underlying_col in micro.columns:
        micro["underlying_change_1s"] = micro[underlying_col].diff()
        micro["underlying_return_1s_bps"] = np.log(micro[underlying_col] / micro[underlying_col].shift(1)) * 10_000
        micro["local_vol_60s_bps"] = micro["underlying_return_1s_bps"].rolling(60, min_periods=20).std()

    return micro, underlying_label


# ============================================================
# Load week index and selected week
# ============================================================

week_index = load_week_index()

st.sidebar.title("Controls")

selected_display = st.sidebar.selectbox(
    "Analysis week",
    options=week_index["display"].tolist(),
    index=len(week_index) - 1,
)

selected_week = week_index.loc[week_index["display"] == selected_display].iloc[0]
week_id = selected_week["week_id"]

st.sidebar.caption(
    "This app intentionally loads only one week at a time. No open-ended date range selector is used."
)


# ============================================================
# Load weekly files
# ============================================================

summary_market = load_week_file(week_id, "summary_metrics.csv")
summary_contract = load_week_file(week_id, "contract_behavior_summary_metrics.csv")

futures_15m = parse_datetimes(load_week_file(week_id, "futures_15m_windows.parquet"))
threshold_matches = parse_datetimes(load_week_file(week_id, "kalshi_threshold_matches.parquet"))
kalshi_markets = parse_datetimes(load_week_file(week_id, "kalshi_markets.parquet"))

contract_summary = parse_datetimes(load_week_file(week_id, "contract_summary.parquet"))
decision_times = parse_datetimes(load_week_file(week_id, "contract_decision_times.parquet"))
sensitivity_bins = load_week_file(week_id, "sensitivity_bins.parquet")
threshold_heatmap = load_week_file(week_id, "threshold_heatmap.parquet")
plot_sample = parse_datetimes(load_week_file(week_id, "contract_plot_sample.parquet"))
contract_evolution = parse_datetimes(load_week_file(week_id, "contract_evolution_1s.parquet"))

contract_summary = add_price_display_columns(contract_summary)
decision_times = add_price_display_columns(decision_times)
sensitivity_bins = add_price_display_columns(sensitivity_bins)
threshold_heatmap = add_price_display_columns(threshold_heatmap)
plot_sample = add_price_display_columns(plot_sample)
contract_evolution = add_price_display_columns(contract_evolution)

plot_sample = add_underlying_display_columns(plot_sample)
contract_evolution = add_underlying_display_columns(contract_evolution)


# ============================================================
# Sidebar filters
# ============================================================

st.sidebar.subheader("Contract filters")

if not contract_summary.empty and "ticker" in contract_summary.columns:
    ticker_options = contract_summary.sort_values("contract_close_time")["ticker"].dropna().unique().tolist()
elif not contract_evolution.empty and "ticker" in contract_evolution.columns:
    ticker_options = contract_evolution["ticker"].dropna().unique().tolist()
else:
    ticker_options = []

selected_ticker = st.sidebar.selectbox(
    "Contract",
    options=ticker_options,
    index=0 if ticker_options else None,
    placeholder="No contracts available",
)

if not plot_sample.empty and "time_to_close_bucket" in plot_sample.columns:
    ttc_options = ["All"] + sorted(plot_sample["time_to_close_bucket"].dropna().astype(str).unique().tolist())
else:
    ttc_options = ["All"]

selected_ttc = st.sidebar.selectbox("Time-to-close bucket", ttc_options)

if not plot_sample.empty and "moneyness_bucket" in plot_sample.columns:
    money_options = ["All"] + sorted(plot_sample["moneyness_bucket"].dropna().astype(str).unique().tolist())
else:
    money_options = ["All"]

selected_money = st.sidebar.selectbox("Moneyness bucket", money_options)

if not plot_sample.empty and "data_source_period" in plot_sample.columns:
    source_options = ["All"] + sorted(plot_sample["data_source_period"].dropna().astype(str).unique().tolist())
else:
    source_options = ["All"]

selected_source = st.sidebar.selectbox("Data source period", source_options)

st.sidebar.subheader("Chart options")

underlying_source_choice = st.sidebar.selectbox(
    "Underlying source for evolution charts",
    options=[
        "Futures trade price",
        "Quote mid from best_bid / best_ask",
        "Modeling underlying_value",
    ],
    index=0,
)

adaptive_contract_axis = st.sidebar.checkbox(
    "Adaptive contract price axis",
    value=True,
    help="Zooms selected contract price charts to the observed range instead of always forcing 0-100 cents.",
)

show_trade_markers = st.sidebar.checkbox(
    "Show trade-second markers",
    value=False,
    help="Trade markers can clutter the price line when a contract trades almost every second.",
)


def apply_common_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    if selected_ttc != "All" and "time_to_close_bucket" in out.columns:
        out = out[out["time_to_close_bucket"].astype(str) == selected_ttc]

    if selected_money != "All" and "moneyness_bucket" in out.columns:
        out = out[out["moneyness_bucket"].astype(str) == selected_money]

    if selected_source != "All" and "data_source_period" in out.columns:
        out = out[out["data_source_period"].astype(str) == selected_source]

    return out


plot_sample_filtered = apply_common_filters(plot_sample)


# ============================================================
# Header
# ============================================================

st.title("BTC Futures and Prediction Contract Price Behavior")

st.markdown(
    """
This dashboard visualizes how short-dated BTC prediction contracts move with respect to BTC futures.
The focus is **price behavior**, not trading strategy: threshold risk, contract sensitivity,
time-to-close effects, and underlying futures movement.
"""
)

st.caption(
    f"Selected week: **{selected_week['week_label']}**. "
    "All loaded data is restricted to this selected week."
)


# ============================================================
# KPI cards
# ============================================================

st.subheader("Selected Week Overview")

k1, k2, k3, k4, k5, k6 = st.columns(6)

k1.metric(
    "Contracts",
    fmt_int(get_metric(summary_contract, "contracts", get_metric(summary_market, "kalshi_markets", 0))),
)

k2.metric(
    "Contract rows",
    fmt_int(get_metric(summary_contract, "contract_behavior_rows", 0)),
)

k3.metric(
    "Trade seconds",
    fmt_int(get_metric(summary_contract, "trade_seconds", 0)),
)

k4.metric(
    "Decision contracts",
    fmt_int(get_metric(summary_contract, "decision_contracts", 0)),
)

k5.metric(
    "Avg decision mins",
    fmt_float(get_metric(summary_contract, "avg_minutes_before_decision", np.nan), 2),
)

k6.metric(
    "Near-money rows",
    fmt_int(get_metric(summary_contract, "near_moneyness_rows_abs_z_bps_10", 0)),
)


# ============================================================
# Tabs
# ============================================================

tabs = st.tabs(
    [
        "Overview",
        "Contract Evolution",
        "Futures Microstructure",
        "Decision Time",
        "Threshold Risk",
        "Sensitivity",
        "Data Preview",
    ]
)


# ============================================================
# Tab 1: Overview
# ============================================================

with tabs[0]:
    st.header("Overview")

    st.markdown(
        """
The dashboard is built from precomputed weekly files. The main contract behavior file is a matched
1-second table with Kalshi contract prices, BTC futures/underlying values, moneyness, time-to-close,
and decision-time fields.
"""
    )

    c1, c2 = st.columns(2)

    with c1:
        y_col = "yes_price_change_display" if "yes_price_change_display" in contract_summary.columns else "yes_price_change"

        if not contract_summary.empty and y_col in contract_summary.columns:
            fig = px.histogram(
                contract_summary,
                x=y_col,
                nbins=50,
                title="Contract YES Price Change Over Full 15-Minute Life",
                labels={
                    y_col: "YES price change (cents)",
                    "count": "Contracts",
                },
            )
            fig.add_vline(x=0, line_dash="dash")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No contract summary price-change data for this week.")

    with c2:
        if not contract_summary.empty and "inferred_winner" in contract_summary.columns:
            winner_counts = (
                contract_summary["inferred_winner"]
                .fillna("Unknown")
                .value_counts()
                .reset_index()
            )
            winner_counts.columns = ["inferred_winner", "count"]

            fig = px.bar(
                winner_counts,
                x="inferred_winner",
                y="count",
                title="Contract Outcomes by Inferred Winner",
                labels={
                    "inferred_winner": "Inferred winner",
                    "count": "Contracts",
                },
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No inferred winner summary available.")

    c3, c4 = st.columns(2)

    with c3:
        overview_source_col, overview_source_label = get_underlying_plot_col(
            plot_sample,
            underlying_source_choice,
        )

        if overview_source_col is not None:
            overview_underlying = (
                plot_sample.sort_values("second")
                .groupby("second", as_index=False)[overview_source_col]
                .mean()
            )

            fig = px.line(
                overview_underlying,
                x="second",
                y=overview_source_col,
                title=f"BTC Underlying / Futures Value ({overview_source_label})",
                labels={
                    "second": "Time",
                    overview_source_col: "Underlying value",
                },
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        elif not futures_15m.empty and "futures_close" in futures_15m.columns:
            fig = px.line(
                futures_15m.sort_values("window_start"),
                x="window_start",
                y="futures_close",
                color="symbol" if "symbol" in futures_15m.columns else None,
                title="BTC Futures 15-Minute Close Price",
                labels={
                    "window_start": "Time",
                    "futures_close": "Futures close",
                    "symbol": "Symbol",
                },
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No futures or underlying path available for this week.")

    with c4:
        if not decision_times.empty and "minutes_before_close_at_decision" in decision_times.columns:
            valid_decisions = decision_times.copy()

            if "decision_found" in valid_decisions.columns:
                valid_decisions = valid_decisions[valid_decisions["decision_found"] == True]

            valid_decisions = valid_decisions.dropna(subset=["minutes_before_close_at_decision"])

            if not valid_decisions.empty:
                fig = px.histogram(
                    valid_decisions,
                    x="minutes_before_close_at_decision",
                    nbins=40,
                    color="inferred_winner" if "inferred_winner" in valid_decisions.columns else None,
                    title="Decision Time Distribution",
                    labels={
                        "minutes_before_close_at_decision": "Minutes before close",
                        "count": "Contracts",
                        "inferred_winner": "Winner",
                    },
                )
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid decision-time rows for this week.")
        else:
            st.info("No decision-time file found for this week.")


# ============================================================
# Tab 2: Contract Evolution
# ============================================================

with tabs[1]:
    st.header("Price Behavior / Contract Evolution")

    st.markdown(
        """
This view follows one selected contract. Instead of showing two similar price-path charts,
the first section shows the distribution of short-horizon contract repricing moves. The combined
path chart below still shows the time-series relationship between the contract and the futures-side series.
"""
    )

    if not selected_ticker or contract_evolution.empty:
        st.warning("No contract evolution data available.")
    else:
        one = contract_evolution[contract_evolution["ticker"] == selected_ticker].copy()

        if one.empty:
            st.warning("No rows found for the selected contract.")
        else:
            one = one.sort_values("second")

            if "underlying_trade_price" in one.columns:
                one["underlying_trade_price"] = one["underlying_trade_price"].ffill()

            underlying_col, underlying_label = get_underlying_plot_col(
                one,
                underlying_source_choice,
            )

            decision_marker_time = get_selected_contract_decision_time(
                selected_ticker=selected_ticker,
                decision_times=decision_times,
                contract_df=one,
            )

            st.caption(f"Selected contract: `{selected_ticker}`")

            meta_cols = safe_cols(
                one,
                [
                    "ticker",
                    "contract_open_time",
                    "contract_close_time",
                    "selected_futures_symbol",
                    "futures_symbol",
                    "inferred_winner",
                    "outcome_yes",
                    "data_source_period",
                ],
            )

            st.dataframe(one[meta_cols].head(1), use_container_width=True)

            if decision_marker_time is not None:
                st.caption(
                    f"Decision marker matched from `contract_decision_times.parquet`: "
                    f"{decision_marker_time}."
                )
            else:
                st.caption("No matched decision marker found for this selected contract.")

            if underlying_col is not None:
                st.caption(f"Underlying source in this view: **{underlying_label}**.")

            yes_col = "yes_price_cents_display" if "yes_price_cents_display" in one.columns else "yes_price_cents"
            no_col = "no_price_cents_display" if "no_price_cents_display" in one.columns else "no_price_cents"

            # ------------------------------------------------------------
            # 1. Contract price-move distribution
            # ------------------------------------------------------------
            st.subheader("Contract Price Move Distribution")

            move_df = one.copy()

            if yes_col in move_df.columns:
                move_df[yes_col] = pd.to_numeric(move_df[yes_col], errors="coerce")
                move_df["yes_price_change_1s_display_local"] = move_df[yes_col].diff()
                move_df["yes_price_change_5s_display_local"] = move_df[yes_col] - move_df[yes_col].shift(5)
                move_df["abs_yes_price_change_5s_display_local"] = move_df["yes_price_change_5s_display_local"].abs()

                move_df_clean = move_df.dropna(
                    subset=["yes_price_change_1s_display_local", "yes_price_change_5s_display_local"]
                ).copy()

                if not move_df_clean.empty:
                    c_move1, c_move2 = st.columns(2)

                    with c_move1:
                        fig_move_hist = px.histogram(
                            move_df_clean,
                            x="yes_price_change_5s_display_local",
                            nbins=60,
                            title="Distribution of 5-Second YES Price Moves",
                            labels={
                                "yes_price_change_5s_display_local": "5-second YES price change (cents)",
                                "count": "Seconds",
                            },
                        )

                        fig_move_hist.add_vline(x=0, line_dash="dash")

                        fig_move_hist.update_layout(
                            height=390,
                            bargap=0.05,
                        )

                        st.plotly_chart(fig_move_hist, use_container_width=True)

                    with c_move2:
                        if "time_to_close_bucket" in move_df_clean.columns:
                            box_df = move_df_clean.copy()
                            box_df = order_time_buckets(box_df)

                            fig_move_box = px.box(
                                box_df,
                                x="time_to_close_bucket",
                                y="abs_yes_price_change_5s_display_local",
                                title="Absolute 5-Second YES Move by Time-to-Close Bucket",
                                labels={
                                    "time_to_close_bucket": "Time to close",
                                    "abs_yes_price_change_5s_display_local": "Absolute 5-second YES move (cents)",
                                },
                            )

                            fig_move_box.update_layout(height=390)

                            st.plotly_chart(fig_move_box, use_container_width=True)
                        else:
                            fig_abs_hist = px.histogram(
                                move_df_clean,
                                x="abs_yes_price_change_5s_display_local",
                                nbins=60,
                                title="Distribution of Absolute 5-Second YES Moves",
                                labels={
                                    "abs_yes_price_change_5s_display_local": "Absolute 5-second YES move (cents)",
                                    "count": "Seconds",
                                },
                            )

                            fig_abs_hist.update_layout(height=390)

                            st.plotly_chart(fig_abs_hist, use_container_width=True)

                    big_move_threshold = move_df_clean["abs_yes_price_change_5s_display_local"].quantile(0.95)

                    big_moves = move_df_clean[
                        move_df_clean["abs_yes_price_change_5s_display_local"] >= big_move_threshold
                    ].copy()

                    st.caption(
                        f"The left chart shows the distribution of 5-second YES price changes for the selected contract. "
                        f"The right chart shows whether larger repricing moves cluster in certain time-to-close buckets. "
                        f"For this selected contract, the 95th percentile absolute 5-second move is about "
                        f"{big_move_threshold:.2f} cents."
                    )

                    with st.expander("Largest selected-contract price moves"):
                        table_cols = safe_cols(
                            big_moves,
                            [
                                "second",
                                "yes_price_change_5s_display_local",
                                "abs_yes_price_change_5s_display_local",
                                yes_col,
                                no_col,
                                "time_to_close_bucket",
                                "moneyness_bucket",
                                "z_bps",
                                "abs_z_bps",
                            ],
                        )

                        st.dataframe(
                            big_moves[table_cols]
                            .sort_values("abs_yes_price_change_5s_display_local", ascending=False)
                            .head(25),
                            use_container_width=True,
                        )
                else:
                    st.info("Not enough valid price observations to compute selected-contract price-move distributions.")
            else:
                st.info("No YES price column found for the selected contract.")

            # ------------------------------------------------------------
            # 2. Combined contract price + underlying chart
            # ------------------------------------------------------------
            st.subheader("Underlying vs Contract Price")

            fig_combined = go.Figure()

            if yes_col in one.columns:
                fig_combined.add_trace(
                    go.Scatter(
                        x=one["second"],
                        y=one[yes_col],
                        mode="lines",
                        name="YES price",
                        yaxis="y1",
                        line=dict(width=2),
                    )
                )

            if no_col in one.columns:
                fig_combined.add_trace(
                    go.Scatter(
                        x=one["second"],
                        y=one[no_col],
                        mode="lines",
                        name="NO price",
                        yaxis="y1",
                        line=dict(width=2),
                    )
                )

            if underlying_col is not None:
                fig_combined.add_trace(
                    go.Scatter(
                        x=one["second"],
                        y=one[underlying_col],
                        mode="lines",
                        name=underlying_label,
                        yaxis="y2",
                        line=dict(width=2),
                    )
                )

                start_underlying = None

                if underlying_source_choice == "Modeling underlying_value":
                    if "reference_underlying" in one.columns and one["reference_underlying"].notna().any():
                        start_underlying = float(one["reference_underlying"].dropna().iloc[0])

                if start_underlying is None and one[underlying_col].notna().any():
                    start_underlying = float(one[underlying_col].dropna().iloc[0])

                if start_underlying is not None:
                    fig_combined.add_shape(
                        type="line",
                        x0=one["second"].min(),
                        x1=one["second"].max(),
                        y0=start_underlying,
                        y1=start_underlying,
                        xref="x",
                        yref="y2",
                        line=dict(dash="dash", width=1),
                    )

                    fig_combined.add_annotation(
                        x=one["second"].max(),
                        y=start_underlying,
                        xref="x",
                        yref="y2",
                        text="Start/reference underlying",
                        showarrow=False,
                        xanchor="right",
                        yanchor="bottom",
                    )

            fig_combined = add_decision_close_shapes(
                fig_combined,
                one,
                show_decision=True,
                decision_marker_time=decision_marker_time,
            )

            fig_combined.update_layout(
                title=f"Kalshi YES/NO Prices vs {underlying_label}",
                xaxis=dict(title="Time"),
                yaxis=dict(
                    title="Contract price (cents)",
                    side="left",
                    range=[0, 100],
                ),
                yaxis2=dict(
                    title="Underlying / futures value",
                    overlaying="y",
                    side="right",
                ),
                height=520,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
            )

            st.plotly_chart(fig_combined, use_container_width=True)

            # ------------------------------------------------------------
            # 3. Threshold-distance / moneyness chart
            # ------------------------------------------------------------
            money_col, money_label, band_width, band_label = get_moneyness_plot_spec(one)

            if money_col is not None:
                money_plot = one.copy()
                money_plot[money_col] = pd.to_numeric(money_plot[money_col], errors="coerce")

                fig_money = go.Figure()

                fig_money.add_trace(
                    go.Scatter(
                        x=money_plot["second"],
                        y=money_plot[money_col],
                        mode="lines",
                        name=money_label,
                        line=dict(width=2),
                    )
                )

                x_min = money_plot["second"].min()
                x_max = money_plot["second"].max()

                if money_col == "abs_z_bps":
                    fig_money.add_shape(
                        type="rect",
                        x0=x_min,
                        x1=x_max,
                        y0=0,
                        y1=band_width,
                        xref="x",
                        yref="y",
                        fillcolor="gray",
                        opacity=0.12,
                        line_width=0,
                        layer="below",
                    )

                    fig_money.add_hline(
                        y=band_width,
                        line_dash="dot",
                        line_width=1,
                        annotation_text=band_label,
                        annotation_position="top left",
                    )
                else:
                    fig_money.add_shape(
                        type="rect",
                        x0=x_min,
                        x1=x_max,
                        y0=-band_width,
                        y1=band_width,
                        xref="x",
                        yref="y",
                        fillcolor="gray",
                        opacity=0.12,
                        line_width=0,
                        layer="below",
                    )

                    fig_money.add_hline(
                        y=band_width,
                        line_dash="dot",
                        line_width=1,
                        annotation_text=band_label,
                        annotation_position="top left",
                    )

                    fig_money.add_hline(
                        y=-band_width,
                        line_dash="dot",
                        line_width=1,
                    )

                fig_money.add_hline(
                    y=0,
                    line_dash="dash",
                    line_width=2,
                    annotation_text="Threshold",
                    annotation_position="bottom left",
                )

                fig_money = add_decision_close_shapes(
                    fig_money,
                    one,
                    show_decision=True,
                    decision_marker_time=decision_marker_time,
                )

                fig_money.update_layout(
                    title="BTC Futures Distance from Contract Threshold",
                    xaxis_title="Time",
                    yaxis_title=money_label,
                    height=390,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                    ),
                )

                st.plotly_chart(fig_money, use_container_width=True)

                st.caption(
                    "This chart shows how close the futures-side price is to the contract threshold. "
                    "Values near zero mean the contract is near the boundary, where small BTC moves can create large YES/NO price changes. "
                    "The shaded band marks the near-threshold zone."
                )
            else:
                st.info("No threshold-distance or moneyness column found for this selected contract.")


# ============================================================
# Tab 3: Futures Microstructure
# ============================================================

with tabs[2]:
    st.header("Futures Microstructure")

    st.markdown(
        """
This page summarizes the futures-side conditions inside the selected week using the available weekly dashboard files.
"""
    )

    micro_base = plot_sample.copy()

    if micro_base.empty:
        st.warning("No weekly plot sample available for futures microstructure charts.")
    else:
        micro, micro_label = make_microstructure_frame(micro_base, underlying_source_choice)

        if micro.empty:
            st.warning("No usable microstructure fields found in the weekly files.")
        else:
            m1, m2, m3, m4 = st.columns(4)

            if "trade_count" in micro.columns:
                m1.metric("Total trade count", fmt_int(micro["trade_count"].sum()))
            else:
                m1.metric("Total trade count", "N/A")

            volume_col = "trade_volume" if "trade_volume" in micro.columns else "trade_size"

            if volume_col in micro.columns:
                m2.metric("Total trade volume/size", fmt_float(micro[volume_col].sum(), 2))
            else:
                m2.metric("Total trade volume/size", "N/A")

            if "spread_bps" in micro.columns:
                m3.metric("Avg spread bps", fmt_float(micro["spread_bps"].mean(), 3))
            elif "spread" in micro.columns:
                m3.metric("Avg spread", fmt_float(micro["spread"].mean(), 3))
            elif "quote_spread" in micro.columns:
                m3.metric("Avg quote spread", fmt_float(micro["quote_spread"].mean(), 3))
            else:
                m3.metric("Avg spread", "N/A")

            if "local_vol_60s_bps" in micro.columns:
                m4.metric("Avg local vol 60s", fmt_float(micro["local_vol_60s_bps"].mean(), 4))
            else:
                m4.metric("Avg local vol 60s", "N/A")

            st.caption(f"Underlying source used for local volatility: **{micro_label}**.")

            c1, c2 = st.columns(2)

            with c1:
                underlying_col, _ = get_underlying_plot_col(micro, underlying_source_choice)

                if underlying_col is None:
                    possible_cols = [
                        c for c in ["underlying_trade_price", "underlying_quote_mid", "underlying_modeling", "underlying_value"]
                        if c in micro.columns
                    ]
                    underlying_col = possible_cols[0] if possible_cols else None

                if underlying_col is not None and underlying_col in micro.columns:
                    fig = px.line(
                        micro,
                        x="second",
                        y=underlying_col,
                        title=f"Selected Futures Underlying Path ({micro_label})",
                        labels={
                            "second": "Time",
                            underlying_col: "Underlying value",
                        },
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No underlying path available.")

            with c2:
                spread_col = None

                if "spread_bps" in micro.columns:
                    spread_col = "spread_bps"
                    spread_label = "Spread bps"
                elif "spread" in micro.columns:
                    spread_col = "spread"
                    spread_label = "Spread"
                elif "quote_spread" in micro.columns:
                    spread_col = "quote_spread"
                    spread_label = "Quote spread"

                if spread_col is not None:
                    fig = px.line(
                        micro,
                        x="second",
                        y=spread_col,
                        title=f"Futures {spread_label} Over Time",
                        labels={
                            "second": "Time",
                            spread_col: spread_label,
                        },
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No spread column available.")

            c3, c4 = st.columns(2)

            with c3:
                y_cols = [c for c in ["trade_count", "trade_size", "trade_volume"] if c in micro.columns]

                if y_cols:
                    fig = go.Figure()

                    if "trade_count" in y_cols:
                        fig.add_trace(
                            go.Scatter(
                                x=micro["second"],
                                y=micro["trade_count"],
                                mode="lines",
                                name="Trade count",
                                yaxis="y1",
                            )
                        )

                    if volume_col in y_cols:
                        fig.add_trace(
                            go.Scatter(
                                x=micro["second"],
                                y=micro[volume_col],
                                mode="lines",
                                name=volume_col,
                                yaxis="y2",
                            )
                        )

                    fig.update_layout(
                        title="Futures Trade Activity",
                        xaxis=dict(title="Time"),
                        yaxis=dict(title="Trade count", side="left"),
                        yaxis2=dict(
                            title="Trade volume / size",
                            overlaying="y",
                            side="right",
                        ),
                        height=400,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                        ),
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No trade count or trade size columns available.")

            with c4:
                if "local_vol_60s_bps" in micro.columns:
                    fig = px.line(
                        micro,
                        x="second",
                        y="local_vol_60s_bps",
                        title="Local 60-Second Futures Volatility",
                        labels={
                            "second": "Time",
                            "local_vol_60s_bps": "Rolling 60s std of 1s log returns, bps",
                        },
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No local volatility could be computed.")

            st.subheader("Spread vs Local Volatility")

            if "local_vol_60s_bps" in micro.columns:
                spread_col = None

                if "spread_bps" in micro.columns:
                    spread_col = "spread_bps"
                elif "spread" in micro.columns:
                    spread_col = "spread"
                elif "quote_spread" in micro.columns:
                    spread_col = "quote_spread"

                if spread_col is not None:
                    scatter = micro.dropna(subset=[spread_col, "local_vol_60s_bps"]).copy()

                    if not scatter.empty:
                        fig = px.scatter(
                            scatter,
                            x="local_vol_60s_bps",
                            y=spread_col,
                            title="Spread vs Local Volatility",
                            labels={
                                "local_vol_60s_bps": "Local 60s volatility, bps",
                                spread_col: "Spread",
                            },
                        )
                        fig.update_layout(height=450)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No valid spread/volatility rows.")
                else:
                    st.info("No spread column available for spread/volatility scatter.")


# ============================================================
# Tab 4: Decision Time
# ============================================================

with tabs[3]:
    st.header("Decision Time")

    st.markdown(
        """
Decision time is kept as a secondary diagnostic view. It uses the precomputed decision-time file,
while the Contract Evolution page matches the selected ticker to that file before drawing its marker.
"""
    )

    if decision_times.empty:
        st.warning("No decision-time file available for this week.")
    else:
        valid_decisions = decision_times.copy()

        if "decision_found" in valid_decisions.columns:
            valid_decisions = valid_decisions[valid_decisions["decision_found"] == True]

        c1, c2 = st.columns(2)

        with c1:
            if "minutes_before_close_at_decision" in valid_decisions.columns and not valid_decisions.empty:
                fig = px.histogram(
                    valid_decisions,
                    x="minutes_before_close_at_decision",
                    color="inferred_winner" if "inferred_winner" in valid_decisions.columns else None,
                    nbins=40,
                    title="Decision Time Distribution",
                    labels={
                        "minutes_before_close_at_decision": "Minutes before close",
                        "count": "Contracts",
                        "inferred_winner": "Winner",
                    },
                )
                fig.update_layout(height=430)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid decision-time rows.")

        with c2:
            if not valid_decisions.empty and "inferred_winner" in valid_decisions.columns:
                win_counts = valid_decisions["inferred_winner"].fillna("Unknown").value_counts().reset_index()
                win_counts.columns = ["inferred_winner", "count"]

                fig = px.pie(
                    win_counts,
                    names="inferred_winner",
                    values="count",
                    title="Decision Contracts by Inferred Winner",
                )
                fig.update_layout(height=430)
                st.plotly_chart(fig, use_container_width=True)

        if not valid_decisions.empty:
            st.subheader("Decision-Time Summary")

            summary = valid_decisions["minutes_before_close_at_decision"].describe()
            st.dataframe(summary.to_frame("minutes_before_close_at_decision"), use_container_width=True)

            table_cols = safe_cols(
                valid_decisions,
                [
                    "ticker",
                    "contract_open_time",
                    "contract_close_time",
                    "inferred_winner",
                    "decision_time",
                    "minutes_before_close_at_decision",
                    "data_source_period",
                ],
            )
            st.dataframe(
                valid_decisions[table_cols].sort_values("minutes_before_close_at_decision", ascending=False).head(200),
                use_container_width=True,
            )


# ============================================================
# Tab 5: Threshold Risk
# ============================================================

with tabs[4]:
    st.header("Threshold / Pin-Like Risk")

    st.markdown(
        """
When BTC futures are close to the contract threshold near settlement, small futures moves can cause large
contract price changes. The heatmap summarizes where contract prices move the most.
"""
    )

    c1, c2 = st.columns(2)

    with c1:
        heat_col = (
            "mean_abs_yes_price_change_5s_display"
            if "mean_abs_yes_price_change_5s_display" in threshold_heatmap.columns
            else "mean_abs_yes_price_change_5s"
        )

        if not threshold_heatmap.empty and heat_col in threshold_heatmap.columns:
            heat = threshold_heatmap.copy()
            heat = order_time_buckets(heat)
            heat = order_moneyness_buckets(heat)

            pivot = heat.pivot_table(
                index="moneyness_bucket",
                columns="time_to_close_bucket",
                values=heat_col,
                aggfunc="mean",
            )

            fig = px.imshow(
                pivot,
                aspect="auto",
                title="Threshold Heatmap: Avg Absolute 5s YES Price Move",
                labels={
                    "x": "Time to close",
                    "y": "Moneyness bucket",
                    "color": "Avg abs 5s price move",
                },
            )
            fig.update_layout(height=460)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No threshold heatmap file available.")

    with c2:
        if not plot_sample_filtered.empty and "abs_z_bps" in plot_sample_filtered.columns:
            fig = px.histogram(
                plot_sample_filtered,
                x="abs_z_bps",
                nbins=60,
                color="time_to_close_bucket" if "time_to_close_bucket" in plot_sample_filtered.columns else None,
                title="Distribution of Absolute Distance to Threshold",
                labels={
                    "abs_z_bps": "Absolute moneyness / threshold distance (bps)",
                    "count": "Rows",
                },
            )
            fig.update_layout(height=460)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No plot sample rows available after filters.")

    st.subheader("Threshold Risk Scatter")

    scatter_y = (
        "abs_yes_price_change_5s_display"
        if "abs_yes_price_change_5s_display" in plot_sample_filtered.columns
        else "abs_yes_price_change_5s"
    )

    if not plot_sample_filtered.empty and scatter_y in plot_sample_filtered.columns:
        scatter = plot_sample_filtered.dropna(subset=["abs_z_bps", scatter_y]).copy()

        if scatter.empty:
            st.warning("No valid scatter rows.")
        else:
            fig = px.scatter(
                scatter,
                x="abs_z_bps",
                y=scatter_y,
                color="time_to_close_bucket" if "time_to_close_bucket" in scatter.columns else None,
                hover_data=safe_cols(scatter, ["ticker", "second", "minutes_to_close", "yes_price_cents_display"]),
                title="Distance to Threshold vs 5s Contract Price Response",
                labels={
                    "abs_z_bps": "Absolute distance to threshold (bps)",
                    scatter_y: "Absolute 5s YES price move (cents)",
                    "time_to_close_bucket": "Time to close",
                },
            )
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Tab 6: Sensitivity
# ============================================================

with tabs[5]:
    st.header("Sensitivity Explorer")

    st.markdown(
        """
These charts show empirical sensitivity, not a formal option Greek. They compare short-horizon BTC futures
moves against short-horizon Kalshi YES price moves.
"""
    )

    c1, c2 = st.columns(2)

    scatter_y = (
        "yes_price_change_5s_display"
        if "yes_price_change_5s_display" in plot_sample_filtered.columns
        else "yes_price_change_5s"
    )

    scatter_x_source, scatter_x_label = get_underlying_plot_col(
        plot_sample_filtered,
        underlying_source_choice,
    )

    with c1:
        if not plot_sample_filtered.empty and scatter_y in plot_sample_filtered.columns:
            scatter = plot_sample_filtered.copy()

            if "underlying_change_5s" in scatter.columns and underlying_source_choice == "Modeling underlying_value":
                scatter_x = "underlying_change_5s"
                x_label = "5s modeling underlying/futures change ($)"
            elif scatter_x_source is not None:
                scatter = scatter.sort_values(["ticker", "second"])
                scatter["selected_underlying_change_5s"] = (
                    scatter[scatter_x_source]
                    - scatter.groupby("ticker")[scatter_x_source].shift(5)
                )
                scatter_x = "selected_underlying_change_5s"
                x_label = f"5s {scatter_x_label} change ($)"
            else:
                scatter_x = None
                x_label = "5s underlying/futures change ($)"

            if scatter_x is None:
                st.warning("No usable underlying column available for the sensitivity scatter.")
            else:
                scatter = scatter.dropna(subset=[scatter_x, scatter_y]).copy()

                if scatter.empty:
                    st.warning("No valid 5-second sensitivity rows.")
                else:
                    fig = px.scatter(
                        scatter,
                        x=scatter_x,
                        y=scatter_y,
                        color="time_to_close_bucket" if "time_to_close_bucket" in scatter.columns else None,
                        hover_data=safe_cols(scatter, ["ticker", "second", "abs_z_bps", "minutes_to_close"]),
                        title="5s Underlying Move vs 5s Contract Price Move",
                        labels={
                            scatter_x: x_label,
                            scatter_y: "5s YES price change (cents)",
                            "time_to_close_bucket": "Time to close",
                        },
                    )
                    fig.add_hline(y=0, line_dash="dash")
                    fig.add_vline(x=0, line_dash="dash")
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No plot sample available.")

    with c2:
        sens_y = (
            "mean_abs_yes_price_change_5s_display"
            if "mean_abs_yes_price_change_5s_display" in sensitivity_bins.columns
            else "mean_abs_yes_price_change_5s"
        )

        if not sensitivity_bins.empty and sens_y in sensitivity_bins.columns:
            sens = sensitivity_bins.copy()
            sens = order_time_buckets(sens)
            sens = order_moneyness_buckets(sens)

            fig = px.line(
                sens,
                x="moneyness_bucket",
                y=sens_y,
                color="time_to_close_bucket",
                markers=True,
                title="Binned Sensitivity Curve",
                labels={
                    "moneyness_bucket": "Moneyness bucket",
                    sens_y: "Avg abs 5s YES price move (cents)",
                    "time_to_close_bucket": "Time to close",
                },
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No sensitivity bins file available.")

    st.subheader("Sensitivity Bin Table")

    if not sensitivity_bins.empty:
        st.dataframe(sensitivity_bins, use_container_width=True)


# ============================================================
# Tab 7: Data Preview
# ============================================================

with tabs[6]:
    st.header("Data Preview")

    st.markdown(
        """
These previews are for auditability. The dashboard loads prepared weekly files instead of processing
the full raw datasets interactively.
"""
    )

    with st.expander("Underlying source check", expanded=True):
        check_rows = []

        for name, df in [
            ("contract_evolution_1s", contract_evolution),
            ("contract_plot_sample", plot_sample),
        ]:
            if df.empty:
                check_rows.append(
                    {
                        "file": name,
                        "has_underlying_trade_price": False,
                        "non_null_underlying_trade_price": 0,
                        "has_quote_mid": False,
                        "non_null_quote_mid": 0,
                        "has_modeling_underlying": False,
                        "non_null_modeling_underlying": 0,
                    }
                )
            else:
                check_rows.append(
                    {
                        "file": name,
                        "has_underlying_trade_price": "underlying_trade_price" in df.columns,
                        "non_null_underlying_trade_price": int(df["underlying_trade_price"].notna().sum()) if "underlying_trade_price" in df.columns else 0,
                        "has_quote_mid": "underlying_quote_mid" in df.columns,
                        "non_null_quote_mid": int(df["underlying_quote_mid"].notna().sum()) if "underlying_quote_mid" in df.columns else 0,
                        "has_modeling_underlying": "underlying_modeling" in df.columns,
                        "non_null_modeling_underlying": int(df["underlying_modeling"].notna().sum()) if "underlying_modeling" in df.columns else 0,
                    }
                )

        st.dataframe(pd.DataFrame(check_rows), use_container_width=True)

    with st.expander("Contract behavior summary metrics"):
        st.dataframe(summary_contract, use_container_width=True)

    with st.expander("Market summary metrics"):
        st.dataframe(summary_market, use_container_width=True)

    with st.expander("Contract summary"):
        st.dataframe(contract_summary.head(200), use_container_width=True)

    with st.expander("Contract decision times"):
        st.dataframe(decision_times.head(200), use_container_width=True)

    with st.expander("Sensitivity bins"):
        st.dataframe(sensitivity_bins, use_container_width=True)

    with st.expander("Threshold heatmap"):
        st.dataframe(threshold_heatmap, use_container_width=True)

    with st.expander("Contract plot sample"):
        st.dataframe(plot_sample.head(200), use_container_width=True)

    with st.expander("Selected contract evolution"):
        if selected_ticker and not contract_evolution.empty:
            preview_cols = safe_cols(
                contract_evolution,
                [
                    "ticker",
                    "second",
                    "underlying_value",
                    "underlying_modeling",
                    "underlying_trade_price",
                    "underlying_quote_mid",
                    "best_bid",
                    "best_ask",
                    "spread",
                    "spread_bps",
                    "quote_spread",
                    "trade_count",
                    "trade_size",
                    "trade_volume",
                    "yes_price_cents",
                    "no_price_cents",
                    "yes_price_cents_display",
                    "no_price_cents_display",
                    "z_bps",
                    "abs_z_bps",
                    "time_to_close_bucket",
                    "moneyness_bucket",
                ],
            )

            st.dataframe(
                contract_evolution[contract_evolution["ticker"] == selected_ticker][preview_cols].head(300),
                use_container_width=True,
            )
        else:
            st.info("No selected contract evolution data.")