import streamlit as st
import plotly.graph_objects as go
import datetime as dt
from trading_helpers import run_all, plot_metric_bar
import pandas as pd

st.set_page_config(
    page_title="Technical Analysis Trading Dashboard",
    layout="wide",      
)


def main():

    # ========== SIDEBAR: DATA ==========
    st.sidebar.header("Data")

    # Ticker for strategies
    ticker = st.sidebar.text_input("Ticker", value="AAPL")

    # Default: last 5 years
    today = dt.date.today()
    default_start = today - dt.timedelta(days=5 * 365)

    col_start, col_end = st.sidebar.columns(2)
    start_date = col_start.date_input("Start", value=default_start)
    end_date   = col_end.date_input("End", value=today)

    interval = st.sidebar.selectbox(
        "Interval",
        ["1d", "1wk", "1mo"],
        index=0,
        help="Yahoo Finance intervals (daily, weekly, monthly).",
    )

    # Convert to strings for yfinance / run_all
    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")

    # ========== SIDEBAR: BENCHMARK ==========
    st.sidebar.header("Benchmark")

    use_custom_bench = st.sidebar.checkbox(
        "Use custom benchmark",
        value=False,
        help="If unchecked, the default TD Nasdaq fund (TDB908 CSV) is used. Data limited from 2015-Nov-27 to 2025-Nov-27",
    )

    if use_custom_bench:
        custom_bench = st.sidebar.text_input(
            "Custom benchmark ticker (optional)",
            value="",
            placeholder="e.g. QQQ, ^NDX, VOO",
        )

        if custom_bench.strip():
            benchmark_mode   = "yf_ticker"
            benchmark_ticker = custom_bench.strip().upper()
        else:
            st.sidebar.info("No ticker entered – using TD Nasdaq fund (TDB908) instead.")
            benchmark_mode   = "td_fund"
            benchmark_ticker = "TDB908"
    else:
        benchmark_mode   = "td_fund"
        benchmark_ticker = "TDB908"


    # ========== LOAD DATA & RUN STRATEGIES ==========
    # (You can wrap this in @st.cache_data once everything is stable)

    def get_results(ticker, start, end, interval, benchmark_mode, benchmark_ticker):
        return run_all(ticker=ticker, start=start, end=end, interval=interval, benchmark_mode=benchmark_mode, benchmark_ticker=benchmark_ticker)

    aapl_clean, bench_df, bench_name, super_results, psar_results, donch_results, metrics_df = get_results(
    ticker, start_str, end_str, interval, benchmark_mode, benchmark_ticker
)
    if benchmark_mode == "td_fund":
        title_text = "Technical Analysis Trading Strategies vs TDB908 Actively Managed Mutual Fund"
    else:
        # bench_name will be e.g. "Buy & Hold QQQ"
        title_text = f"Technical Analysis Trading Strategies vs {bench_name}"

    st.title(title_text)
    start_ts = pd.to_datetime(start_str)
    end_ts   = pd.to_datetime(end_str)

    def clip_to_range(df):
        if "Date" not in df.columns:
            return df
        mask = (df["Date"] >= start_ts) & (df["Date"] <= end_ts)
        return df.loc[mask].reset_index(drop=True)

    super_results = clip_to_range(super_results)
    psar_results  = clip_to_range(psar_results)
    donch_results = clip_to_range(donch_results)
    bench_df      = clip_to_range(bench_df)
    aapl_clean    = clip_to_range(aapl_clean)


    # ========== VIEW CONTROLS ==========
    st.sidebar.header("View")

    view_mode = st.sidebar.radio(
        "What do you want to see?",
        ["Equity curves", "Trade signals", "Metrics"],
        index=0,
    )

    def add_trade_flags(df):
        """
        Given a DataFrame with 'position' (0/1) and 'Date', 'Close',
        add boolean columns 'entry' (buy) and 'exit' (sell).
        """
        out = df.copy()
        out["prev_pos"] = out["position"].shift(1).fillna(0)
        out["entry"] = (out["position"] == 1) & (out["prev_pos"] == 0)
        out["exit"] = (out["position"] == 0) & (out["prev_pos"] == 1)
        return out

    # ======================================================================
    #  EQUITY CURVES
    # ======================================================================
    if view_mode == "Equity curves":
        curve_view = st.sidebar.selectbox(
            "Curve view",
            ["Overlay (all strategies + benchmarks)", "Single strategy vs benchmarks"],
            index=0,
        )

        # ---------- Overlay: All equity curves ----------
        if curve_view.startswith("Overlay"):
            st.subheader("All Equity Curves")

            fig = go.Figure()

            # Donchian Breakout
            fig.add_trace(
                go.Scatter(
                    x=donch_results["Date"],
                    y=donch_results["strategy_equity"],
                    mode="lines",
                    name="Donchian Breakout",
                )
            )

            # Supertrend
            fig.add_trace(
                go.Scatter(
                    x=super_results["Date"],
                    y=super_results["strategy_equity"],
                    mode="lines",
                    name="Supertrend",
                )
            )

            # Parabolic SAR
            fig.add_trace(
                go.Scatter(
                    x=psar_results["Date"],
                    y=psar_results["strategy_equity"],
                    mode="lines",
                    name="Parabolic SAR",
                )
            )

            # Buy & Hold baseline (same dates as strategies)
            fig.add_trace(
                go.Scatter(
                    x=super_results["Date"],
                    y=super_results["buyhold_equity"],
                    mode="lines",
                    name=f"Buy & Hold {ticker}"
                )
            )

            if "bench_equity" in bench_df.columns:
                bench_equity = bench_df["bench_equity"].copy()
            else:
                # legacy TD fund CSV that still uses 'fund_equity'
                bench_equity = bench_df["fund_equity"].copy()

            # Rebase so first value = 1.0
            if len(bench_equity) > 0:
                bench_equity = bench_equity / bench_equity.iloc[0]

            fig.add_trace(
                go.Scatter(
                x=bench_df["Date"],
                y=bench_equity,
                mode="lines",
                name=bench_name,
        )
    )

            fig.update_layout(
                title=f"Equity Curves – {ticker} Strategies vs Benchmarks",
                xaxis_title="Date",
                yaxis_title="Portfolio Value (start = 1.0)",
                xaxis_rangeslider_visible=False,
                legend_title="Curve",
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ---------- Single strategy vs benchmarks ----------
        else:
            st.subheader("Single Strategy vs Buy & Hold & TDB908")

            choice = st.sidebar.selectbox(
                "Strategy:",
                ["Supertrend", "Parabolic SAR", "Donchian Breakout"],
            )

            if choice == "Supertrend":
                df = super_results.copy()
                col_label = "Supertrend"
            elif choice == "Parabolic SAR":
                df = psar_results.copy()
                col_label = "Parabolic SAR"
            else:
                df = donch_results.copy()
                col_label = "Donchian Breakout"

            fig = go.Figure()

            # Selected strategy
            fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=df["strategy_equity"],
                    mode="lines",
                    name=col_label
                )
            )

            # Buy & Hold
            fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=df["buyhold_equity"],
                    mode="lines",
                    name=f"Buy & Hold {ticker}"
                
                )
            )

            # TDB908 Fund
            if "bench_equity" in bench_df.columns:
                bench_equity = bench_df["bench_equity"].copy()
            else:
                bench_equity = bench_df["fund_equity"].copy()

            if len(bench_equity) > 0:
                bench_equity = bench_equity / bench_equity.iloc[0]


            fig.add_trace(
                go.Scatter(
                    x=bench_df["Date"],
                    y=bench_equity,
                    mode="lines",
                    name=bench_name,
                )
            )

            fig.update_layout(
                title=f"Equity Curve – {col_label} vs Benchmarks",
                xaxis_title="Date",
                yaxis_title="Portfolio Value (start = 1.0)",
                xaxis_rangeslider_visible=False,
                legend_title="Curve",
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ======================================================================
    #  METRICS
    # ======================================================================
    elif view_mode == "Metrics":
        st.sidebar.subheader("Metric")

        metric_choice = st.sidebar.selectbox(
            "Choose metric to display",
            ["Total Return", "Annual Return", "Success Rate", "Sharpe Ratio", "Table"],
            index=0,
        )

        # ---------- Metrics table ----------
        if metric_choice == "Table":
            st.subheader("Performance Metrics Table")
            st.dataframe(
                metrics_df.style.format(
                    {
                        "Success Rate (%)": "{:.2f}",
                        "Total Return (%)": "{:.2f}",
                        "Annual Return (%)": "{:.2f}",
                        "Sharpe Ratio": "{:.2f}",
                    }
                )
            )

        # ---------- Metric bar charts ----------
        else:
            metric_map = {
                "Total Return": "Total Return (%)",
                "Annual Return": "Annual Return (%)",
                "Success Rate": "Success Rate (%)",
                "Sharpe Ratio": "Sharpe Ratio",
            }
            metric_col = metric_map[metric_choice]

            st.subheader(f"{metric_choice} – Comparison")

            # Use your helper to keep styling consistent
            fig = plot_metric_bar(
                metrics_df,
                metric_col=metric_col,
                title=f"{metric_choice} Comparison",
                is_percent=("%" in metric_col),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ======================================================================
    #  TRADE SIGNALS            
    # ======================================================================
    elif view_mode == "Trade signals":
        st.subheader("Trade Signals – Price with Buy/Sell Markers")

        # Choose which strategy's signals to view
        strat_name = st.sidebar.selectbox(
            "Strategy for signals",
            ["Donchian Breakout", "Supertrend", "Parabolic SAR"],
            index=0,
        )

        if strat_name == "Donchian Breakout":
            df = donch_results.copy()
        elif strat_name == "Supertrend":
            df = super_results.copy()
        else:
            df = psar_results.copy()

        # Ensure we have the columns we need
        if not {"Date", "Close", "position"}.issubset(df.columns):
            st.error(
                f"{strat_name} results do not contain 'Date', 'Close', and 'position' columns."
            )
        else:
            df_sig = add_trade_flags(df)

            fig = go.Figure()

            # 1) Price line
            fig.add_trace(
                go.Scatter(
                    x=df_sig["Date"],
                    y=df_sig["Close"],
                    mode="lines",
                    name=f"{ticker} Close",
                )
            )

            # 2) Strategy-specific indicator overlay (if available)
            if strat_name == "Donchian Breakout":
                # Use whatever Donchian columns you have; adjust names if needed
                if "DC_high_20" in df_sig.columns and "DC_low_10" in df_sig.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df_sig["Date"],
                            y=df_sig["DC_high_20"],
                            mode="lines",
                            name="Donchian Upper",
                            line=dict(width=1),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=df_sig["Date"],
                            y=df_sig["DC_low_10"],
                            mode="lines",
                            name="Donchian Lower",
                            line=dict(width=1),
                        )
                    )
            elif strat_name == "Supertrend":
                if "SUPERT" in df_sig.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df_sig["Date"],
                            y=df_sig["SUPERT"],
                            mode="lines",
                            name="Supertrend Line",
                            line=dict(width=1),
                        )
                    )
            else:  # Parabolic SAR
                # We’ll assume a 'PSAR' column – adjust if your code uses a different name
                if "PSAR" in df_sig.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df_sig["Date"],
                            y=df_sig["PSAR"],
                            mode="markers",
                            name="Parabolic SAR",
                            marker=dict(size=5),
                        )
                    )

            # 3) BUY markers
            buys = df_sig["entry"]
            fig.add_trace(
                go.Scatter(
                    x=df_sig.loc[buys, "Date"],
                    y=df_sig.loc[buys, "Close"],
                    mode="markers",
                    name="Buy",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color="green",
                        line=dict(width=1, color="black"),
                    ),
                )
            )

            # 4) SELL markers
            sells = df_sig["exit"]
            fig.add_trace(
                go.Scatter(
                    x=df_sig.loc[sells, "Date"],
                    y=df_sig.loc[sells, "Close"],
                    mode="markers",
                    name="Sell",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color="red",
                        line=dict(width=1, color="black"),
                    ),
                )
            )

            fig.update_layout(
                title=f"{strat_name} – Trade Signals on {ticker}",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False,
                legend_title="Legend",
                height=600,
            )

            st.plotly_chart(fig, use_container_width=True)




if __name__ == "__main__":
    main()
