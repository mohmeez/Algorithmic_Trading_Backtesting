import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from trading_helpers import run_all, plot_metric_bar if you move that too
import pandas as pd

from trading_helpers import run_all, build_metrics_df  # we’ll use run_all for now

# Optional: bar-plot helper (you can also move this into trading_helpers if you want)
def plot_metric_bar(metrics_df, metric_col, title, is_percent=True):
    df_plot = metrics_df.sort_values(metric_col, ascending=False).copy()

    fig = px.bar(
        df_plot,
        x=metric_col,
        y="Strategy",
        orientation="h",
        text=df_plot[metric_col].round(2),
        title=title,
    )
    fig.update_traces(textposition="outside")
    max_val = df_plot[metric_col].max()
    fig.update_xaxes(range=[0, max_val * 1.15])
    fig.update_layout(
        xaxis_title=metric_col,
        yaxis_title="Strategy",
        height=400,
        showlegend=False,
    )
    if is_percent:
        fig.update_xaxes(ticksuffix="%")
    return fig


def main():
    st.title("AAPL Trading Strategies vs TD Mutual Fund")

    # Load + run everything via helper
    aapl_clean, fund, super_results, psar_results, donch_results, metrics_df = run_all()

    view = st.sidebar.radio(
        "View:",
        [
            "Overlay: All Equity Curves",
            "Single Strategy vs Benchmarks",
            "Metrics: Total Return",
            "Metrics: Annual Return",
            "Metrics: Success Rate",
            "Metrics: Sharpe Ratio",
            "Metrics: Table",
        ],
    )

    # ---------- Overlay ----------
    if view == "Overlay: All Equity Curves":
        st.subheader("All Equity Curves")

        # Build combined DF like in your notebook
        combined = donch_results[["Date", "strategy_equity", "buyhold_equity"]].rename(
            columns={
                "strategy_equity": "Donchian Breakout",
                "buyhold_equity": "Buy & Hold AAPL",
            }
        )
        combined = combined.merge(
            super_results[["Date", "strategy_equity"]].rename(columns={"strategy_equity": "Supertrend"}),
            on="Date", how="inner"
        )
        combined = combined.merge(
            psar_results[["Date", "strategy_equity"]].rename(columns={"strategy_equity": "Parabolic SAR"}),
            on="Date", how="inner"
        )
        combined = combined.merge(
            fund[["Date", "fund_equity"]].rename(columns={"fund_equity": "TDB908 Fund"}),
            on="Date", how="inner"
        )

        options = list(combined.columns[1:])  # all equity columns
        selected = st.sidebar.multiselect(
            "Select curves to show:",
            options,
            default=options,
        )

        fig = go.Figure()
        for name in selected:
            fig.add_trace(go.Scatter(
                x=combined["Date"],
                y=combined[name],
                mode="lines",
                name=name
            ))

        fig.update_layout(
            title="Equity Curves – Selected Strategies and Benchmarks",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (start = 1.0)",
            xaxis_rangeslider_visible=False,
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------- Single strategy vs benchmarks ----------
    elif view == "Single Strategy vs Benchmarks":
        st.subheader("Single Strategy vs Buy & Hold & TDB908")

        choice = st.sidebar.selectbox(
            "Strategy:",
            ["Supertrend", "Parabolic SAR", "Donchian Breakout"],
        )

        if choice == "Supertrend":
            df = super_results.copy()
            name = "Supertrend"
        elif choice == "Parabolic SAR":
            df = psar_results.copy()
            name = "Parabolic SAR"
        else:
            df = donch_results.copy()
            name = "Donchian Breakout"

        merged = df[["Date", "strategy_equity", "buyhold_equity"]].rename(
            columns={
                "strategy_equity": name,
                "buyhold_equity": "Buy & Hold AAPL",
            }
        )
        merged = merged.merge(
            fund[["Date", "fund_equity"]].rename(columns={"fund_equity": "TDB908 Fund"}),
            on="Date", how="inner"
        )

        fig = go.Figure()
        for col in [name, "Buy & Hold AAPL", "TDB908 Fund"]:
            fig.add_trace(go.Scatter(
                x=merged["Date"],
                y=merged[col],
                mode="lines",
                name=col,
            ))

        fig.update_layout(
            title=f"Equity Curve – {name} vs Benchmarks",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (start = 1.0)",
            xaxis_rangeslider_visible=False,
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------- Metrics views ----------
    elif view == "Metrics: Total Return":
        st.subheader("Total Return (%) – Comparison")
        fig = plot_metric_bar(metrics_df, "Total Return (%)", "Total Return Comparison", is_percent=True)
        st.plotly_chart(fig, use_container_width=True)

    elif view == "Metrics: Annual Return":
        st.subheader("Annual Return (%) – Comparison")
        fig = plot_metric_bar(metrics_df, "Annual Return (%)", "Annualized Return Comparison", is_percent=True)
        st.plotly_chart(fig, use_container_width=True)

    elif view == "Metrics: Success Rate":
        st.subheader("Success Rate (%) – Comparison")
        fig = plot_metric_bar(metrics_df, "Success Rate (%)", "Success Rate Comparison", is_percent=True)
        st.plotly_chart(fig, use_container_width=True)

    elif view == "Metrics: Sharpe Ratio":
        st.subheader("Sharpe Ratio – Comparison")
        fig = plot_metric_bar(metrics_df, "Sharpe Ratio", "Sharpe Ratio Comparison", is_percent=False)
        st.plotly_chart(fig, use_container_width=True)

    elif view == "Metrics: Table":
        st.subheader("Metrics Table")
        st.dataframe(
            metrics_df.style.format({
                "Success Rate (%)": "{:.2f}",
                "Total Return (%)": "{:.2f}",
                "Annual Return (%)": "{:.2f}",
                "Sharpe Ratio": "{:.2f}",
            })
        )


if __name__ == "__main__":
    main()
