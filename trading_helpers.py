import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.express as px
import yfinance as yf

# -----------------------
# Config
# -----------------------
TRADING_DAYS = 252         # trading days per year
RISK_FREE_ANNUAL = 0.02    # 2% annual risk-free (change if needed)


# -----------------------
# Data loading
# -----------------------
import yfinance as yf
import pandas as pd

import yfinance as yf
import pandas as pd
import numpy as np

def load_price_data_yf(
    ticker: str = "AAPL",
    start=None,           # <--- MUST BE HERE
    end=None,             # <--- MUST BE HERE
    interval: str = "1d", # <--- MUST BE HERE
) -> pd.DataFrame:
    """
    Load historical price data from Yahoo Finance using yfinance,
    using the same logic as the notebook and allowing start/end/interval.

    Returns columns: Date, Open, High, Low, Close, Volume (if available),
    Adj Close (if available), daily_ret.
    """
    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            f"No data returned by yfinance for {ticker} "
            f"with start={start}, end={end}, interval={interval}"
        )

    aapl = data.copy()

    # Handle MultiIndex columns like ('Open','AAPL')
    if isinstance(aapl.columns, pd.MultiIndex):
        aapl.columns = aapl.columns.droplevel(1)

    aapl.columns.name = None

    # Index -> column "Date"
    aapl = aapl.reset_index()

    needed_cols = ["Date", "Open", "High", "Low", "Close"]
    if "Volume" in aapl.columns:
        needed_cols.append("Volume")
    if "Adj Close" in aapl.columns:
        needed_cols.append("Adj Close")

    aapl = aapl[needed_cols]

    aapl["Date"] = pd.to_datetime(aapl["Date"])
    aapl = aapl.sort_values("Date").reset_index(drop=True)

    # Use Adj Close for returns if available
    price_col = "Adj Close" if "Adj Close" in aapl.columns else "Close"
    aapl["daily_ret"] = aapl[price_col].pct_change().fillna(0)

    return aapl

def load_benchmark_yf(
    ticker: str,
    start: str = None,
    end: str = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Load a benchmark (index ETF, etc.) from Yahoo Finance and build a
    buy-and-hold equity curve.
    Returns: Date, Adj Close, bench_ret, bench_equity.
    """
    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,   # use adjusted prices
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No data returned for benchmark ticker {ticker}")

    data = data.reset_index()

    # Adj Close if available, else Close
    price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
    p = data[price_col].astype(float)

    data["bench_ret"] = p.pct_change().fillna(0.0)
    data["bench_equity"] = (1 + data["bench_ret"]).cumprod()

    return data[["Date", price_col, "bench_ret", "bench_equity"]]


def load_fund(path: str = "TDB908.csv") -> pd.DataFrame:
    """
    Load TD Nasdaq Index fund data from CSV and build fund_ret + fund_equity.

    Expects a 'Date' column and a 'Return' column like '0.87%' (string).
    """
    fund = pd.read_csv(path)
    fund["Date"] = pd.to_datetime(fund["Date"])
    fund = fund.sort_values("Date").reset_index(drop=True)

    # Convert 'Return' like "0.87%" -> decimal 0.0087
    fund["fund_ret"] = (
        fund["Return"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .astype(float)
        / 100.0
    )

    # Equity curve starting at 1.0
    fund["fund_equity"] = (1 + fund["fund_ret"]).cumprod()

    return fund




# -----------------------
# Strategy 1: Donchian breakout
# -----------------------

def add_donchian_atr_indicators(df,
                                upper_len=20,
                                lower_len_entry=20,
                                lower_len_exit=10,
                                atr_len=14):
    """
    Adds:
      - DC_high_20: rolling 20-day highest high (entry breakout level)
      - DC_low_20:  rolling 20-day lowest low  (context only, optional)
      - DC_low_10:  rolling 10-day lowest low  (exit)
      - ATR:        14-day Average True Range
    """
    df = df.copy()

    df["DC_high_20"] = df["High"].rolling(window=upper_len, min_periods=upper_len).max()
    df["DC_low_20"]  = df["Low"].rolling(window=lower_len_entry, min_periods=lower_len_entry).min()
    df["DC_low_10"]  = df["Low"].rolling(window=lower_len_exit, min_periods=lower_len_exit).min()

    df["ATR"] = ta.atr(high=df["High"], low=df["Low"], close=df["Close"], length=atr_len)

    return df


def run_donchian_breakout_strategy(df,
                                   upper_len=20,
                                   lower_len_entry=20,
                                   lower_len_exit=10,
                                   atr_len=14,
                                   atr_mult=2.0,
                                   max_holding=60,
                                   initial_capital=1.0):
    """
    Donchian Channel Volatility Breakout (long-only).

    ENTRY:
        - Close_t > DC_high_20_(t-1)  (new 20-day breakout)

    EXIT:
        - Close_t < DC_low_10_t       (short-term breakdown)
        OR
        - Close_t < entry_price - atr_mult * ATR_at_entry  (vol stop)
        OR
        - holding_days >= max_holding  (time stop)
    """
    df = df.copy()
    df = add_donchian_atr_indicators(df,
                                     upper_len=upper_len,
                                     lower_len_entry=lower_len_entry,
                                     lower_len_exit=lower_len_exit,
                                     atr_len=atr_len)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["position"] = 0
    in_position = False
    holding_days = 0
    entry_price = np.nan
    entry_atr = np.nan

    pos_col = df.columns.get_loc("position")

    for i in range(len(df)):
        row = df.iloc[i]

        # --- clean NA check: use .any() on a small Series of the 3 columns ---
        has_na = pd.isna(row[["DC_high_20", "DC_low_10", "ATR"]]).any()

        if (i == 0) or has_na:
            # not enough history yet or indicators not ready
            df.iloc[i, pos_col] = 1 if in_position else 0
            continue

        prev = df.iloc[i - 1]

        # ENTRY: breakout above previous 20-day high
        breakout = row["Close"] > prev["DC_high_20"]

        if (not in_position) and breakout:
            in_position = True
            holding_days = 0
            entry_price = row["Close"]
            entry_atr = row["ATR"]
            df.iloc[i, pos_col] = 1

        elif in_position:
            holding_days += 1
            stop_level = entry_price - atr_mult * entry_atr
            breakdown  = row["Close"] < row["DC_low_10"]
            vol_stop   = row["Close"] < stop_level
            time_exit  = holding_days >= max_holding

            if breakdown or vol_stop or time_exit:
                in_position = False
                holding_days = 0
                entry_price = np.nan
                entry_atr = np.nan
                df.iloc[i, pos_col] = 0
            else:
                df.iloc[i, pos_col] = 1
        else:
            df.iloc[i, pos_col] = 0

    # Strategy daily returns
    df["strategy_ret"] = df["position"].shift(1) * df["daily_ret"]
    df["strategy_ret"] = df["strategy_ret"].fillna(0)

    # Equity curves
    df["strategy_equity"] = initial_capital * (1 + df["strategy_ret"]).cumprod()
    df["buyhold_equity"]  = initial_capital * (1 + df["daily_ret"]).cumprod()

    return df



# -----------------------
# Strategy 2: Supertrend
# -----------------------

import numpy as np
import pandas as pd
import pandas_ta as ta

def add_supertrend(df, length=10, multiplier=3.0):
    """
    Adds Supertrend (trend-following volatility band).

    Creates:
      - SUPERT      : Supertrend line
      - SUPERT_DIR  : Direction (+1 uptrend, -1 downtrend)

    Handles both:
      - ta.supertrend(...)
      - df.ta.supertrend(...) (which may return None and just append to df)
    """
    df = df.copy()

    # Try the functional API first
    st_df = ta.supertrend(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        length=length,
        multiplier=multiplier,
    )

    if st_df is None:
        # Fallback: use the DataFrame accessor, which appends columns to df
        df.ta.supertrend(length=length, multiplier=multiplier, append=True)

        # Find the columns that were just created
        super_cols = [c for c in df.columns if c.startswith("SUPERT")]

        if len(super_cols) < 2:
            raise ValueError(
                "pandas_ta.supertrend did not create expected SUPERT columns. "
                f"Found columns: {super_cols}"
            )

        df["SUPERT"] = df[super_cols[0]]
        df["SUPERT_DIR"] = df[super_cols[1]]

    else:
        # ta.supertrend returned something – usually a DataFrame
        if isinstance(st_df, pd.DataFrame):
            df["SUPERT"] = st_df.iloc[:, 0]
            if st_df.shape[1] > 1:
                df["SUPERT_DIR"] = st_df.iloc[:, 1]
            else:
                # Derive direction if only one column is returned
                df["SUPERT_DIR"] = np.where(df["Close"] >= df["SUPERT"], 1, -1)
        else:
            # If it's a Series, treat it as the line and infer direction
            df["SUPERT"] = st_df
            df["SUPERT_DIR"] = np.where(df["Close"] >= df["SUPERT"], 1, -1)

    return df



def run_supertrend_strategy(df,
                            length=10,
                            multiplier=3.0,
                            initial_capital=1.0):
    """
    Supertrend Trend-Following (long-only).

    ENTRY: Close crosses ABOVE Supertrend.
    EXIT : Close crosses BELOW Supertrend.
    """
    df = df.copy()
    df = add_supertrend(df, length=length, multiplier=multiplier)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["position"] = 0
    in_position = False
    pos_col = df.columns.get_loc("position")

    for i in range(len(df)):
        row = df.iloc[i]

        if i == 0 or pd.isna(row["SUPERT"]):
            df.iloc[i, pos_col] = 1 if in_position else 0
            continue

        prev = df.iloc[i - 1]

        cross_up   = (prev["Close"] <= prev["SUPERT"]) and (row["Close"] > row["SUPERT"])
        cross_down = (prev["Close"] >= prev["SUPERT"]) and (row["Close"] < row["SUPERT"])

        if (not in_position) and cross_up:
            in_position = True
            df.iloc[i, pos_col] = 1
        elif in_position and cross_down:
            in_position = False
            df.iloc[i, pos_col] = 0
        else:
            df.iloc[i, pos_col] = 1 if in_position else 0

    df["strategy_ret"] = df["position"].shift(1) * df["daily_ret"]
    df["strategy_ret"] = df["strategy_ret"].fillna(0)

    df["strategy_equity"] = initial_capital * (1 + df["strategy_ret"]).cumprod()
    df["buyhold_equity"]  = initial_capital * (1 + df["daily_ret"]).cumprod()

    return df


# -----------------------
# Strategy 3: Parabolic SAR
# -----------------------

def add_psar(df, step=0.02, max_step=0.2):
    """
    Adds a unified Parabolic SAR series (PSAR) to df.
    Combines long/short SAR into a single PSAR series.
    """
    df = df.copy()

    psar = ta.psar(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        step=step,
        max_step=max_step,
    )

    long_sar  = psar.iloc[:, 0]
    short_sar = psar.iloc[:, 1]

    df["PSAR"] = long_sar.fillna(short_sar)

    return df


def run_psar_trend_strategy(df,
                            step=0.02,
                            max_step=0.2,
                            initial_capital=1.0):
    """
    Parabolic SAR Trend Strategy (long-only).

    ENTRY: PSAR flips from ABOVE price to BELOW price.
    EXIT : PSAR flips from BELOW price to ABOVE price.
    """
    df = df.copy()
    df = add_psar(df, step=step, max_step=max_step)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["position"] = 0
    in_position = False
    pos_col = df.columns.get_loc("position")

    for i in range(len(df)):
        row = df.iloc[i]

        if i == 0 or pd.isna(row["PSAR"]):
            df.iloc[i, pos_col] = 1 if in_position else 0
            continue

        prev = df.iloc[i - 1]

        flip_up   = (prev["PSAR"] >= prev["Close"]) and (row["PSAR"] < row["Close"])
        flip_down = (prev["PSAR"] <= prev["Close"]) and (row["PSAR"] > row["Close"])

        if (not in_position) and flip_up:
            in_position = True
            df.iloc[i, pos_col] = 1
        elif in_position and flip_down:
            in_position = False
            df.iloc[i, pos_col] = 0
        else:
            df.iloc[i, pos_col] = 1 if in_position else 0

    df["strategy_ret"] = df["position"].shift(1) * df["daily_ret"]
    df["strategy_ret"] = df["strategy_ret"].fillna(0)

    df["strategy_equity"] = initial_capital * (1 + df["strategy_ret"]).cumprod()
    df["buyhold_equity"]  = initial_capital * (1 + df["daily_ret"]).cumprod()

    return df


# -----------------------
# Metrics
# -----------------------

def success_rate(returns: pd.Series) -> float:
    """
    Fraction of periods with positive return.
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0
    wins = (returns > 0).sum()
    return wins / len(returns)


def total_and_annual_return(equity: pd.Series,
                            periods_per_year: int = TRADING_DAYS):
    """
    Computes total return and annualized return from an equity curve.
    """
    equity = equity.dropna()
    if len(equity) < 2:
        return 0.0, 0.0

    total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
    n_periods = len(equity) - 1
    if n_periods <= 0:
        return total_ret, 0.0

    ann_ret = (1.0 + total_ret) ** (periods_per_year / n_periods) - 1.0
    return total_ret, ann_ret


def sharpe_ratio(returns: pd.Series,
                 risk_free_annual: float = RISK_FREE_ANNUAL,
                 periods_per_year: int = TRADING_DAYS) -> float:
    """
    Annualized Sharpe ratio using excess returns over risk-free rate.
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan

    rf_period = (1.0 + risk_free_annual) ** (1.0 / periods_per_year) - 1.0
    excess = returns - rf_period

    mean_excess = excess.mean()
    std_excess = excess.std(ddof=1)

    if std_excess == 0:
        return np.nan

    return (mean_excess / std_excess) * np.sqrt(periods_per_year)


def summarize_strategy(name, ret_series, equity_series, periods_per_year):
    sr   = success_rate(ret_series)
    tot, ann = total_and_annual_return(equity_series, periods_per_year=periods_per_year)
    sh   = sharpe_ratio(ret_series, periods_per_year=periods_per_year)
    return {
        "Strategy": name,
        "Success Rate (%)": sr * 100,
        "Total Return (%)": tot * 100,
        "Annual Return (%)": ann * 100,
        "Sharpe Ratio": sh,
    }


def build_metrics_df(
    super_results,
    psar_results,
    donch_results,
    bench_name: str,
    bench_ret: pd.Series,
    bench_equity: pd.Series,
    periods_per_year: int,
    main_ticker: str,          # <--- add this
) -> pd.DataFrame:
    rows = []
    rows.append(
        summarize_strategy(
            "Supertrend",
            super_results["strategy_ret"],
            super_results["strategy_equity"],
            periods_per_year,
        )
    )
    rows.append(
        summarize_strategy(
            "Parabolic SAR",
            psar_results["strategy_ret"],
            psar_results["strategy_equity"],
            periods_per_year,
        )
    )
    rows.append(
        summarize_strategy(
            "Donchian Breakout",
            donch_results["strategy_ret"],
            donch_results["strategy_equity"],
            periods_per_year,
        )
    )
    rows.append(
        summarize_strategy(
            f"Buy & Hold {main_ticker.upper()}",
            donch_results["daily_ret"],
            donch_results["buyhold_equity"],
            periods_per_year,
        )
    )
    rows.append(
        summarize_strategy(
            bench_name,
            bench_ret,
            bench_equity,
            periods_per_year,
        )
    )

    return pd.DataFrame(rows)



# -----------------------
# High-level helper
# -----------------------

def run_all(
    ticker: str = "AAPL",
    start=None,
    end=None,
    interval: str = "1d",
    fund_path: str = "TDB908.csv",
    benchmark_ticker: str = "QQQ",
    benchmark_mode: str = "td_fund",
):
    # Map interval to periods per year
    if interval == "1d":
        periods_per_year = 252
    elif interval == "1wk":
        periods_per_year = 52
    elif interval == "1mo":
        periods_per_year = 12
    else:
        periods_per_year = TRADING_DAYS

    # 1) Main price data
    aapl_clean = load_price_data_yf(
        ticker=ticker,
        start=start,
        end=end,
        interval=interval,
    )

    # 2) Strategies on main ticker
    donch_results = run_donchian_breakout_strategy(aapl_clean)
    super_results = run_supertrend_strategy(aapl_clean)
    psar_results  = run_psar_trend_strategy(aapl_clean)

    # 3) Benchmark selection
    if benchmark_mode == "td_fund":
        # Use TD Nasdaq mutual fund from CSV
        bench_df = load_fund(fund_path)

        # Align to same date window as ticker data
        min_date = aapl_clean["Date"].min()
        max_date = aapl_clean["Date"].max()
        bench_df = bench_df[
            (bench_df["Date"] >= min_date) & (bench_df["Date"] <= max_date)
        ].reset_index(drop=True)

        bench_name   = "TDB908 Fund"
        bench_ret    = bench_df["fund_ret"]
        bench_equity = bench_df["fund_equity"]

    else:  # benchmark_mode == "yf_ticker"
        bench_df = load_benchmark_yf(
            ticker=benchmark_ticker,
            start=start,
            end=end,
            interval=interval,
        )

        # Align to same date window as ticker data (defensive)
        min_date = aapl_clean["Date"].min()
        max_date = aapl_clean["Date"].max()
        bench_df = bench_df[
            (bench_df["Date"] >= min_date) & (bench_df["Date"] <= max_date)
        ].reset_index(drop=True)

        bench_name   = f"Buy & Hold {benchmark_ticker}"
        bench_ret    = bench_df["bench_ret"]
        bench_equity = bench_df["bench_equity"]

    # 4) Metrics
    metrics_df = build_metrics_df(
        super_results,
        psar_results,
        donch_results,
        bench_name,
        bench_ret,
        bench_equity,
        periods_per_year=periods_per_year,
        main_ticker=ticker,
    )

    # 5) Return everything the app needs
    return aapl_clean, bench_df, bench_name, super_results, psar_results, donch_results, metrics_df

def plot_metric_bar(metrics_df, metric_col, title, is_percent=True):
    """
    Create a horizontal bar chart comparing a single metric
    (e.g., 'Total Return (%)') across all strategies/benchmarks.
    """
    df_plot = metrics_df.sort_values(metric_col, ascending=False).copy()

    fig = px.bar(
        df_plot,
        x=metric_col,
        y="Strategy",
        orientation="h",
        text=df_plot[metric_col].round(2),
        title=title,
    )

    # Put labels outside the bars
    fig.update_traces(textposition="outside")

    # Add a bit of space on the right for labels
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


# -----------------------


