from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st
import yfinance as yf

from valuation import (
    compute_dcf_result,
    dollars,
    exit_multiple_terminal_value,
    margin_label,
    multiple,
    parse_peer_tickers,
    perpetuity_terminal_value,
    project_free_cash_flows,
    scaled_money,
    select_representative_fcf,
    signed_percent,
)


PEER_MAP = {
    "KO": ["PEP", "MNST", "KDP"],
    "AAPL": ["MSFT", "GOOGL", "HPQ"],
    "MSFT": ["AAPL", "GOOGL", "ORCL"],
    "GOOGL": ["MSFT", "META", "AMZN"],
    "AMZN": ["WMT", "COST", "BABA"],
    "META": ["GOOGL", "SNAP", "PINS"],
    "TSLA": ["GM", "F", "RIVN"],
    "NVDA": ["AMD", "INTC", "AVGO"],
    "JPM": ["BAC", "C", "WFC"],
    "XOM": ["CVX", "COP", "SHEL"],
}


@dataclass
class MarketSnapshot:
    ticker: str
    price: float | None
    trailing_pe: float | None
    forward_pe: float | None
    shares_outstanding: float | None
    cash: float | None
    debt: float | None
    ttm_fcf: float | None
    latest_annual_fcf: float | None
    annual_fcf_history: list[float]
    warnings: list[str]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(converted):
        return None
    return converted


def _first_info_value(info: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = _as_float(info.get(key))
        if value is not None:
            return value
    return None


def _latest_statement_value(statement: pd.DataFrame, row_names: list[str]) -> float | None:
    if statement is None or statement.empty:
        return None
    for row_name in row_names:
        if row_name in statement.index:
            series = statement.loc[row_name].dropna()
            if not series.empty:
                return _as_float(series.iloc[0])
    return None


def _cash_flow_values(statement: pd.DataFrame, periods: int) -> pd.DataFrame:
    if statement is None or statement.empty:
        return pd.DataFrame()

    operating_cash_flow = None
    capital_expenditures = None
    for candidate in ["Operating Cash Flow", "Total Cash From Operating Activities"]:
        if candidate in statement.index:
            operating_cash_flow = statement.loc[candidate]
            break
    for candidate in ["Capital Expenditure", "Capital Expenditures"]:
        if candidate in statement.index:
            capital_expenditures = statement.loc[candidate]
            break

    if operating_cash_flow is None or capital_expenditures is None:
        return pd.DataFrame()

    cash_flow_frame = pd.DataFrame(
        {
            "Operating Cash Flow": operating_cash_flow,
            "Capital Expenditures": capital_expenditures,
        }
    ).dropna()
    cash_flow_frame = cash_flow_frame.head(periods)
    if cash_flow_frame.empty:
        return pd.DataFrame()
    cash_flow_frame["Free Cash Flow"] = (
        cash_flow_frame["Operating Cash Flow"]
        - cash_flow_frame["Capital Expenditures"].abs()
    )
    return cash_flow_frame


def _latest_shares_from_income_statement(yf_ticker: yf.Ticker) -> float | None:
    for statement_name in ["income_stmt", "quarterly_income_stmt"]:
        try:
            statement = getattr(yf_ticker, statement_name)
        except Exception:
            continue
        shares = _latest_statement_value(
            statement,
            ["Diluted Average Shares", "Diluted Shares Outstanding", "Basic Average Shares"],
        )
        if shares is not None:
            return shares
    return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_snapshot(ticker: str) -> MarketSnapshot:
    warnings: list[str] = []
    yf_ticker = yf.Ticker(ticker)

    try:
        info = yf_ticker.info or {}
    except Exception as exc:
        info = {}
        warnings.append(f"{ticker}: unable to fetch quote metadata ({exc}).")

    price = _first_info_value(info, ["currentPrice", "regularMarketPrice", "previousClose"])
    trailing_pe = _first_info_value(info, ["trailingPE"])
    forward_pe = _first_info_value(info, ["forwardPE"])
    shares = _first_info_value(
        info,
        ["sharesOutstanding", "impliedSharesOutstanding"],
    )
    if shares is None:
        shares = _latest_shares_from_income_statement(yf_ticker)

    if price is None:
        try:
            history = yf_ticker.history(period="5d", auto_adjust=False)
            if not history.empty:
                price = _as_float(history["Close"].dropna().iloc[-1])
        except Exception as exc:
            warnings.append(f"{ticker}: unable to fetch recent close price ({exc}).")

    try:
        balance_sheet = yf_ticker.quarterly_balance_sheet
        cash = _latest_statement_value(
            balance_sheet,
            ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
        )
        debt = _latest_statement_value(
            balance_sheet,
            ["Total Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"],
        )
        if debt is None:
            short_debt = _latest_statement_value(
                balance_sheet,
                ["Current Debt", "Short Long Term Debt", "Current Debt And Capital Lease Obligation"],
            )
            long_debt = _latest_statement_value(
                balance_sheet,
                ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
            )
            if short_debt is not None or long_debt is not None:
                debt = (short_debt or 0.0) + (long_debt or 0.0)
    except Exception as exc:
        cash = None
        debt = None
        warnings.append(f"{ticker}: unable to fetch quarterly balance sheet ({exc}).")

    try:
        quarterly_cash_flow = yf_ticker.quarterly_cashflow
        quarterly_fcf_frame = _cash_flow_values(quarterly_cash_flow, periods=4)
        ttm_fcf = (
            _as_float(quarterly_fcf_frame["Free Cash Flow"].sum())
            if len(quarterly_fcf_frame) >= 4
            else None
        )
    except Exception as exc:
        ttm_fcf = None
        warnings.append(f"{ticker}: unable to fetch quarterly cash flow statement ({exc}).")

    try:
        annual_cash_flow = yf_ticker.cashflow
        annual_fcf_frame = _cash_flow_values(annual_cash_flow, periods=4)
        annual_fcf_history = [
            float(value) for value in annual_fcf_frame["Free Cash Flow"].dropna().tolist()
        ]
        latest_annual_fcf = annual_fcf_history[0] if annual_fcf_history else None
    except Exception as exc:
        annual_fcf_history = []
        latest_annual_fcf = None
        warnings.append(f"{ticker}: unable to fetch annual cash flow statement ({exc}).")

    if shares is None:
        warnings.append("Diluted shares outstanding were missing from yfinance.")
    if cash is None:
        warnings.append("Cash balance was missing from yfinance.")
    if debt is None:
        warnings.append("Debt balance was missing from yfinance.")
    if ttm_fcf is None and latest_annual_fcf is None:
        warnings.append("Free cash flow was missing from yfinance.")

    return MarketSnapshot(
        ticker=ticker,
        price=price,
        trailing_pe=trailing_pe,
        forward_pe=forward_pe,
        shares_outstanding=shares,
        cash=cash,
        debt=debt,
        ttm_fcf=ttm_fcf,
        latest_annual_fcf=latest_annual_fcf,
        annual_fcf_history=annual_fcf_history,
        warnings=warnings,
    )


@st.cache_data(ttl=900, show_spinner=False)
def fetch_peer_table(peer_tickers: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for peer_ticker in peer_tickers:
        snapshot = fetch_market_snapshot(peer_ticker)
        rows.append(
            {
                "Ticker": peer_ticker,
                "Price": snapshot.price,
                "Trailing PE": snapshot.trailing_pe,
                "Forward PE": snapshot.forward_pe,
            }
        )
    return pd.DataFrame(rows)


def manual_billions_input(label: str, default_value: float | None, key: str) -> float:
    default_billions = 0.0 if default_value is None else default_value / 1_000_000_000
    entered_billions = st.number_input(
        label,
        value=float(default_billions),
        step=0.1,
        format="%.3f",
        key=key,
    )
    return entered_billions * 1_000_000_000


def manual_shares_input(default_value: float | None) -> float:
    default_billions = 0.0 if default_value is None else default_value / 1_000_000_000
    entered_billions = st.number_input(
        "Manual Entry: Diluted Shares Outstanding in Billions",
        min_value=0.0,
        value=float(default_billions),
        step=0.1,
        format="%.3f",
    )
    return entered_billions * 1_000_000_000


def render_aggregate_metric(label: str, value: float | None) -> None:
    scaled_value, unit = scaled_money(value)
    metric_value = "N/A" if scaled_value is None else f"${scaled_value:,.2f}"
    st.metric(label, metric_value, help=unit)


def render_app_chrome() -> None:
    st.markdown(
        """
        <style>
        :root {
            --surface: #ffffff;
            --surface-muted: #f7f9fb;
            --border: #d8e0e8;
            --text-soft: #5d6b7a;
            --accent: #0b6e69;
            --accent-soft: #e6f4f1;
            --blue-soft: #edf4ff;
            --amber-soft: #fff4df;
        }

        .stApp {
            background:
                linear-gradient(180deg, #f5f8fb 0%, #ffffff 34%),
                #ffffff;
        }

        [data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #17212b;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }

        .app-hero {
            border: 1px solid var(--border);
            border-radius: 8px;
            background: linear-gradient(135deg, #ffffff 0%, #f7fbfb 52%, #edf4ff 100%);
            padding: 1.4rem 1.6rem;
            margin-bottom: 1rem;
            box-shadow: 0 12px 30px rgba(23, 33, 43, 0.06);
        }

        .app-eyebrow {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }

        .app-title {
            color: #17212b;
            font-size: 2.15rem;
            font-weight: 760;
            letter-spacing: 0;
            line-height: 1.15;
            margin: 0;
        }

        .app-subtitle {
            color: var(--text-soft);
            font-size: 1rem;
            line-height: 1.55;
            margin: 0.45rem 0 0;
            max-width: 780px;
        }

        .section-kicker {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 740;
            letter-spacing: 0;
            text-transform: uppercase;
            margin: 1.15rem 0 0.1rem;
        }

        .section-title {
            color: #17212b;
            font-size: 1.25rem;
            font-weight: 720;
            line-height: 1.25;
            margin: 0 0 0.2rem;
        }

        .section-copy {
            color: var(--text-soft);
            font-size: 0.94rem;
            line-height: 1.5;
            margin-bottom: 0.75rem;
        }

        .formula-strip {
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent);
            border-radius: 8px;
            background: var(--surface);
            color: #263442;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0 1rem;
        }

        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 8px 22px rgba(23, 33, 43, 0.045);
        }

        [data-testid="stMetricLabel"] {
            color: var(--text-soft);
            font-weight: 680;
        }

        [data-testid="stMetricValue"] {
            color: #17212b;
            font-weight: 760;
        }

        [data-testid="stTabs"] button {
            border-radius: 8px 8px 0 0;
            font-weight: 680;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 8px 22px rgba(23, 33, 43, 0.035);
        }

        .stAlert {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(ticker: str) -> None:
    st.markdown(
        f"""
        <div class="app-hero">
            <div class="app-eyebrow">Interactive equity valuation</div>
            <h1 class="app-title">Stock Valuation Sandbox</h1>
            <p class="app-subtitle">
                Current focus: <strong>{ticker}</strong>. Compare market multiples,
                normalize free cash flow, and translate enterprise value into an
                estimated DCF price per share.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-kicker">{kicker}</div>
        <div class="section-title">{title}</div>
        <div class="section-copy">{copy}</div>
        """,
        unsafe_allow_html=True,
    )


def render_formula(text: str) -> None:
    st.markdown(f'<div class="formula-strip">{text}</div>', unsafe_allow_html=True)


def render_manual_guardrails(snapshot: MarketSnapshot) -> tuple[float | None, float, float, float]:
    for warning in snapshot.warnings:
        st.warning(warning)

    cash = snapshot.cash
    debt = snapshot.debt
    shares = snapshot.shares_outstanding

    with st.sidebar.expander("Manual data overrides", expanded=bool(snapshot.warnings)):
        st.caption("Use these when yfinance fields are stale, missing, or not applicable.")
        if cash is None:
            cash = manual_billions_input(
                "Manual Entry: Total Cash in Billions USD",
                cash,
                "manual_cash",
            )
        else:
            override_cash = st.checkbox("Override Total Cash", value=False)
            if override_cash:
                cash = manual_billions_input(
                    "Manual Entry: Total Cash in Billions USD",
                    cash,
                    "manual_cash_override",
                )

        if debt is None:
            debt = manual_billions_input(
                "Manual Entry: Total Debt in Billions USD",
                debt,
                "manual_debt",
            )
        else:
            override_debt = st.checkbox("Override Total Debt", value=False)
            if override_debt:
                debt = manual_billions_input(
                    "Manual Entry: Total Debt in Billions USD",
                    debt,
                    "manual_debt_override",
                )

        if shares is None:
            shares = manual_shares_input(shares)
        else:
            override_shares = st.checkbox("Override Shares Outstanding", value=False)
            if override_shares:
                shares = manual_shares_input(shares)

    with st.sidebar.expander("FCF projection baseline", expanded=True):
        baseline_options = [
            "TTM FCF",
            "Latest Fiscal Year FCF",
            "4-Year Average FCF",
            "Manual Normalized FCF",
        ]
        default_index = 0 if snapshot.ttm_fcf is not None else 1
        if snapshot.ttm_fcf is None and snapshot.latest_annual_fcf is None:
            default_index = 3
        baseline_method = st.selectbox(
            "Representative FCF for Projection",
            baseline_options,
            index=default_index,
            help=(
                "Choose the annual free cash flow baseline to grow forward. "
                "Use normalized FCF when one-off working capital, capex, or cyclicality distorts recent results."
            ),
        )
        manual_fcf = None
        if baseline_method == "Manual Normalized FCF":
            suggested_default = (
                snapshot.ttm_fcf
                or snapshot.latest_annual_fcf
                or (
                    sum(snapshot.annual_fcf_history) / len(snapshot.annual_fcf_history)
                    if snapshot.annual_fcf_history
                    else None
                )
            )
            manual_fcf = manual_billions_input(
                "Manual Entry: Normalized Annual FCF in Billions USD",
                suggested_default,
                "manual_normalized_fcf",
            )
        baseline = select_representative_fcf(
            baseline_method,
            snapshot.ttm_fcf,
            snapshot.latest_annual_fcf,
            snapshot.annual_fcf_history,
            manual_fcf,
        )
        if baseline.value is None:
            st.warning(
                "Selected FCF baseline is unavailable. Choose Manual Normalized FCF."
            )
        else:
            scaled_value, unit = scaled_money(baseline.value)
            st.info(f"Using {baseline.label}: ${scaled_value:,.2f} {unit}")
            st.caption(baseline.description)

    return baseline.value, cash or 0.0, debt or 0.0, shares or 0.0


def main() -> None:
    st.set_page_config(page_title="Stock Valuation Sandbox", page_icon="SV", layout="wide")
    render_app_chrome()

    with st.sidebar:
        st.markdown("## Sandbox Controls")
        st.caption("Set the market input, return hurdle, and DCF growth assumptions.")
        ticker = st.text_input("Ticker Symbol", value="KO").strip().upper() or "KO"
        st.markdown("### DCF Assumptions")
        discount_rate_pct = st.number_input(
            "Required Discount Rate (%)",
            min_value=0.0,
            value=7.5,
            step=0.1,
            help="Your hurdle rate. Often calculated as 10-Year Treasury Yield + Risk Premium.",
        )
        stage_one_growth_pct = st.number_input(
            "Stage 1 Growth Rate (%)",
            value=4.0,
            step=0.1,
            help="Annual FCF growth for years 1-10.",
        )
        perpetuity_growth_pct = st.number_input(
            "Stage 2 Perpetuity Growth Rate (%)",
            value=3.0,
            step=0.1,
            help="Long-run FCF growth from year 11 onward.",
        )
        exit_multiple = st.number_input(
            "Target Exit Multiple (P/FCF)",
            min_value=0.0,
            value=33.0,
            step=0.5,
        )

    render_hero(ticker)

    discount_rate = discount_rate_pct / 100
    stage_one_growth_rate = stage_one_growth_pct / 100
    perpetuity_growth_rate = perpetuity_growth_pct / 100

    with st.spinner(f"Fetching market data for {ticker}..."):
        snapshot = fetch_market_snapshot(ticker)

    baseline_fcf, cash, debt, shares = render_manual_guardrails(snapshot)
    current_price = snapshot.price or 0.0

    render_section(
        "Market Snapshot",
        "Fast read on valuation inputs",
        "These are the live market and balance sheet inputs feeding the peer and DCF views.",
    )
    metric_cols = st.columns(6)
    metric_cols[0].metric("Current Price", dollars(snapshot.price))
    metric_cols[1].metric("Trailing PE", multiple(snapshot.trailing_pe))
    metric_cols[2].metric("Forward PE", multiple(snapshot.forward_pe))
    metric_cols[3].metric("Shares", f"{shares / 1_000_000_000:,.2f}B" if shares else "N/A")
    with metric_cols[4]:
        render_aggregate_metric("DCF Baseline FCF", baseline_fcf)
    with metric_cols[5]:
        net_cash = cash - debt
        render_aggregate_metric("Net Cash / Debt", net_cash)

    st.divider()
    relative_tab, dcf_tab = st.tabs(["Peer Multiples", "DCF Engine"])

    with relative_tab:
        render_section(
            "Relative Valuation",
            "Peer group reversion estimate",
            "Compare the ticker's current earnings power against a simple peer average trailing PE.",
        )
        mapped_peers = PEER_MAP.get(ticker)
        if mapped_peers:
            peer_input = st.text_input(
                "Peer tickers",
                value=", ".join(mapped_peers),
                help="Editable fallback peer set.",
            )
        else:
            st.info("No default peer mapping found. Enter comma-separated peer tickers.")
            peer_input = st.text_input("Peer tickers", value="")

        peers = [peer for peer in parse_peer_tickers(peer_input) if peer != ticker]
        if peers:
            with st.spinner("Fetching peer multiples..."):
                peer_frame = fetch_peer_table(tuple(peers))
            display_frame = peer_frame.copy()
            display_frame["Price"] = display_frame["Price"].map(dollars)
            display_frame["Trailing PE"] = display_frame["Trailing PE"].map(multiple)
            display_frame["Forward PE"] = display_frame["Forward PE"].map(multiple)
            st.dataframe(display_frame, use_container_width=True, hide_index=True)

            valid_peer_pes = peer_frame["Trailing PE"].dropna()
            valid_peer_pes = valid_peer_pes[valid_peer_pes > 0]
            peer_avg_pe = _as_float(valid_peer_pes.mean())
            target_eps = (
                current_price / snapshot.trailing_pe
                if current_price > 0 and snapshot.trailing_pe and snapshot.trailing_pe > 0
                else None
            )
            implied_price = (
                peer_avg_pe * target_eps
                if peer_avg_pe is not None and target_eps is not None
                else None
            )
            upside = (
                implied_price / current_price - 1
                if implied_price is not None and current_price > 0
                else None
            )

            valuation_cols = st.columns(3)
            valuation_cols[0].metric("Peer Average Trailing PE", multiple(peer_avg_pe))
            valuation_cols[1].metric("Target EPS", dollars(target_eps))
            valuation_cols[2].metric(
                "Implied Reversion Price",
                dollars(implied_price),
                delta=signed_percent(upside),
            )
        else:
            st.warning("Enter at least one peer ticker to run relative valuation.")

    with dcf_tab:
        render_section(
            "Intrinsic Valuation",
            "10-year DCF converted to price per share",
            "The model projects annual FCF, discounts the enterprise value, then bridges to equity value and divides by shares.",
        )
        if baseline_fcf is None or baseline_fcf == 0 or shares <= 0 or current_price <= 0:
            st.error(
                "DCF requires current price, representative FCF, and shares outstanding. "
                "Use manual overrides if yfinance did not provide them."
            )
            return

        assumption_cols = st.columns(4)
        assumption_cols[0].metric("Baseline FCF", f"${baseline_fcf / 1_000_000_000:,.2f}B")
        assumption_cols[1].metric("Discount Rate", f"{discount_rate_pct:.1f}%")
        assumption_cols[2].metric("Stage 1 Growth", f"{stage_one_growth_pct:.1f}%")
        assumption_cols[3].metric("Exit Multiple", multiple(exit_multiple))

        if snapshot.annual_fcf_history:
            annual_history_frame = pd.DataFrame(
                {
                    "Fiscal Period": [f"Year {index}" for index in range(1, len(snapshot.annual_fcf_history) + 1)],
                    "Annual FCF": snapshot.annual_fcf_history,
                }
            )
            annual_history_frame["Annual FCF"] = annual_history_frame["Annual FCF"].map(
                lambda value: f"${value / 1_000_000_000:,.2f}B"
            )
            with st.expander("Historical annual FCF baseline context"):
                st.dataframe(annual_history_frame, use_container_width=True, hide_index=True)

        projected_fcfs = project_free_cash_flows(baseline_fcf, stage_one_growth_rate)
        projection_frame = pd.DataFrame(
            {
                "Year": list(range(1, 11)),
                "Projected FCF": projected_fcfs,
                "Discount Factor": [
                    1 / ((1 + discount_rate) ** year) for year in range(1, 11)
                ],
                "Present Value": [
                    fcf / ((1 + discount_rate) ** year)
                    for year, fcf in enumerate(projected_fcfs, start=1)
                ],
            }
        )

        year_ten_fcf = projected_fcfs[-1]
        exit_terminal = exit_multiple_terminal_value(year_ten_fcf, exit_multiple)
        exit_result = compute_dcf_result(
            projected_fcfs,
            exit_terminal,
            discount_rate,
            cash,
            debt,
            shares,
            current_price,
        )

        perpetuity_result = None
        perpetuity_error = None
        try:
            perpetuity_terminal = perpetuity_terminal_value(
                year_ten_fcf,
                discount_rate,
                perpetuity_growth_rate,
            )
            perpetuity_result = compute_dcf_result(
                projected_fcfs,
                perpetuity_terminal,
                discount_rate,
                cash,
                debt,
                shares,
                current_price,
            )
        except ValueError as exc:
            perpetuity_error = str(exc)

        summary_rows = [
            {
                "Method": "Exit Multiple",
                "Current Price": current_price,
                "DCF Price / Share": exit_result.intrinsic_value_per_share,
                "Margin of Safety": margin_label(exit_result.margin_of_safety),
                "Enterprise Value": exit_result.enterprise_value,
                "Cash": cash,
                "Debt": debt,
                "Equity Value": exit_result.equity_value,
                "Shares Outstanding": shares,
            }
        ]
        if perpetuity_result is not None:
            summary_rows.append(
                {
                    "Method": "Perpetuity Growth",
                    "Current Price": current_price,
                    "DCF Price / Share": perpetuity_result.intrinsic_value_per_share,
                    "Margin of Safety": margin_label(perpetuity_result.margin_of_safety),
                    "Enterprise Value": perpetuity_result.enterprise_value,
                    "Cash": cash,
                    "Debt": debt,
                    "Equity Value": perpetuity_result.equity_value,
                    "Shares Outstanding": shares,
                }
            )

        render_section(
            "DCF Output",
            "Price per share summary",
            "Positive margin of safety means the DCF estimate is above the current market price.",
        )
        render_formula(
            "DCF price per share = (Enterprise Value + Cash - Debt) / Diluted Shares Outstanding"
        )
        result_cols = st.columns(2)
        result_cols[0].metric(
            "Exit Multiple DCF Price / Share",
            dollars(exit_result.intrinsic_value_per_share),
            delta=signed_percent(exit_result.margin_of_safety),
            help=f"Current market price: {dollars(current_price)}",
        )
        if perpetuity_result is not None:
            result_cols[1].metric(
                "Perpetuity DCF Price / Share",
                dollars(perpetuity_result.intrinsic_value_per_share),
                delta=signed_percent(perpetuity_result.margin_of_safety),
                help=f"Current market price: {dollars(current_price)}",
            )
        else:
            result_cols[1].metric("Perpetuity DCF Price / Share", "Disabled")
            st.error(
                f"Perpetuity calculation disabled: {perpetuity_error} "
                "The denominator discount rate minus growth rate must be positive."
            )

        summary_frame = pd.DataFrame(summary_rows)
        for column in ["Current Price", "DCF Price / Share"]:
            summary_frame[column] = summary_frame[column].map(dollars)
        for column in ["Enterprise Value", "Cash", "Debt", "Equity Value"]:
            summary_frame[column] = summary_frame[column].map(
                lambda value: f"${value / 1_000_000_000:,.2f}B"
            )
        summary_frame["Shares Outstanding"] = summary_frame["Shares Outstanding"].map(
            lambda value: f"{value / 1_000_000_000:,.2f}B"
        )
        st.dataframe(summary_frame, use_container_width=True, hide_index=True)

        render_section(
            "Bridge",
            "Enterprise value to equity value",
            "The row below shows the exit-multiple method bridge all the way into the final per-share DCF price.",
        )
        bridge_cols = st.columns(5)
        bridge_cols[0].metric(
            "PV of 10-Year FCF",
            f"${exit_result.present_value_cash_flows / 1_000_000_000:,.2f}B",
        )
        bridge_cols[1].metric(
            "Enterprise Value",
            f"${exit_result.enterprise_value / 1_000_000_000:,.2f}B",
        )
        bridge_cols[2].metric(
            "Equity Value",
            f"${exit_result.equity_value / 1_000_000_000:,.2f}B",
        )
        bridge_cols[3].metric("Shares", f"{shares / 1_000_000_000:,.2f}B")
        bridge_cols[4].metric("DCF Price / Share", dollars(exit_result.intrinsic_value_per_share))

        formatted_projection = projection_frame.copy()
        formatted_projection["Projected FCF"] = formatted_projection["Projected FCF"].map(
            lambda value: f"${value / 1_000_000_000:,.2f}B"
        )
        formatted_projection["Discount Factor"] = formatted_projection["Discount Factor"].map(
            lambda value: f"{value:.3f}"
        )
        formatted_projection["Present Value"] = formatted_projection["Present Value"].map(
            lambda value: f"${value / 1_000_000_000:,.2f}B"
        )
        render_section(
            "Projection Detail",
            "10-year FCF schedule",
            "Projected cash flows are discounted year by year before terminal value is added.",
        )
        st.dataframe(formatted_projection, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
