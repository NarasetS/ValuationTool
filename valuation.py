from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import mean


@dataclass(frozen=True)
class DcfResult:
    intrinsic_value_per_share: float
    enterprise_value: float
    equity_value: float
    terminal_value: float
    present_value_terminal: float
    present_value_cash_flows: float
    margin_of_safety: float | None


@dataclass(frozen=True)
class FcfBaseline:
    value: float | None
    label: str
    description: str


def is_valid_number(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return isfinite(number)


def parse_peer_tickers(raw_value: str) -> list[str]:
    return sorted(
        {
            token.strip().upper()
            for token in raw_value.split(",")
            if token.strip()
        }
    )


def select_representative_fcf(
    method: str,
    ttm_fcf: float | None,
    latest_annual_fcf: float | None,
    annual_fcf_history: list[float],
    manual_fcf: float | None = None,
) -> FcfBaseline:
    method = method.strip().lower()
    if method == "manual normalized fcf":
        return FcfBaseline(
            manual_fcf,
            "Manual Normalized FCF",
            "User-entered annualized free cash flow baseline.",
        )
    if method == "ttm fcf":
        return FcfBaseline(
            ttm_fcf,
            "TTM FCF",
            "Sum of the latest four quarterly free cash flow values.",
        )
    if method == "latest fiscal year fcf":
        return FcfBaseline(
            latest_annual_fcf,
            "Latest Fiscal Year FCF",
            "Most recent annual operating cash flow minus capital expenditures.",
        )
    if method == "4-year average fcf":
        valid_values = [value for value in annual_fcf_history if is_valid_number(value)]
        return FcfBaseline(
            mean(valid_values) if valid_values else None,
            "4-Year Average FCF",
            "Average annual free cash flow across the available last four fiscal years.",
        )
    raise ValueError(f"Unknown FCF baseline method: {method}")


def project_free_cash_flows(
    baseline_fcf: float,
    stage_one_growth_rate: float,
    years: int = 10,
) -> list[float]:
    if years <= 0:
        raise ValueError("Projection years must be positive.")
    return [
        baseline_fcf * ((1 + stage_one_growth_rate) ** year)
        for year in range(1, years + 1)
    ]


def present_value(future_value: float, discount_rate: float, year: int) -> float:
    if discount_rate <= -1:
        raise ValueError("Discount rate must be greater than -100%.")
    if year <= 0:
        raise ValueError("Discount year must be positive.")
    return future_value / ((1 + discount_rate) ** year)


def exit_multiple_terminal_value(year_ten_fcf: float, exit_multiple: float) -> float:
    return year_ten_fcf * exit_multiple


def perpetuity_terminal_value(
    year_ten_fcf: float,
    discount_rate: float,
    perpetuity_growth_rate: float,
) -> float:
    if perpetuity_growth_rate >= discount_rate:
        raise ValueError(
            "Perpetuity growth rate must be lower than the discount rate."
        )
    return year_ten_fcf * (1 + perpetuity_growth_rate) / (
        discount_rate - perpetuity_growth_rate
    )


def compute_dcf_result(
    projected_fcfs: list[float],
    terminal_value: float,
    discount_rate: float,
    cash: float,
    debt: float,
    shares_outstanding: float,
    current_price: float | None = None,
) -> DcfResult:
    if not projected_fcfs:
        raise ValueError("At least one projected cash flow is required.")
    if shares_outstanding <= 0:
        raise ValueError("Shares outstanding must be positive.")

    pv_cash_flows = sum(
        present_value(fcf, discount_rate, year)
        for year, fcf in enumerate(projected_fcfs, start=1)
    )
    pv_terminal = present_value(terminal_value, discount_rate, len(projected_fcfs))
    enterprise_value = pv_cash_flows + pv_terminal
    equity_value = enterprise_value + cash - debt
    intrinsic_value_per_share = equity_value / shares_outstanding
    margin_of_safety = (
        None
        if current_price is None or intrinsic_value_per_share <= 0
        else 1 - (current_price / intrinsic_value_per_share)
    )

    return DcfResult(
        intrinsic_value_per_share=intrinsic_value_per_share,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        terminal_value=terminal_value,
        present_value_terminal=pv_terminal,
        present_value_cash_flows=pv_cash_flows,
        margin_of_safety=margin_of_safety,
    )


def margin_label(margin_of_safety: float | None) -> str:
    if margin_of_safety is None:
        return "N/A"
    if margin_of_safety >= 0:
        return f"Margin of safety {margin_of_safety:.1%}"
    return f"Overvalued by {abs(margin_of_safety):.1%}"


def signed_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.1%}"


def dollars(value: float | None) -> str:
    if value is None or not is_valid_number(value):
        return "N/A"
    return f"${float(value):,.2f}"


def multiple(value: float | None) -> str:
    if value is None or not is_valid_number(value):
        return "N/A"
    return f"{float(value):,.1f}x"


def scaled_money(value: float | None) -> tuple[float | None, str]:
    if value is None or not is_valid_number(value):
        return None, "N/A"
    absolute_value = abs(float(value))
    if absolute_value >= 1_000_000_000:
        return float(value) / 1_000_000_000, "Billions USD"
    if absolute_value >= 1_000_000:
        return float(value) / 1_000_000, "Millions USD"
    return float(value), "USD"
