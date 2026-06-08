import pytest

from valuation import (
    compute_dcf_result,
    margin_label,
    parse_peer_tickers,
    perpetuity_terminal_value,
    project_free_cash_flows,
    select_representative_fcf,
)


def test_project_free_cash_flows_compounds_from_baseline():
    projected = project_free_cash_flows(100.0, 0.10, years=3)

    assert projected == pytest.approx([110.0, 121.0, 133.1])


def test_perpetuity_terminal_value_requires_growth_below_discount_rate():
    with pytest.raises(ValueError):
        perpetuity_terminal_value(100.0, discount_rate=0.03, perpetuity_growth_rate=0.03)


def test_compute_dcf_result_bridges_ev_to_equity_value():
    result = compute_dcf_result(
        projected_fcfs=[100.0, 100.0],
        terminal_value=1_000.0,
        discount_rate=0.10,
        cash=50.0,
        debt=20.0,
        shares_outstanding=10.0,
        current_price=80.0,
    )

    expected_pv_cash_flows = 100.0 / 1.10 + 100.0 / (1.10**2)
    expected_pv_terminal = 1_000.0 / (1.10**2)
    expected_enterprise_value = expected_pv_cash_flows + expected_pv_terminal
    expected_equity_value = expected_enterprise_value + 50.0 - 20.0

    assert result.enterprise_value == pytest.approx(expected_enterprise_value)
    assert result.equity_value == pytest.approx(expected_equity_value)
    assert result.intrinsic_value_per_share == pytest.approx(expected_equity_value / 10.0)


def test_margin_label_flags_overvaluation():
    assert margin_label(-0.25) == "Overvalued by 25.0%"
    assert margin_label(0.20) == "Margin of safety 20.0%"


def test_parse_peer_tickers_normalizes_and_deduplicates():
    assert parse_peer_tickers(" pep, mnst,PEP ,, kdp ") == ["KDP", "MNST", "PEP"]


def test_select_representative_fcf_uses_ttm_when_selected():
    baseline = select_representative_fcf(
        "TTM FCF",
        ttm_fcf=120.0,
        latest_annual_fcf=100.0,
        annual_fcf_history=[100.0, 90.0, 80.0, 70.0],
    )

    assert baseline.value == 120.0
    assert baseline.label == "TTM FCF"


def test_select_representative_fcf_can_average_annual_history():
    baseline = select_representative_fcf(
        "4-Year Average FCF",
        ttm_fcf=None,
        latest_annual_fcf=100.0,
        annual_fcf_history=[100.0, 90.0, 80.0, 70.0],
    )

    assert baseline.value == pytest.approx(85.0)


def test_select_representative_fcf_accepts_manual_normalized_value():
    baseline = select_representative_fcf(
        "Manual Normalized FCF",
        ttm_fcf=None,
        latest_annual_fcf=None,
        annual_fcf_history=[],
        manual_fcf=95.0,
    )

    assert baseline.value == 95.0
