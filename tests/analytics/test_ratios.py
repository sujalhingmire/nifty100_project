"""
test_ratios.py — Unit Tests for Financial Ratio Engine
=======================================================
Sprint 2 · Days 08–09 · Nifty 100 Financial Intelligence Platform

Covers:
    Normal calculations · Zero denominator · Negative equity · Negative assets
    Borrowings zero · Interest zero · Debt Free label · High leverage flag
    OPM mismatch · Financial company · Missing values · NaN · Large numbers
    Negative profits · Return None cases · Boundary conditions

Run:
    pytest tests/analytics/test_ratios.py -v
    pytest tests/analytics/test_ratios.py -v --cov=src/analytics/ratios --cov-report=term-missing
"""

from __future__ import annotations

import math
import logging

import pytest

from src.analytics.ratios import (
    DEBT_FREE_LABEL,
    HIGH_LEVERAGE_THRESHOLD,
    ICR_WARNING_THRESHOLD,
    OPM_MISMATCH_TOLERANCE,
    FinancialRatios,
    LeverageRatios,
    ProfitabilityRatios,
    calculate_asset_turnover,
    calculate_debt_to_equity,
    calculate_interest_coverage_ratio,
    calculate_net_debt,
    calculate_net_profit_margin,
    calculate_operating_profit_margin,
    calculate_return_on_assets,
    calculate_return_on_capital_employed,
    calculate_return_on_equity,
    compute_all_ratios,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COMPANY = "TESTCO"
YEAR = "2024-03"


def _approx(expected: float, rel: float = 1e-6) -> pytest.approx:  # type: ignore[type-arg]
    return pytest.approx(expected, rel=rel)


# ===========================================================================
# TEST GROUP 1 — Net Profit Margin
# ===========================================================================


class TestNetProfitMargin:
    """16+ tests for calculate_net_profit_margin."""

    # ------------------------------------------------------------------ #
    # Normal calculations
    # ------------------------------------------------------------------ #

    def test_npm_normal(self):
        """Standard positive profit and sales."""
        result = calculate_net_profit_margin(100, 500, company_id=COMPANY, year=YEAR)
        assert result == _approx(20.0)

    def test_npm_negative_profit(self):
        """Net loss (negative net_profit) must produce a negative margin."""
        result = calculate_net_profit_margin(-50, 500, company_id=COMPANY, year=YEAR)
        assert result == _approx(-10.0)

    def test_npm_large_numbers(self):
        """Very large financial figures (e.g. Reliance scale) must not overflow."""
        result = calculate_net_profit_margin(
            69_621, 8_76_021, company_id="RELIANCE", year=YEAR
        )
        assert result is not None
        assert result == _approx((69_621 / 8_76_021) * 100, rel=1e-4)

    def test_npm_small_margin(self):
        """Thin margin (e.g. commodity trading) near zero."""
        result = calculate_net_profit_margin(1, 1000, company_id=COMPANY, year=YEAR)
        assert result == _approx(0.1)

    def test_npm_hundred_percent(self):
        """Margin of exactly 100% when profit equals sales."""
        result = calculate_net_profit_margin(500, 500)
        assert result == _approx(100.0)

    # ------------------------------------------------------------------ #
    # Return None cases
    # ------------------------------------------------------------------ #

    def test_npm_zero_sales_returns_none(self):
        """sales == 0 → must return None."""
        result = calculate_net_profit_margin(100, 0, company_id=COMPANY, year=YEAR)
        assert result is None

    def test_npm_negative_sales_returns_none(self):
        """sales < 0 → must return None (invalid denominator)."""
        result = calculate_net_profit_margin(100, -500, company_id=COMPANY, year=YEAR)
        assert result is None

    def test_npm_none_sales_returns_none(self):
        """None sales → must return None."""
        result = calculate_net_profit_margin(100, None)
        assert result is None

    def test_npm_none_profit_returns_none(self):
        """None net_profit → must return None."""
        result = calculate_net_profit_margin(None, 500)
        assert result is None

    # ------------------------------------------------------------------ #
    # NaN / Infinity / Invalid
    # ------------------------------------------------------------------ #

    def test_npm_nan_sales_returns_none(self):
        """NaN sales → must return None."""
        result = calculate_net_profit_margin(100, float("nan"))
        assert result is None

    def test_npm_inf_profit_returns_none(self):
        """Infinite net_profit → must return None."""
        result = calculate_net_profit_margin(float("inf"), 500)
        assert result is None

    def test_npm_string_input_returns_none(self):
        """Non-numeric string → must return None."""
        result = calculate_net_profit_margin("abc", 500)
        assert result is None

    # ------------------------------------------------------------------ #
    # Boundary conditions
    # ------------------------------------------------------------------ #

    def test_npm_boundary_sales_just_above_zero(self):
        """sales just above zero should compute without error."""
        result = calculate_net_profit_margin(1, 1e-10)
        assert result is not None
        assert result > 0

    def test_npm_zero_profit(self):
        """Zero net_profit → margin of 0.0 (not None)."""
        result = calculate_net_profit_margin(0, 500)
        assert result == _approx(0.0)


# ===========================================================================
# TEST GROUP 2 — Operating Profit Margin
# ===========================================================================


class TestOperatingProfitMargin:
    """Tests for calculate_operating_profit_margin including OPM cross-check."""

    def test_opm_normal(self):
        """Standard OPM calculation."""
        result = calculate_operating_profit_margin(
            150, 1000, company_id=COMPANY, year=YEAR
        )
        assert result == _approx(15.0)

    def test_opm_zero_sales_returns_none(self):
        """sales == 0 → None."""
        result = calculate_operating_profit_margin(
            150, 0, company_id=COMPANY, year=YEAR
        )
        assert result is None

    def test_opm_mismatch_logs_warning(self, caplog):
        """OPM divergence > 1% must trigger a WARNING log."""
        with caplog.at_level(logging.WARNING, logger="src.analytics.ratios"):
            calculate_operating_profit_margin(
                150,
                1000,
                opm_percentage=10.0,    # computed=15, source=10 → diff=5 > 1
                company_id=COMPANY,
                year=YEAR,
            )
        assert "OPM mismatch" in caplog.text

    def test_opm_no_mismatch_within_tolerance(self, caplog):
        """OPM divergence ≤ 1% must NOT log an OPM-mismatch warning."""
        with caplog.at_level(logging.WARNING, logger="src.analytics.ratios"):
            calculate_operating_profit_margin(
                150,
                1000,
                opm_percentage=15.5,    # diff = 0.5 ≤ 1 → no warning
                company_id=COMPANY,
                year=YEAR,
            )
        assert "OPM mismatch" not in caplog.text

    def test_opm_missing_source_opm(self):
        """Missing opm_percentage (None) should still compute correctly."""
        result = calculate_operating_profit_margin(
            200, 1000, opm_percentage=None, company_id=COMPANY, year=YEAR
        )
        assert result == _approx(20.0)

    def test_opm_nan_operating_profit_returns_none(self):
        """NaN operating_profit → None."""
        result = calculate_operating_profit_margin(float("nan"), 1000)
        assert result is None


# ===========================================================================
# TEST GROUP 3 — Return on Equity
# ===========================================================================


class TestReturnOnEquity:

    def test_roe_normal(self):
        """Standard positive equity computation."""
        result = calculate_return_on_equity(
            100, 200, 300, company_id=COMPANY, year=YEAR
        )
        assert result == _approx(20.0)   # 100/500 * 100

    def test_roe_negative_equity_returns_none(self):
        """Negative total equity → None."""
        result = calculate_return_on_equity(
            100, -200, -300, company_id=COMPANY, year=YEAR
        )
        assert result is None

    def test_roe_zero_equity_returns_none(self):
        """equity_capital + reserves == 0 → None."""
        result = calculate_return_on_equity(
            100, 0, 0, company_id=COMPANY, year=YEAR
        )
        assert result is None

    def test_roe_negative_profit(self):
        """Negative net_profit (loss-making) → negative ROE."""
        result = calculate_return_on_equity(
            -50, 200, 300, company_id=COMPANY, year=YEAR
        )
        assert result == _approx(-10.0)

    def test_roe_missing_net_profit_returns_none(self):
        """None net_profit → None."""
        result = calculate_return_on_equity(
            None, 200, 300, company_id=COMPANY, year=YEAR
        )
        assert result is None

    def test_roe_missing_reserves_treats_as_zero(self):
        """Missing reserves defaults to 0; computation continues."""
        result = calculate_return_on_equity(
            50, 250, None, company_id=COMPANY, year=YEAR
        )
        assert result == _approx(20.0)   # 50/250 * 100


# ===========================================================================
# TEST GROUP 4 — Return on Capital Employed
# ===========================================================================


class TestReturnOnCapitalEmployed:

    def test_roce_normal(self):
        """EBIT = op_profit − dep; ROCE = EBIT / capital_employed."""
        # EBIT = 500 - 50 = 450; CE = 200 + 300 + 0 = 500; ROCE = 90%
        result = calculate_return_on_capital_employed(
            500, 50, 200, 300, 0, company_id=COMPANY, year=YEAR
        )
        assert result == _approx(90.0)

    def test_roce_with_borrowings(self):
        """Capital employed must include borrowings."""
        # EBIT=450; CE=200+300+100=600; ROCE=75%
        result = calculate_return_on_capital_employed(
            500, 50, 200, 300, 100, company_id=COMPANY, year=YEAR
        )
        assert result == _approx(75.0)

    def test_roce_zero_capital_employed_returns_none(self):
        """CE ≤ 0 → None."""
        result = calculate_return_on_capital_employed(
            500, 50, -200, -300, 0, company_id=COMPANY, year=YEAR
        )
        assert result is None

    def test_roce_financial_sector_logs_info(self, caplog):
        """Financial-sector companies should log an INFO about benchmark."""
        with caplog.at_level(logging.INFO, logger="src.analytics.ratios"):
            calculate_return_on_capital_employed(
                500, 50, 200, 300, 0,
                broad_sector="Financials",
                company_id="HDFCBANK",
                year=YEAR,
            )
        assert "Financial sector" in caplog.text

    def test_roce_missing_depreciation_defaults_to_zero(self):
        """Missing depreciation treated as 0."""
        # EBIT = 500 - 0 = 500; CE = 200+300+0=500; ROCE=100%
        result = calculate_return_on_capital_employed(
            500, None, 200, 300, 0, company_id=COMPANY, year=YEAR
        )
        assert result == _approx(100.0)


# ===========================================================================
# TEST GROUP 5 — Return on Assets
# ===========================================================================


class TestReturnOnAssets:

    def test_roa_normal(self):
        """Standard ROA."""
        result = calculate_return_on_assets(
            100, 2000, company_id=COMPANY, year=YEAR
        )
        assert result == _approx(5.0)

    def test_roa_negative_assets_returns_none(self):
        """total_assets ≤ 0 → None."""
        result = calculate_return_on_assets(100, -2000)
        assert result is None

    def test_roa_zero_assets_returns_none(self):
        """total_assets == 0 → None."""
        result = calculate_return_on_assets(100, 0)
        assert result is None

    def test_roa_negative_profit(self):
        """Loss-making → negative ROA."""
        result = calculate_return_on_assets(-100, 2000)
        assert result == _approx(-5.0)

    def test_roa_nan_assets_returns_none(self):
        """NaN total_assets → None."""
        result = calculate_return_on_assets(100, float("nan"))
        assert result is None


# ===========================================================================
# TEST GROUP 6 — Debt to Equity
# ===========================================================================


class TestDebtToEquity:

    def test_de_normal(self):
        """Standard D/E computation."""
        ratio, flag = calculate_debt_to_equity(
            500, 200, 300, company_id=COMPANY, year=YEAR
        )
        assert ratio == _approx(1.0)
        assert flag is False

    def test_de_zero_borrowings_returns_zero(self):
        """borrowings == 0 → D/E = 0.0 (not None), no flag."""
        ratio, flag = calculate_debt_to_equity(
            0, 200, 300, company_id=COMPANY, year=YEAR
        )
        assert ratio == 0.0
        assert flag is False

    def test_de_high_leverage_flag_non_financial(self):
        """D/E > 5 for non-Financial company → high_leverage_flag = True."""
        ratio, flag = calculate_debt_to_equity(
            3001, 200, 300, broad_sector="Industrials",
            company_id=COMPANY, year=YEAR
        )
        assert ratio is not None
        assert ratio > HIGH_LEVERAGE_THRESHOLD
        assert flag is True

    def test_de_high_leverage_financial_no_flag(self):
        """D/E > 5 for Financial company → flag must remain False."""
        ratio, flag = calculate_debt_to_equity(
            3001, 200, 300, broad_sector="Financials",
            company_id="HDFCBANK", year=YEAR
        )
        assert ratio is not None
        assert ratio > HIGH_LEVERAGE_THRESHOLD
        assert flag is False

    def test_de_negative_equity_returns_none(self):
        """Negative equity → (None, False)."""
        ratio, flag = calculate_debt_to_equity(
            500, -200, -300, company_id=COMPANY, year=YEAR
        )
        assert ratio is None
        assert flag is False

    def test_de_missing_borrowings_returns_none(self):
        """Missing borrowings → (None, False)."""
        ratio, flag = calculate_debt_to_equity(
            None, 200, 300, company_id=COMPANY, year=YEAR
        )
        assert ratio is None
        assert flag is False

    def test_de_exact_threshold_boundary(self):
        """D/E exactly == 5.0 for non-Financial → flag should be False (not > 5)."""
        ratio, flag = calculate_debt_to_equity(
            500, 50, 50, broad_sector="Industrials",
            company_id=COMPANY, year=YEAR
        )
        assert ratio == _approx(5.0)
        assert flag is False   # exactly 5.0, not > 5.0

    def test_de_just_above_threshold_boundary(self):
        """D/E just above 5.0 for non-Financial → flag True."""
        # 501 / 100 = 5.01
        ratio, flag = calculate_debt_to_equity(
            501, 50, 50, broad_sector="Industrials",
            company_id=COMPANY, year=YEAR
        )
        assert ratio is not None
        assert ratio > 5.0
        assert flag is True


# ===========================================================================
# TEST GROUP 7 — Interest Coverage Ratio
# ===========================================================================


class TestInterestCoverageRatio:

    def test_icr_normal(self):
        """Standard ICR with no special labels."""
        icr, label, warn = calculate_interest_coverage_ratio(
            500, 100, 100, company_id=COMPANY, year=YEAR
        )
        assert icr == _approx(6.0)
        assert label == ""
        assert warn is False

    def test_icr_zero_interest_returns_debt_free(self):
        """interest == 0 → ICR = None, label = 'Debt Free', warning = False."""
        icr, label, warn = calculate_interest_coverage_ratio(
            500, 100, 0, company_id=COMPANY, year=YEAR
        )
        assert icr is None
        assert label == DEBT_FREE_LABEL
        assert warn is False

    def test_icr_below_warning_threshold(self):
        """ICR < 1.5 → icr_warning = True."""
        # (100 + 0) / 100 = 1.0 < 1.5
        icr, label, warn = calculate_interest_coverage_ratio(
            100, 0, 100, company_id=COMPANY, year=YEAR
        )
        assert icr == _approx(1.0)
        assert warn is True

    def test_icr_exactly_at_warning_threshold(self):
        """ICR == 1.5 → icr_warning = False (not strictly less-than)."""
        # (150 + 0) / 100 = 1.5
        icr, label, warn = calculate_interest_coverage_ratio(
            150, 0, 100, company_id=COMPANY, year=YEAR
        )
        assert icr == _approx(1.5)
        assert warn is False

    def test_icr_missing_interest_returns_none(self):
        """None interest → (None, '', False)."""
        icr, label, warn = calculate_interest_coverage_ratio(500, 100, None)
        assert icr is None
        assert label == ""
        assert warn is False

    def test_icr_missing_operating_profit_returns_none(self):
        """None operating_profit with non-zero interest → (None, '', False)."""
        icr, label, warn = calculate_interest_coverage_ratio(None, 100, 200)
        assert icr is None
        assert warn is False

    def test_icr_other_income_included(self):
        """other_income must be added to numerator."""
        # (400 + 100) / 100 = 5.0
        icr, _, _ = calculate_interest_coverage_ratio(
            400, 100, 100, company_id=COMPANY, year=YEAR
        )
        assert icr == _approx(5.0)

    def test_icr_nan_interest_returns_none(self):
        """NaN interest → (None, '', False)."""
        icr, label, warn = calculate_interest_coverage_ratio(500, 100, float("nan"))
        assert icr is None


# ===========================================================================
# TEST GROUP 8 — Net Debt
# ===========================================================================


class TestNetDebt:

    def test_net_debt_normal(self):
        """Standard net debt computation."""
        result = calculate_net_debt(500, 200, company_id=COMPANY, year=YEAR)
        assert result == _approx(300.0)

    def test_net_debt_net_cash_positive(self):
        """investments > borrowings → negative net debt (net cash)."""
        result = calculate_net_debt(100, 400)
        assert result == _approx(-300.0)

    def test_net_debt_zero_borrowings(self):
        """Debt-free company: borrowings = 0."""
        result = calculate_net_debt(0, 200)
        assert result == _approx(-200.0)

    def test_net_debt_missing_investments_defaults_zero(self):
        """Missing investments treated as 0."""
        result = calculate_net_debt(500, None)
        assert result == _approx(500.0)

    def test_net_debt_missing_borrowings_returns_none(self):
        """Missing borrowings → None."""
        result = calculate_net_debt(None, 200)
        assert result is None


# ===========================================================================
# TEST GROUP 9 — Asset Turnover
# ===========================================================================


class TestAssetTurnover:

    def test_asset_turnover_normal(self):
        """Standard asset turnover."""
        result = calculate_asset_turnover(
            1000, 500, company_id=COMPANY, year=YEAR
        )
        assert result == _approx(2.0)

    def test_asset_turnover_zero_assets_returns_none(self):
        """total_assets == 0 → None."""
        result = calculate_asset_turnover(1000, 0)
        assert result is None

    def test_asset_turnover_negative_assets_returns_none(self):
        """total_assets < 0 → None."""
        result = calculate_asset_turnover(1000, -500)
        assert result is None

    def test_asset_turnover_missing_sales_returns_none(self):
        """None sales → None."""
        result = calculate_asset_turnover(None, 500)
        assert result is None

    def test_asset_turnover_nan_assets_returns_none(self):
        """NaN total_assets → None."""
        result = calculate_asset_turnover(1000, float("nan"))
        assert result is None

    def test_asset_turnover_large_numbers(self):
        """Large-scale numbers don't cause overflow."""
        result = calculate_asset_turnover(8_76_021, 12_00_000)
        assert result is not None
        assert result == _approx(8_76_021 / 12_00_000, rel=1e-4)


# ===========================================================================
# TEST GROUP 10 — compute_all_ratios orchestrator
# ===========================================================================


class TestComputeAllRatios:
    """Integration-style tests verifying the orchestrator wires everything."""

    @pytest.fixture()
    def healthy_row(self) -> dict:
        return {
            "company_id": "TCS",
            "year": "2024-03",
            "sales": 2_25_458,
            "operating_profit": 48_534,
            "opm_percentage": 21.5,
            "other_income": 3_800,
            "interest": 0,
            "depreciation": 5_800,
            "net_profit": 34_990,
            "equity_capital": 366,
            "reserves": 80_000,
            "borrowings": 0,
            "investments": 5_000,
            "total_assets": 1_20_000,
        }

    def test_orchestrator_returns_financial_ratios_type(self, healthy_row):
        """Return type must be FinancialRatios."""
        result = compute_all_ratios(healthy_row, broad_sector="Information Technology")
        assert isinstance(result, FinancialRatios)

    def test_orchestrator_debt_free_company(self, healthy_row):
        """Debt-free company: D/E == 0.0, ICR label == 'Debt Free'."""
        result = compute_all_ratios(healthy_row, broad_sector="Information Technology")
        assert result.leverage.debt_to_equity == 0.0
        assert result.leverage.icr_label == DEBT_FREE_LABEL

    def test_orchestrator_profitability_computed(self, healthy_row):
        """Profitability ratios are populated for a healthy row."""
        result = compute_all_ratios(healthy_row)
        assert result.profitability.net_profit_margin is not None
        assert result.profitability.return_on_equity is not None
        assert result.profitability.return_on_assets is not None

    def test_orchestrator_opm_mismatch_flag(self):
        """OPM mismatch flag should be True when computed vs source diverge > 1%."""
        row = {
            "company_id": "BADCO",
            "year": "2024-03",
            "sales": 1000,
            "operating_profit": 150,
            "opm_percentage": 10.0,   # computed=15, diff=5 → mismatch
            "other_income": 0,
            "interest": 50,
            "depreciation": 30,
            "net_profit": 70,
            "equity_capital": 100,
            "reserves": 400,
            "borrowings": 0,
            "investments": 0,
            "total_assets": 800,
        }
        result = compute_all_ratios(row)
        assert result.profitability.opm_mismatch is True

    def test_orchestrator_empty_row(self):
        """Completely empty row should not raise; all ratios should be None."""
        result = compute_all_ratios({})
        assert result.profitability.net_profit_margin is None
        assert result.leverage.debt_to_equity is None
        assert result.efficiency.asset_turnover is None

    def test_orchestrator_high_leverage_non_financial(self):
        """Non-Financial company with D/E > 5 should set high_leverage_flag."""
        row = {
            "company_id": "HIGHDEBT",
            "year": "2024-03",
            "sales": 500,
            "operating_profit": 100,
            "opm_percentage": 20.0,
            "other_income": 0,
            "interest": 80,
            "depreciation": 20,
            "net_profit": 20,
            "equity_capital": 10,
            "reserves": 90,
            "borrowings": 600,    # D/E = 600/100 = 6 > 5
            "investments": 0,
            "total_assets": 800,
        }
        result = compute_all_ratios(row, broad_sector="Industrials")
        assert result.leverage.high_leverage_flag is True

    def test_orchestrator_negative_equity(self):
        """Negative equity: ROE and D/E should return None."""
        row = {
            "company_id": "STRESSCO",
            "year": "2024-03",
            "sales": 500,
            "operating_profit": 100,
            "opm_percentage": 20.0,
            "other_income": 0,
            "interest": 80,
            "depreciation": 20,
            "net_profit": 20,
            "equity_capital": -200,
            "reserves": -300,
            "borrowings": 500,
            "investments": 0,
            "total_assets": 400,
        }
        result = compute_all_ratios(row)
        assert result.profitability.return_on_equity is None
        assert result.leverage.debt_to_equity is None

        