"""
tests/kpi/test_cashflow_kpis.py
================================
Nifty 100 Financial Intelligence Platform
------------------------------------------
Production-grade pytest test suite for cashflow_kpis.py.

Covers
------
* Normal FCF
* Negative FCF
* PAT zero
* Sales zero
* Operating profit zero
* High Quality CFO
* Moderate CFO
* Accrual Risk
* Capital Allocation patterns
* CSV generation
* Log generation
* Missing / NaN values
* Batch processing via process_all_companies()

Python : 3.10+
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import pandas as pd
import pytest

# Point output artefacts at a temp directory so tests never pollute cwd
_TEST_OUTPUT = Path(__file__).parent.parent.parent / "_test_output"
os.environ.setdefault("NIFTY_OUTPUT_DIR", str(_TEST_OUTPUT))

# Import AFTER setting env-var so the module picks it up
from src.analytics.cashflow_kpis import (  # noqa: E402
    CashFlowRow,
    CapitalAllocationRow,
    calculate_capex_intensity,
    calculate_fcf,
    calculate_fcf_conversion,
    calculate_cfo_quality,
    classify_capital_allocation,
    generate_capital_allocation_csv,
    process_all_companies,
    process_company_cashflows,
    write_edge_case_log,
    _EDGE_CASE_LOG,
    _CAPITAL_ALLOCATION_CSV,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_output(tmp_path, monkeypatch):
    """
    Redirect all file output to a pytest-managed tmp directory so each test
    runs in isolation with no leftover artefacts.
    """
    monkeypatch.setenv("NIFTY_OUTPUT_DIR", str(tmp_path))
    # Patch the module-level path constants so they use the new directory
    import src.analytics.cashflow_kpis as mod
    monkeypatch.setattr(mod, "_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(mod, "_CAPITAL_ALLOCATION_CSV", tmp_path / "capital_allocation.csv")
    monkeypatch.setattr(mod, "_EDGE_CASE_LOG", tmp_path / "ratio_edge_cases.log")
    yield tmp_path


def make_row(
    company_id: str = "TEST",
    year: int = 2023,
    operating_activity: float = 1000.0,
    investing_activity: float = -400.0,
    financing_activity: float = -200.0,
    sales: float = 5000.0,
    pat: float = 800.0,
    operating_profit: float = 900.0,
) -> CashFlowRow:
    """Factory helper for :class:`CashFlowRow`."""
    return CashFlowRow(
        company_id=company_id,
        year=year,
        operating_activity=operating_activity,
        investing_activity=investing_activity,
        financing_activity=financing_activity,
        sales=sales,
        pat=pat,
        operating_profit=operating_profit,
    )


# ---------------------------------------------------------------------------
# Test 1 – Normal FCF (positive result)
# ---------------------------------------------------------------------------

class TestCalculateFCF:
    def test_normal_fcf_positive(self):
        """FCF = CFO + CFI where result is positive."""
        row = make_row(operating_activity=1500.0, investing_activity=-400.0)
        result = calculate_fcf(row)
        assert result.fcf == pytest.approx(1100.0)
        assert result.company_id == "TEST"
        assert result.year == 2023

    # Test 2 – Negative FCF is valid (never forced positive)
    def test_negative_fcf_is_valid(self):
        """FCF must remain negative when CFO + CFI < 0."""
        row = make_row(operating_activity=300.0, investing_activity=-800.0)
        result = calculate_fcf(row)
        assert result.fcf == pytest.approx(-500.0)
        assert result.fcf < 0, "Negative FCF must not be forced positive"

    def test_fcf_nan_cfo_returns_none(self):
        """NaN operating_activity → fcf=None."""
        row = make_row(operating_activity=float("nan"))
        result = calculate_fcf(row)
        assert result.fcf is None

    def test_fcf_nan_cfi_returns_none(self):
        """NaN investing_activity → fcf=None."""
        row = make_row(investing_activity=float("nan"))
        result = calculate_fcf(row)
        assert result.fcf is None

    def test_fcf_zero_values(self):
        """FCF with both inputs at zero."""
        row = make_row(operating_activity=0.0, investing_activity=0.0)
        result = calculate_fcf(row)
        assert result.fcf == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 3 – PAT zero → CFO Quality score = None
# ---------------------------------------------------------------------------

class TestCFOQuality:
    def test_pat_zero_returns_none(self):
        """When PAT is 0, the ratio for that year is skipped; if no valid
        ratios exist the score must be None."""
        rows = [make_row(year=2023, pat=0.0, operating_activity=500.0)]
        result = calculate_cfo_quality(rows)
        assert result.score is None
        assert result.label is None

    # Test 5 – High Quality CFO
    def test_high_quality_cfo(self):
        """Average CFO/PAT > 1.0 → High Quality."""
        rows = [
            make_row(year=y, operating_activity=1200.0, pat=900.0)
            for y in range(2019, 2024)
        ]
        result = calculate_cfo_quality(rows)
        assert result.score is not None
        assert result.score > 1.0
        assert result.label == "High Quality"

    # Test 6 – Moderate CFO
    def test_moderate_cfo(self):
        """Average CFO/PAT between 0.5 and 1.0 → Moderate."""
        rows = [
            make_row(year=y, operating_activity=700.0, pat=1000.0)
            for y in range(2019, 2024)
        ]
        result = calculate_cfo_quality(rows)
        assert result.score is not None
        assert 0.5 <= result.score <= 1.0
        assert result.label == "Moderate"

    # Test 7 – Accrual Risk
    def test_accrual_risk_cfo(self):
        """Average CFO/PAT < 0.5 → Accrual Risk."""
        rows = [
            make_row(year=y, operating_activity=200.0, pat=1000.0)
            for y in range(2019, 2024)
        ]
        result = calculate_cfo_quality(rows)
        assert result.score is not None
        assert result.score < 0.5
        assert result.label == "Accrual Risk"

    def test_cfo_quality_uses_last_5_years(self):
        """Score is computed over the most-recent 5 years even when more rows
        are provided."""
        rows = [
            # Years 2017-2018 have LOW ratio (0.1) → should be excluded
            make_row(year=2017, operating_activity=100.0, pat=1000.0),
            make_row(year=2018, operating_activity=100.0, pat=1000.0),
            # Years 2019-2023 have HIGH ratio (>1.0) → last 5 years
            *[
                make_row(year=y, operating_activity=1500.0, pat=900.0)
                for y in range(2019, 2024)
            ],
        ]
        result = calculate_cfo_quality(rows)
        assert result.label == "High Quality"

    def test_empty_rows_returns_none(self):
        """Empty row list → score=None, label=None."""
        result = calculate_cfo_quality([])
        assert result.score is None
        assert result.label is None


# ---------------------------------------------------------------------------
# Test 4 – Sales zero → CapEx Intensity = None
# ---------------------------------------------------------------------------

class TestCapExIntensity:
    def test_sales_zero_returns_none(self):
        """sales==0 must return value=None, label=None."""
        row = make_row(sales=0.0, investing_activity=-400.0)
        result = calculate_capex_intensity(row)
        assert result.value is None
        assert result.label is None

    def test_asset_light(self):
        """Intensity < 3% → Asset Light."""
        row = make_row(investing_activity=-100.0, sales=5000.0)
        result = calculate_capex_intensity(row)
        # 100/5000*100 = 2%
        assert result.value == pytest.approx(2.0)
        assert result.label == "Asset Light"

    def test_moderate_capex(self):
        """Intensity 3–8% → Moderate."""
        row = make_row(investing_activity=-250.0, sales=5000.0)
        result = calculate_capex_intensity(row)
        # 250/5000*100 = 5%
        assert result.value == pytest.approx(5.0)
        assert result.label == "Moderate"

    def test_capital_intensive(self):
        """Intensity ≥ 8% → Capital Intensive."""
        row = make_row(investing_activity=-500.0, sales=5000.0)
        result = calculate_capex_intensity(row)
        # 500/5000*100 = 10%
        assert result.value == pytest.approx(10.0)
        assert result.label == "Capital Intensive"

    def test_absolute_value_of_investing(self):
        """Positive investing_activity is also treated via abs()."""
        row = make_row(investing_activity=300.0, sales=5000.0)
        result = calculate_capex_intensity(row)
        assert result.value == pytest.approx(6.0)
        assert result.label == "Moderate"


# ---------------------------------------------------------------------------
# Test 8 – Operating profit zero → FCF Conversion = None
# ---------------------------------------------------------------------------

class TestFCFConversion:
    def test_operating_profit_zero_returns_none(self):
        """operating_profit==0 must return value=None."""
        row = make_row(operating_profit=0.0)
        result = calculate_fcf_conversion(row, fcf=600.0)
        assert result.value is None

    def test_normal_fcf_conversion(self):
        """FCF Conversion = FCF / operating_profit * 100."""
        row = make_row(operating_profit=1000.0)
        result = calculate_fcf_conversion(row, fcf=800.0)
        assert result.value == pytest.approx(80.0)

    def test_fcf_none_propagates(self):
        """When fcf is None, conversion must also be None."""
        row = make_row(operating_profit=1000.0)
        result = calculate_fcf_conversion(row, fcf=None)
        assert result.value is None

    def test_negative_fcf_conversion(self):
        """Negative FCF produces negative conversion rate."""
        row = make_row(operating_profit=1000.0)
        result = calculate_fcf_conversion(row, fcf=-500.0)
        assert result.value == pytest.approx(-50.0)


# ---------------------------------------------------------------------------
# Test 9 – Capital Allocation patterns
# ---------------------------------------------------------------------------

class TestCapitalAllocation:
    def _row(self, cfo: float, cfi: float, cff: float) -> CashFlowRow:
        return make_row(
            operating_activity=cfo,
            investing_activity=cfi,
            financing_activity=cff,
        )

    def test_reinvestor(self):
        row = self._row(500, -300, -100)
        result = classify_capital_allocation(row, cfo_quality_label="Moderate")
        assert result.pattern_label == "Reinvestor"
        assert result.cfo_sign == "+"
        assert result.cfi_sign == "-"
        assert result.cff_sign == "-"

    def test_shareholder_returns_override(self):
        """(+,-,-) with High Quality CFO → Shareholder Returns."""
        row = self._row(1200, -200, -400)
        result = classify_capital_allocation(row, cfo_quality_label="High Quality")
        assert result.pattern_label == "Shareholder Returns"

    def test_liquidating_assets(self):
        row = self._row(300, 500, -100)
        result = classify_capital_allocation(row)
        assert result.pattern_label == "Liquidating Assets"

    def test_distress_signal(self):
        row = self._row(-400, 600, 800)
        result = classify_capital_allocation(row)
        assert result.pattern_label == "Distress Signal"

    def test_growth_funded_by_debt(self):
        row = self._row(-200, -500, 900)
        result = classify_capital_allocation(row)
        assert result.pattern_label == "Growth Funded by Debt"

    def test_cash_accumulator(self):
        row = self._row(500, 300, 200)
        result = classify_capital_allocation(row)
        assert result.pattern_label == "Cash Accumulator"

    def test_pre_revenue(self):
        row = self._row(-100, -200, -50)
        result = classify_capital_allocation(row)
        assert result.pattern_label == "Pre-Revenue"

    def test_mixed_pattern(self):
        row = self._row(400, -200, 100)
        result = classify_capital_allocation(row)
        assert result.pattern_label == "Mixed"

    def test_invalid_nan_returns_unknown(self):
        row = make_row(operating_activity=float("nan"))
        result = classify_capital_allocation(row)
        assert result.pattern_label == "Unknown"


# ---------------------------------------------------------------------------
# Test 10 – CSV generation
# ---------------------------------------------------------------------------

class TestCSVGeneration:
    def test_csv_created(self, clean_output):
        """generate_capital_allocation_csv() must create the file."""
        rows = [
            CapitalAllocationRow("CO1", 2022, "+", "-", "-", "Reinvestor"),
            CapitalAllocationRow("CO1", 2023, "+", "-", "-", "Shareholder Returns"),
        ]
        path = generate_capital_allocation_csv(rows)
        assert path.exists()

    def test_csv_header_and_data(self, clean_output):
        """CSV must contain correct headers and row data."""
        rows = [
            CapitalAllocationRow("AAPL", 2021, "+", "-", "+", "Mixed"),
        ]
        path = generate_capital_allocation_csv(rows)
        with path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            records = list(reader)
        assert len(records) == 1
        assert records[0]["company_id"] == "AAPL"
        assert records[0]["year"] == "2021"
        assert records[0]["pattern_label"] == "Mixed"

    def test_csv_empty_rows(self, clean_output):
        """Empty row list → CSV with header only."""
        path = generate_capital_allocation_csv([])
        with path.open(encoding="utf-8") as fh:
            content = fh.read()
        assert "company_id" in content
        lines = [ln for ln in content.splitlines() if ln.strip()]
        assert len(lines) == 1  # header only


# ---------------------------------------------------------------------------
# Test 11 – Log generation
# ---------------------------------------------------------------------------

class TestLogGeneration:
    def test_edge_case_log_created(self, clean_output):
        """write_edge_case_log() must create ratio_edge_cases.log."""
        write_edge_case_log("COMP_A", 2022, "Test entry", "INFO")
        log_path = clean_output / "ratio_edge_cases.log"
        assert log_path.exists()

    def test_log_contains_required_fields(self, clean_output):
        """Every log entry must include timestamp, severity, company, year, reason."""
        write_edge_case_log("COMP_B", 2020, "Division by zero in CapEx", "WARNING")
        log_path = clean_output / "ratio_edge_cases.log"
        content = log_path.read_text(encoding="utf-8")
        assert "COMP_B" in content
        assert "2020" in content
        assert "WARNING" in content
        assert "Division by zero" in content
        # Timestamp format check: should contain 'T' (ISO 8601)
        assert "T" in content

    def test_log_appends_multiple_entries(self, clean_output):
        """Multiple calls must append distinct lines."""
        write_edge_case_log("C1", 2021, "PAT zero", "WARNING")
        write_edge_case_log("C2", 2022, "Sales zero", "WARNING")
        log_path = clean_output / "ratio_edge_cases.log"
        lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# Test 12 – Missing values (NaN / None propagation through batch)
# ---------------------------------------------------------------------------

class TestMissingValues:
    def test_nan_sales_in_dataframe(self, clean_output):
        """Rows with NaN sales should not crash; CapEx Intensity returns None."""
        df = pd.DataFrame([
            {
                "company_id": "NAN_CO",
                "year": 2022,
                "operating_activity": 500.0,
                "investing_activity": -100.0,
                "financing_activity": -50.0,
                "sales": None,   # NaN
                "pat": 400.0,
                "operating_profit": 450.0,
            }
        ])
        results, csv_path = process_all_companies(df)
        # Row with NaN is excluded → company result with no rows
        assert len(results) == 1
        assert results[0].fcf_results == []  # row was dropped

    def test_string_numbers_coerced(self, clean_output):
        """String numeric values in DataFrame must be coerced correctly."""
        df = pd.DataFrame([
            {
                "company_id": "STR_CO",
                "year": "2023",
                "operating_activity": "1000",
                "investing_activity": "-300",
                "financing_activity": "-200",
                "sales": "8000",
                "pat": "700",
                "operating_profit": "850",
            }
        ])
        results, _ = process_all_companies(df)
        assert len(results) == 1
        fcf = results[0].fcf_results[0].fcf
        assert fcf == pytest.approx(700.0)  # 1000 + (-300)

    def test_duplicate_years_deduped(self, clean_output):
        """Duplicate year rows for one company must be deduplicated."""
        rows = [
            make_row(company_id="DUP_CO", year=2022, operating_activity=800.0),
            make_row(company_id="DUP_CO", year=2022, operating_activity=999.0),  # dup
        ]
        result = process_company_cashflows(rows)
        # Only one FCF result should exist
        assert len(result.fcf_results) == 1
        # First occurrence wins
        assert result.fcf_results[0].fcf == pytest.approx(800.0 + (-400.0))

    def test_unsorted_years_processed_correctly(self, clean_output):
        """Rows given in reverse year order must be sorted before processing."""
        rows = [
            make_row(company_id="SORT_CO", year=2023, operating_activity=1000.0),
            make_row(company_id="SORT_CO", year=2021, operating_activity=600.0),
            make_row(company_id="SORT_CO", year=2022, operating_activity=800.0),
        ]
        result = process_company_cashflows(rows)
        years = [r.year for r in result.fcf_results]
        assert years == sorted(years)

    def test_infinity_in_cfo_returns_none_fcf(self, clean_output):
        """Infinite CFO → FCF must be None."""
        row = make_row(operating_activity=math.inf)
        result = calculate_fcf(row)
        assert result.fcf is None


# ---------------------------------------------------------------------------
# Integration – process_all_companies with a realistic batch
# ---------------------------------------------------------------------------

class TestBatchProcessing:
    def _build_df(self, n_companies: int = 3, n_years: int = 5) -> pd.DataFrame:
        records = []
        for i in range(n_companies):
            for y in range(2019, 2019 + n_years):
                records.append({
                    "company_id": f"COMP_{i:02d}",
                    "year": y,
                    "operating_activity": 1000.0 + i * 100,
                    "investing_activity": -(300.0 + i * 50),
                    "financing_activity": -(200.0 + i * 30),
                    "sales": 5000.0 + i * 500,
                    "pat": 800.0 + i * 80,
                    "operating_profit": 900.0 + i * 90,
                })
        return pd.DataFrame(records)

    def test_all_companies_processed(self, clean_output):
        df = self._build_df(n_companies=5, n_years=5)
        results, csv_path = process_all_companies(df)
        assert len(results) == 5

    def test_csv_row_count_matches(self, clean_output):
        """CSV row count = total company-year records across all companies."""
        n_companies, n_years = 3, 4
        df = self._build_df(n_companies=n_companies, n_years=n_years)
        _, csv_path = process_all_companies(df)
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == n_companies * n_years

    def test_missing_required_column_raises(self, clean_output):
        """DataFrame missing required column must raise ValueError."""
        df = pd.DataFrame([{"company_id": "X", "year": 2023}])
        with pytest.raises(ValueError, match="missing required columns"):
            process_all_companies(df)

    def test_cfo_quality_label_present(self, clean_output):
        df = self._build_df(n_companies=2, n_years=5)
        results, _ = process_all_companies(df)
        for r in results:
            assert r.cfo_quality is not None
            assert r.cfo_quality.label in {"High Quality", "Moderate", "Accrual Risk"}

    def test_capital_allocation_signs_valid(self, clean_output):
        df = self._build_df(n_companies=2, n_years=3)
        results, _ = process_all_companies(df)
        valid_signs = {"+", "-", "?"}
        for r in results:
            for ca in r.capital_allocation_rows:
                assert ca.cfo_sign in valid_signs
                assert ca.cfi_sign in valid_signs
                assert ca.cff_sign in valid_signs