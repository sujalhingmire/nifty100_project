"""
test_edge_case_validation.py — Day 13 pytest suite
=====================================================
Covers:
    * Financial vs non-Financial sector carve-out (high_leverage_flag)
    * ROCE cross-check — match / mismatch / boundary
    * ROE cross-check — match / mismatch / boundary
    * TCS-style known ROE anomaly (source << computed)
    * Anomaly category assignment
    * Structured logging to output/ratio_edge_cases.log
    * Threshold boundary behaviour at exactly 5%
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: ensure the project root (which contains the ``src`` package) is
# on sys.path. Needed when this file is executed directly with
# ``python tests/kpi/test_edge_case_validation.py``, and harmless/no-op when
# run properly via ``pytest`` from the project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_BOOTSTRAP))

from src.analytics import ratio_edge_cases as edge
from src.analytics import ratios


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_edge_case_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the edge-case log to a temp file and reset the singleton.

    Ensures each test writes to (and reads from) an isolated
    ``ratio_edge_cases.log`` instead of the real project ``output/`` dir,
    and that the module-level cached logger is rebuilt every test.
    """
    log_path = tmp_path / "ratio_edge_cases.log"
    monkeypatch.setattr(edge, "RATIO_EDGE_CASES_LOG_PATH", log_path)
    monkeypatch.setattr(edge, "_edge_case_logger", None)

    # Drop any handlers left on the named logger from a previous test run.
    named_logger = __import__("logging").getLogger("ratio_edge_cases")
    for handler in list(named_logger.handlers):
        named_logger.removeHandler(handler)

    yield log_path


# ---------------------------------------------------------------------------
# TASK 1 — Financial sector carve-out
# ---------------------------------------------------------------------------


class TestFinancialSectorCarveOut:
    def test_financial_company_high_leverage_flag_forced_false(self) -> None:
        """D/E > 5 for a Financials company must NOT set high_leverage_flag."""
        de_ratio, high_leverage = ratios.calculate_debt_to_equity(
            borrowings=1000,
            equity_capital=10,
            reserves=90,  # total equity = 100 → D/E = 10 (> 5)
            broad_sector="Financials",
            company_id="HDFCBANK",
            year="2024",
        )
        assert de_ratio == pytest.approx(10.0)
        assert high_leverage is False

    def test_non_financial_company_high_leverage_flag_true(self) -> None:
        """D/E > 5 for a non-Financials company MUST set high_leverage_flag."""
        de_ratio, high_leverage = ratios.calculate_debt_to_equity(
            borrowings=1000,
            equity_capital=10,
            reserves=90,
            broad_sector="Industrials",
            company_id="TATASTEEL",
            year="2024",
        )
        assert de_ratio == pytest.approx(10.0)
        assert high_leverage is True

    def test_is_financial_company_true(self) -> None:
        assert edge.is_financial_company("Financials") is True

    def test_is_financial_company_false(self) -> None:
        assert edge.is_financial_company("Information Technology") is False

    def test_is_financial_company_handles_blank(self) -> None:
        assert edge.is_financial_company("") is False

    def test_is_financial_company_case_insensitive_variant(self) -> None:
        assert edge.is_financial_company("financials") is True


# ---------------------------------------------------------------------------
# TASK 2 — ROCE cross-check
# ---------------------------------------------------------------------------


class TestValidateROCE:
    def test_roce_match_within_threshold_no_anomaly(self) -> None:
        result = edge.validate_roce(
            computed_roce=18.0,
            source_roce=20.0,  # diff = 2.0 <= 5.0
            company_id="INFY",
            year="2024",
        )
        assert result is None

    def test_roce_mismatch_above_threshold_logs_anomaly(self) -> None:
        result = edge.validate_roce(
            computed_roce=30.0,
            source_roce=15.0,  # diff = 15.0 > 5.0
            company_id="RELIANCE",
            year="2024",
        )
        assert result is not None
        assert result.metric == "ROCE"
        assert result.computed_value == pytest.approx(30.0)
        assert result.source_value == pytest.approx(15.0)
        assert result.difference == pytest.approx(15.0)
        assert result.category in edge.VALID_CATEGORIES

    def test_roce_missing_values_skipped(self) -> None:
        assert edge.validate_roce(None, 12.0, company_id="X", year="2024") is None
        assert edge.validate_roce(12.0, None, company_id="X", year="2024") is None


# ---------------------------------------------------------------------------
# TASK 3 — ROE cross-check
# ---------------------------------------------------------------------------


class TestValidateROE:
    def test_roe_match_within_threshold_no_anomaly(self) -> None:
        result = edge.validate_roe(
            computed_roe=22.0,
            source_roe=24.5,  # diff = 2.5 <= 5.0
            company_id="HCLTECH",
            year="2024",
        )
        assert result is None

    def test_roe_mismatch_above_threshold_logs_anomaly(self) -> None:
        result = edge.validate_roe(
            computed_roe=35.0,
            source_roe=20.0,  # diff = 15.0 > 5.0
            company_id="WIPRO",
            year="2024",
        )
        assert result is not None
        assert result.metric == "ROE"
        assert result.difference == pytest.approx(15.0)

    def test_tcs_known_anomaly_source_far_below_computed(self) -> None:
        """TCS: source ROE = 0.52 (known bad data) vs a realistic computed ROE.

        Computed ROE must be the value used for analytics; the anomaly must
        still be logged and categorised, and the source value preserved
        only as the (unused) comparison figure.
        """
        result = edge.validate_roe(
            computed_roe=45.0,
            source_roe=0.52,
            company_id="TCS",
            year="2024",
        )
        assert result is not None
        assert result.computed_value == pytest.approx(45.0)
        assert result.source_value == pytest.approx(0.52)
        # Computed value is what analytics consumers should use — confirm
        # the helper never mutates/overwrites it.
        assert result.computed_value != result.source_value
        assert result.category == edge.CATEGORY_DATA_SOURCE_ISSUE

    def test_roe_missing_values_skipped(self) -> None:
        assert edge.validate_roe(None, 12.0, company_id="X", year="2024") is None
        assert edge.validate_roe(12.0, None, company_id="X", year="2024") is None


# ---------------------------------------------------------------------------
# TASK 4 — Category assignment
# ---------------------------------------------------------------------------


class TestCategoriseAnomaly:
    def test_data_source_issue_for_near_zero_source(self) -> None:
        category, _ = edge.categorise_anomaly(
            computed_value=45.0, source_value=0.52, difference=44.48
        )
        assert category == edge.CATEGORY_DATA_SOURCE_ISSUE

    def test_data_source_issue_for_large_multiple(self) -> None:
        category, _ = edge.categorise_anomaly(
            computed_value=100.0, source_value=5.0, difference=95.0
        )
        assert category == edge.CATEGORY_DATA_SOURCE_ISSUE

    def test_version_difference_mid_band(self) -> None:
        category, _ = edge.categorise_anomaly(
            computed_value=25.0, source_value=15.0, difference=10.0
        )
        assert category == edge.CATEGORY_VERSION_DIFFERENCE

    def test_formula_discrepancy_for_comparable_magnitudes(self) -> None:
        category, _ = edge.categorise_anomaly(
            computed_value=22.0, source_value=20.0, difference=2.0
        )
        assert category == edge.CATEGORY_FORMULA_DISCREPANCY

    def test_category_is_always_one_of_three_valid_values(self) -> None:
        samples = [
            (30.0, 15.0, 15.0),
            (45.0, 0.52, 44.48),
            (22.0, 20.0, 2.0),
            (60.0, 8.0, 52.0),
        ]
        for computed, source, diff in samples:
            category, reason = edge.categorise_anomaly(computed, source, diff)
            assert category in edge.VALID_CATEGORIES
            assert isinstance(reason, str) and reason


# ---------------------------------------------------------------------------
# TASK 5 — Structured logging
# ---------------------------------------------------------------------------


class TestLogging:
    def test_anomaly_written_to_ratio_edge_cases_log(
        self, _isolate_edge_case_log: Path
    ) -> None:
        edge.validate_roce(
            computed_roce=30.0,
            source_roce=10.0,
            company_id="ITC",
            year="2023",
        )
        assert _isolate_edge_case_log.exists()
        content = _isolate_edge_case_log.read_text(encoding="utf-8")
        assert "ITC" in content
        assert "2023" in content
        assert "ROCE" in content
        assert "Category=" in content
        assert "Severity=" in content
        assert "Recommended Action=" in content

    def test_log_entry_contains_all_required_fields(
        self, _isolate_edge_case_log: Path
    ) -> None:
        edge.validate_roe(
            computed_roe=45.0,
            source_roe=0.52,
            company_id="TCS",
            year="2024",
        )
        content = _isolate_edge_case_log.read_text(encoding="utf-8")
        required_fields = [
            "Timestamp=",
            "Company=",
            "Year=",
            "Metric=",
            "Computed=",
            "Source=",
            "Difference=",
            "Category=",
            "Severity=",
            "Explanation=",
            "Recommended Action=",
        ]
        for fld in required_fields:
            assert fld in content

    def test_no_log_entry_when_within_threshold(
        self, _isolate_edge_case_log: Path
    ) -> None:
        edge.validate_roce(
            computed_roce=20.0, source_roce=21.0, company_id="LT", year="2024"
        )
        assert not _isolate_edge_case_log.exists()

    def test_process_validation_logs_both_roce_and_roe_anomalies(
        self, _isolate_edge_case_log: Path
    ) -> None:
        anomalies = edge.process_validation(
            company_id="AXISBANK",
            year="2024",
            broad_sector="Financials",
            computed_roce=20.0,
            source_roce=5.0,
            computed_roe=30.0,
            source_roe=10.0,
        )
        assert len(anomalies) == 2
        content = _isolate_edge_case_log.read_text(encoding="utf-8")
        assert content.count("Company=AXISBANK") == 2


# ---------------------------------------------------------------------------
# TASK 6 / Threshold boundary
# ---------------------------------------------------------------------------


class TestThresholdBoundary:
    def test_exactly_five_percent_difference_is_not_an_anomaly(self) -> None:
        """Difference == 5.0 must NOT trigger logging (rule is '> 5%')."""
        result = edge.validate_roce(
            computed_roce=25.0, source_roce=20.0, company_id="SBIN", year="2024"
        )
        assert result is None

    def test_just_above_five_percent_is_an_anomaly(self) -> None:
        result = edge.validate_roce(
            computed_roce=25.01, source_roce=20.0, company_id="SBIN", year="2024"
        )
        assert result is not None
        assert result.difference == pytest.approx(5.01)

    def test_custom_threshold_override(self) -> None:
        """A caller-supplied threshold is honoured for both ROCE and ROE."""
        assert (
            edge.validate_roce(
                computed_roce=12.0,
                source_roce=10.0,
                company_id="X",
                year="2024",
                threshold=1.0,
            )
            is not None
        )
        assert (
            edge.validate_roe(
                computed_roe=12.0,
                source_roe=10.0,
                company_id="X",
                year="2024",
                threshold=1.0,
            )
            is not None
        )


# ---------------------------------------------------------------------------
# Severity mapping sanity checks
# ---------------------------------------------------------------------------


class TestSeverity:
    def test_severity_escalates_with_difference(self) -> None:
        assert edge._severity_for_difference(6.0) == "LOW"
        assert edge._severity_for_difference(11.0) == "MEDIUM"
        assert edge._severity_for_difference(21.0) == "HIGH"
        assert edge._severity_for_difference(51.0) == "CRITICAL"