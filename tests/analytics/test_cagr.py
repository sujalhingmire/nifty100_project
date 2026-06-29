"""
test_cagr.py — Unit Tests for CAGR Engine
==========================================
Sprint 2 · Day 10 · Nifty 100 Financial Intelligence Platform

Covers:
    Normal CAGR · Revenue CAGR · PAT CAGR · EPS CAGR
    Turnaround · Decline to Loss · Both Negative · Zero Base
    Insufficient Years · Missing Values · NaN Handling
    Duplicate Years · Sorted Years · Large Numbers · Decimal Precision
    Batch processing · Output schema · Empty DataFrame · Error resilience

Run:
    pytest tests/analytics/test_cagr.py -v
    pytest tests/analytics/test_cagr.py -v \
        --cov=src.analytics.cagr --cov-report=term-missing
"""

from __future__ import annotations

import logging
import pandas as pd
import pytest

from src.analytics.cagr import (
    METRIC_COLUMNS,
    OUTPUT_COLUMNS,
    PERIODS,
    CAGRFlag,
    CAGRResult,
    calculate_cagr,
    compute_all_cagrs,
    compute_company_cagr,
    compute_growth_metric,
    determine_cagr_flag,
    validate_cagr_inputs,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

COMPANY = "TESTCO"
YEAR = "2024-03"


def _r(value: object, flag: CAGRFlag) -> CAGRResult:
    """Shorthand to build an expected CAGRResult."""
    return CAGRResult(value=value, flag=flag)


def _make_pl_df(
    company_id: str,
    years: list[str],
    sales: list[float],
    net_profit: list[float],
    eps: list[float],
) -> pd.DataFrame:
    """Build a minimal P&L DataFrame for one company."""
    return pd.DataFrame(
        {
            "company_id": company_id,
            "year": years,
            "sales": sales,
            "net_profit": net_profit,
            "eps": eps,
        }
    )


@pytest.fixture()
def tcs_df() -> pd.DataFrame:
    """11-year TCS-like P&L slice (FY2014–FY2024), all values positive."""
    years = [
        "2014-03", "2015-03", "2016-03", "2017-03", "2018-03",
        "2019-03", "2020-03", "2021-03", "2022-03", "2023-03", "2024-03",
    ]
    sales = [
        86_917, 1_08_646, 1_23_104, 1_27_771, 1_31_651,
        1_46_463, 1_56_949, 1_64_177, 1_91_754, 2_25_458, 2_40_893,
    ]
    net_profit = [
        19_164, 22_067, 23_906, 26_257, 25_826,
        31_472, 32_340, 32_430, 38_327, 42_147, 46_099,
    ]
    eps = [
        97.37, 112.02, 121.37, 133.41, 131.24,
        159.82, 164.29, 164.73, 102.74, 114.29, 125.18,
    ]
    return _make_pl_df("TCS", years, sales, net_profit, eps)


@pytest.fixture()
def multi_company_df(tcs_df: pd.DataFrame) -> pd.DataFrame:
    """Two-company DataFrame for batch-processing tests."""
    infy = _make_pl_df(
        "INFY",
        ["2014-03", "2015-03", "2016-03", "2017-03", "2018-03",
         "2019-03", "2020-03", "2021-03", "2022-03", "2023-03", "2024-03"],
        [52_791, 53_319, 62_441, 68_484, 70_522,
         82_675, 90_791, 1_00_472, 1_21_641, 1_46_767, 1_53_670],
        [12_164, 12_329, 13_491, 14_353, 16_029,
         15_410, 16_594, 19_351, 22_110, 24_095, 26_248],
        [21.27, 21.57, 23.59, 25.36, 28.46,
         35.93, 38.97, 45.71, 52.52, 57.09, 62.27],
    )
    return pd.concat([tcs_df, infy], ignore_index=True)


# ===========================================================================
# TEST GROUP 1 — determine_cagr_flag
# ===========================================================================


class TestDetermineCagrFlag:
    """Unit tests for the flag classification function."""

    def test_both_positive_normal(self) -> None:
        assert determine_cagr_flag(100.0, 200.0) == CAGRFlag.NORMAL

    def test_start_positive_end_negative_decline(self) -> None:
        assert determine_cagr_flag(100.0, -50.0) == CAGRFlag.DECLINE_TO_LOSS

    def test_start_negative_end_positive_turnaround(self) -> None:
        assert determine_cagr_flag(-100.0, 200.0) == CAGRFlag.TURNAROUND

    def test_both_negative(self) -> None:
        assert determine_cagr_flag(-100.0, -50.0) == CAGRFlag.BOTH_NEGATIVE

    def test_zero_start_zero_base(self) -> None:
        assert determine_cagr_flag(0.0, 200.0) == CAGRFlag.ZERO_BASE

    def test_zero_start_exact_tolerance(self) -> None:
        """Value within _ZERO_TOL must be treated as zero."""
        assert determine_cagr_flag(1e-10, 200.0) == CAGRFlag.ZERO_BASE

    def test_start_positive_end_zero_both_negative_branch(self) -> None:
        """end == 0 with positive start falls through to BOTH_NEGATIVE branch.
        The CAGR formula would yield -100% (total loss); returning None is correct.
        """
        assert determine_cagr_flag(100.0, 0.0) == CAGRFlag.BOTH_NEGATIVE


# ===========================================================================
# TEST GROUP 2 — validate_cagr_inputs
# ===========================================================================


class TestValidateCagrInputs:
    """Tests for input validation helper."""

    def test_valid_inputs_returns_floats(self) -> None:
        s, e, flag = validate_cagr_inputs(100, 200, 5)
        assert s == pytest.approx(100.0)
        assert e == pytest.approx(200.0)
        assert flag is None

    def test_none_start_returns_missing(self) -> None:
        _, _, flag = validate_cagr_inputs(None, 200, 5)
        assert flag == CAGRFlag.MISSING

    def test_none_end_returns_missing(self) -> None:
        _, _, flag = validate_cagr_inputs(100, None, 5)
        assert flag == CAGRFlag.MISSING

    def test_nan_start_returns_missing(self) -> None:
        _, _, flag = validate_cagr_inputs(float("nan"), 200, 5)
        assert flag == CAGRFlag.MISSING

    def test_inf_end_returns_missing(self) -> None:
        _, _, flag = validate_cagr_inputs(100, float("inf"), 5)
        assert flag == CAGRFlag.MISSING

    def test_string_numeric_accepted(self) -> None:
        """String '100' must be coerced to float 100.0."""
        s, e, flag = validate_cagr_inputs("100", "200", 5)
        assert s == pytest.approx(100.0)
        assert e == pytest.approx(200.0)
        assert flag is None

    def test_string_non_numeric_returns_missing(self) -> None:
        _, _, flag = validate_cagr_inputs("abc", 200, 5)
        assert flag == CAGRFlag.MISSING

    def test_invalid_years_zero_returns_insufficient(self) -> None:
        _, _, flag = validate_cagr_inputs(100, 200, 0)
        assert flag == CAGRFlag.INSUFFICIENT

    def test_invalid_years_negative_returns_insufficient(self) -> None:
        _, _, flag = validate_cagr_inputs(100, 200, -3)
        assert flag == CAGRFlag.INSUFFICIENT


# ===========================================================================
# TEST GROUP 3 — calculate_cagr (core formula)
# ===========================================================================


class TestCalculateCagr:
    """Tests for the primary CAGR calculation function."""

    # ------------------------------------------------------------------ #
    # Normal calculations
    # ------------------------------------------------------------------ #

    def test_normal_doubling_5yr(self) -> None:
        """100 → 200 in 5 years = (2^0.2 - 1) × 100 ≈ 14.87%."""
        result = calculate_cagr(100, 200, 5)
        assert result.flag == CAGRFlag.NORMAL
        assert result.value == pytest.approx(14.87, abs=0.01)

    def test_normal_tripling_10yr(self) -> None:
        """100 → 300 in 10 years = (3^0.1 - 1) × 100 ≈ 11.61%."""
        result = calculate_cagr(100, 300, 10)
        assert result.flag == CAGRFlag.NORMAL
        assert result.value == pytest.approx(11.61, abs=0.01)

    def test_normal_3yr_window(self) -> None:
        """1000 → 1331 in 3 years = exactly 10.00%."""
        result = calculate_cagr(1000, 1331, 3)
        assert result.flag == CAGRFlag.NORMAL
        assert result.value == pytest.approx(10.00, abs=0.01)

    def test_normal_decimal_precision_two_places(self) -> None:
        """Result must be rounded to exactly 2 decimal places."""
        result = calculate_cagr(100, 161.05, 5)
        assert result.flag == CAGRFlag.NORMAL
        assert result.value is not None
        # Verify it is truly rounded to 2dp (not 3+ dp).
        assert result.value == round(result.value, 2)

    def test_normal_large_numbers(self) -> None:
        """Reliance-scale numbers must compute without overflow."""
        result = calculate_cagr(3_00_000, 8_00_000, 10)
        assert result.flag == CAGRFlag.NORMAL
        assert result.value is not None
        expected = round(((8_00_000 / 3_00_000) ** 0.1 - 1) * 100, 2)
        assert result.value == pytest.approx(expected, rel=1e-5)

    def test_normal_shrinkage_negative_cagr(self) -> None:
        """Positive start > positive end → negative CAGR (contraction)."""
        result = calculate_cagr(200, 100, 5)
        assert result.flag == CAGRFlag.NORMAL
        assert result.value is not None
        assert result.value < 0

    # ------------------------------------------------------------------ #
    # Edge cases — all six
    # ------------------------------------------------------------------ #

    def test_decline_to_loss(self) -> None:
        """Start > 0, End < 0 → (None, DECLINE_TO_LOSS)."""
        result = calculate_cagr(100, -50, 5)
        assert result.value is None
        assert result.flag == CAGRFlag.DECLINE_TO_LOSS

    def test_turnaround(self) -> None:
        """Start < 0, End > 0 → (None, TURNAROUND)."""
        result = calculate_cagr(-100, 200, 5)
        assert result.value is None
        assert result.flag == CAGRFlag.TURNAROUND

    def test_both_negative(self) -> None:
        """Both negative → (None, BOTH_NEGATIVE)."""
        result = calculate_cagr(-200, -50, 5)
        assert result.value is None
        assert result.flag == CAGRFlag.BOTH_NEGATIVE

    def test_zero_base(self) -> None:
        """Start == 0 → (None, ZERO_BASE)."""
        result = calculate_cagr(0, 200, 5)
        assert result.value is None
        assert result.flag == CAGRFlag.ZERO_BASE

    def test_missing_start(self) -> None:
        """None start → (None, MISSING)."""
        result = calculate_cagr(None, 200, 5)
        assert result.value is None
        assert result.flag == CAGRFlag.MISSING

    def test_missing_end(self) -> None:
        """None end → (None, MISSING)."""
        result = calculate_cagr(100, None, 5)
        assert result.value is None
        assert result.flag == CAGRFlag.MISSING

    def test_nan_start(self) -> None:
        """NaN start → (None, MISSING)."""
        result = calculate_cagr(float("nan"), 200, 5)
        assert result.value is None
        assert result.flag == CAGRFlag.MISSING

    def test_nan_end(self) -> None:
        """NaN end → (None, MISSING)."""
        result = calculate_cagr(100, float("nan"), 5)
        assert result.value is None
        assert result.flag == CAGRFlag.MISSING

    def test_turnaround_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """TURNAROUND case must emit a WARNING log."""
        with caplog.at_level(logging.WARNING, logger="src.analytics.cagr"):
            calculate_cagr(-100, 200, 5, metric="revenue", company_id=COMPANY)
        assert "TURNAROUND" in caplog.text

    def test_normal_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Successful CAGR must emit an INFO log."""
        with caplog.at_level(logging.INFO, logger="src.analytics.cagr"):
            calculate_cagr(100, 200, 5, metric="revenue", company_id=COMPANY)
        assert "CAGR computed" in caplog.text


# ===========================================================================
# TEST GROUP 4 — compute_growth_metric
# ===========================================================================


class TestComputeGrowthMetric:
    """Tests for the Series-aware window helper."""

    @pytest.fixture()
    def five_year_series(self) -> pd.Series:
        return pd.Series([100.0, 110.0, 121.0, 133.1, 146.41, 161.05])

    def test_sufficient_3yr(self, five_year_series: pd.Series) -> None:
        """3-year window from index 3 → start=100, end=133.1."""
        result = compute_growth_metric(five_year_series, end_idx=3, years=3)
        assert result.flag == CAGRFlag.NORMAL
        assert result.value == pytest.approx(10.0, abs=0.01)

    def test_sufficient_5yr(self, five_year_series: pd.Series) -> None:
        """5-year window from index 5 → start=100, end=161.05."""
        result = compute_growth_metric(five_year_series, end_idx=5, years=5)
        assert result.flag == CAGRFlag.NORMAL
        assert result.value == pytest.approx(10.0, abs=0.01)

    def test_insufficient_not_enough_history(self, five_year_series: pd.Series) -> None:
        """Requesting 10yr from index 3 (only 3 available) → INSUFFICIENT."""
        result = compute_growth_metric(five_year_series, end_idx=3, years=10)
        assert result.flag == CAGRFlag.INSUFFICIENT
        assert result.value is None

    def test_insufficient_first_row(self, five_year_series: pd.Series) -> None:
        """Index 0 with any years > 0 → INSUFFICIENT."""
        result = compute_growth_metric(five_year_series, end_idx=0, years=3)
        assert result.flag == CAGRFlag.INSUFFICIENT

    def test_insufficient_logs_warning(
        self,
        five_year_series: pd.Series,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """INSUFFICIENT must emit a WARNING log."""
        with caplog.at_level(logging.WARNING, logger="src.analytics.cagr"):
            compute_growth_metric(
                five_year_series, end_idx=2, years=10,
                metric="revenue", company_id=COMPANY, year=YEAR,
            )
        assert "insufficient history" in caplog.text.lower()


# ===========================================================================
# TEST GROUP 5 — compute_company_cagr
# ===========================================================================


class TestComputeCompanyCagr:
    """Tests for the per-company row-by-row orchestrator."""

    def test_returns_one_record_per_row(self, tcs_df: pd.DataFrame) -> None:
        """Number of records must equal number of rows in the group."""
        records = compute_company_cagr(tcs_df)
        assert len(records) == len(tcs_df)

    def test_company_id_set_on_records(self, tcs_df: pd.DataFrame) -> None:
        """All records must carry the correct company_id."""
        records = compute_company_cagr(tcs_df)
        assert all(r.company_id == "TCS" for r in records)

    def test_first_rows_have_insufficient_3yr(
        self, tcs_df: pd.DataFrame
    ) -> None:
        """First 3 rows cannot have a 3-year CAGR."""
        records = compute_company_cagr(tcs_df)
        for rec in records[:3]:
            assert rec.revenue_cagr_3yr is None
            assert rec.revenue_cagr_3yr_flag == str(CAGRFlag.INSUFFICIENT)

    def test_row_with_10yr_history_has_cagr(
        self, tcs_df: pd.DataFrame
    ) -> None:
        """Last row (index 10) has 10 years of history → 10yr CAGR computed."""
        records = compute_company_cagr(tcs_df)
        last = records[-1]
        assert last.revenue_cagr_10yr is not None
        assert last.revenue_cagr_10yr_flag == str(CAGRFlag.NORMAL)

    def test_duplicate_years_handled(self) -> None:
        """Duplicate year rows are deduplicated (keep last)."""
        df = _make_pl_df(
            "DUP",
            ["2020-03", "2020-03", "2021-03", "2022-03", "2023-03"],
            [100, 100, 110, 121, 133],
            [10, 10, 11, 12, 13],
            [5.0, 5.0, 5.5, 6.0, 6.5],
        )
        records = compute_company_cagr(df)
        # After dedup: 4 unique years → 4 records.
        assert len(records) == 4

    def test_unsorted_years_sorted_internally(self) -> None:
        """Rows given in reverse order must produce correct CAGR values."""
        df = _make_pl_df(
            "SORT",
            ["2023-03", "2022-03", "2021-03", "2020-03"],
            [133.1, 121.0, 110.0, 100.0],
            [10.0, 9.0, 8.0, 7.0],
            [5.0, 4.5, 4.0, 3.5],
        )
        records = compute_company_cagr(df)
        # After sort: 2020→2021→2022→2023; last row (2023) 3yr = 100→133.1
        last = records[-1]
        assert last.revenue_cagr_3yr is not None
        assert last.revenue_cagr_3yr == pytest.approx(10.0, abs=0.02)

    def test_all_metric_keys_present(self, tcs_df: pd.DataFrame) -> None:
        """All 9 CAGR value+flag pairs must be attributes of every record."""
        records = compute_company_cagr(tcs_df)
        for rec in records:
            for metric in METRIC_COLUMNS:
                for n in PERIODS:
                    assert hasattr(rec, f"{metric}_cagr_{n}yr")
                    assert hasattr(rec, f"{metric}_cagr_{n}yr_flag")

    def test_pat_turnaround_flag(self) -> None:
        """PAT going from negative to positive must produce TURNAROUND flag."""
        df = _make_pl_df(
            "TURN",
            ["2019-03", "2020-03", "2021-03", "2022-03"],
            [500, 600, 700, 800],
            [-50, 10, 30, 60],  # net_profit: negative then positive
            [1.0, 2.0, 3.0, 4.0],
        )
        records = compute_company_cagr(df)
        last = records[-1]
        # 3yr: start=net_profit[0]=-50, end=net_profit[3]=60 → TURNAROUND
        assert last.pat_cagr_3yr is None
        assert last.pat_cagr_3yr_flag == str(CAGRFlag.TURNAROUND)

    def test_eps_decline_to_loss_flag(self) -> None:
        """EPS going positive → negative must produce DECLINE_TO_LOSS flag."""
        df = _make_pl_df(
            "DECL",
            ["2019-03", "2020-03", "2021-03", "2022-03"],
            [500, 600, 700, 800],
            [50, 60, 70, 80],
            [10.0, 8.0, 5.0, -2.0],  # eps: positive → negative
        )
        records = compute_company_cagr(df)
        last = records[-1]
        assert last.eps_cagr_3yr is None
        assert last.eps_cagr_3yr_flag == str(CAGRFlag.DECLINE_TO_LOSS)


# ===========================================================================
# TEST GROUP 6 — compute_all_cagrs (batch entry point)
# ===========================================================================


class TestComputeAllCagrs:
    """Tests for the batch DataFrame processor."""

    def test_output_schema_matches_output_columns(
        self, tcs_df: pd.DataFrame
    ) -> None:
        """Output DataFrame must have exactly the columns in OUTPUT_COLUMNS."""
        result = compute_all_cagrs(tcs_df)
        assert list(result.columns) == OUTPUT_COLUMNS

    def test_row_count_matches_input(self, tcs_df: pd.DataFrame) -> None:
        """One output row per input row (after dedup/sort)."""
        result = compute_all_cagrs(tcs_df)
        assert len(result) == len(tcs_df)

    def test_multi_company_row_count(
        self, multi_company_df: pd.DataFrame
    ) -> None:
        """Batch processes all companies; total rows = sum of each company."""
        result = compute_all_cagrs(multi_company_df)
        assert len(result) == len(multi_company_df)

    def test_multi_company_ids_present(
        self, multi_company_df: pd.DataFrame
    ) -> None:
        """Both company IDs must appear in the output."""
        result = compute_all_cagrs(multi_company_df)
        assert set(result["company_id"].unique()) == {"TCS", "INFY"}

    def test_empty_dataframe_returns_empty_with_schema(self) -> None:
        """Empty input must return empty DataFrame with correct columns."""
        empty = pd.DataFrame(
            columns=["company_id", "year", "sales", "net_profit", "eps"]
        )
        result = compute_all_cagrs(empty)
        assert result.empty
        assert list(result.columns) == OUTPUT_COLUMNS

    def test_missing_column_raises_value_error(self) -> None:
        """Missing required column must raise ValueError."""
        bad_df = pd.DataFrame({"company_id": ["TCS"], "year": ["2024-03"]})
        with pytest.raises(ValueError, match="missing columns"):
            compute_all_cagrs(bad_df)

    def test_nan_cells_produce_missing_flags(self) -> None:
        """NaN values in metric cells must produce MISSING flags, not raise."""
        df = _make_pl_df(
            "NANTEST",
            ["2020-03", "2021-03", "2022-03", "2023-03"],
            [float("nan"), float("nan"), float("nan"), float("nan")],
            [10, 11, 12, 13],
            [1.0, 1.1, 1.2, 1.3],
        )
        result = compute_all_cagrs(df)
        assert not result.empty
        # All revenue CAGR values must be None / NaN (not computable).
        for col in ["revenue_cagr_3yr", "revenue_cagr_5yr", "revenue_cagr_10yr"]:
            assert result[col].isna().all()

    def test_zero_base_revenue_flag_stored(self) -> None:
        """ZERO_BASE flag must appear in output for companies with zero sales."""
        df = _make_pl_df(
            "ZERO",
            ["2020-03", "2021-03", "2022-03", "2023-03"],
            [0, 100, 110, 121],   # sales[0] == 0 → ZERO_BASE for 3yr at row 3
            [10, 11, 12, 13],
            [1.0, 1.1, 1.2, 1.3],
        )
        result = compute_all_cagrs(df)
        last_row = result[result["year"] == "2023-03"].iloc[0]
        assert last_row["revenue_cagr_3yr_flag"] == str(CAGRFlag.ZERO_BASE)
        assert pd.isna(last_row["revenue_cagr_3yr"])

    def test_cagr_values_rounded_to_2dp(self, tcs_df: pd.DataFrame) -> None:
        """All non-null CAGR values in output must have at most 2 decimal places."""
        result = compute_all_cagrs(tcs_df)
        cagr_value_cols = [c for c in OUTPUT_COLUMNS if c.endswith("yr")
                           and not c.endswith("flag")]
        for col in cagr_value_cols:
            for val in result[col].dropna():
                assert val == round(float(val), 2), (
                    f"Column {col} value {val} not rounded to 2dp"
                )

    def test_output_is_sqlite_compatible_types(
        self, tcs_df: pd.DataFrame
    ) -> None:
        """Flag columns must be strings; value columns must be float or NaN."""
        result = compute_all_cagrs(tcs_df)
        flag_cols = [c for c in OUTPUT_COLUMNS if c.endswith("_flag")]
        value_cols = [c for c in OUTPUT_COLUMNS
                      if c.endswith("yr") and not c.endswith("_flag")]
        for col in flag_cols:
            # After astype(object), dtype is numpy object regardless of pandas version.
            assert result[col].dtype == object, (
                f"{col} should be object dtype"
            )
        for col in value_cols:
            # After explicit astype(float), dtype must be float64.
            assert result[col].dtype == "float64", (
                f"{col} unexpected dtype {result[col].dtype}"
            )

    def test_both_negative_pat_flag(self) -> None:
        """BOTH_NEGATIVE PAT must produce None value and correct flag."""
        df = _make_pl_df(
            "BOTHNEG",
            ["2019-03", "2020-03", "2021-03", "2022-03"],
            [500, 600, 700, 800],
            [-100, -80, -60, -40],   # both negative every year
            [5.0, 6.0, 7.0, 8.0],
        )
        result = compute_all_cagrs(df)
        last = result[result["year"] == "2022-03"].iloc[0]
        assert pd.isna(last["pat_cagr_3yr"])
        assert last["pat_cagr_3yr_flag"] == str(CAGRFlag.BOTH_NEGATIVE)

    def test_insufficient_history_first_rows(
        self, tcs_df: pd.DataFrame
    ) -> None:
        """Rows without enough history must have INSUFFICIENT flags."""
        result = compute_all_cagrs(tcs_df)
        # Row 0: no CAGR is possible for any window.
        row0 = result.iloc[0]
        assert row0["revenue_cagr_3yr_flag"] == str(CAGRFlag.INSUFFICIENT)
        assert row0["revenue_cagr_5yr_flag"] == str(CAGRFlag.INSUFFICIENT)
        assert row0["revenue_cagr_10yr_flag"] == str(CAGRFlag.INSUFFICIENT)

    def test_revenue_cagr_10yr_known_value(
        self, tcs_df: pd.DataFrame
    ) -> None:
        """Spot-check a known 10yr revenue CAGR against manual computation."""
        result = compute_all_cagrs(tcs_df)
        last = result[result["year"] == "2024-03"].iloc[0]
        # sales[0]=86_917, sales[10]=2_40_893, n=10
        expected = round(((2_40_893 / 86_917) ** 0.1 - 1) * 100, 2)
        assert last["revenue_cagr_10yr"] == pytest.approx(expected, rel=1e-4)

    def test_eps_cagr_5yr_known_value(
        self, tcs_df: pd.DataFrame
    ) -> None:
        """Spot-check a known 5yr EPS CAGR against manual computation."""
        result = compute_all_cagrs(tcs_df)
        last = result[result["year"] == "2024-03"].iloc[0]
        # eps[5]=159.82 (2019), eps[10]=125.18 (2024), n=5
        expected = round(((125.18 / 159.82) ** 0.2 - 1) * 100, 2)
        assert last["eps_cagr_5yr"] == pytest.approx(expected, rel=1e-4)

    def test_exception_in_company_group_does_not_crash_batch(
        self,
    ) -> None:
        """If compute_company_cagr raises for one company the batch continues."""
        import unittest.mock as mock

        good_df = _make_pl_df(
            "GOOD",
            ["2021-03", "2022-03", "2023-03", "2024-03"],
            [100, 110, 121, 133],
            [10, 11, 12, 13],
            [1.0, 1.1, 1.2, 1.3],
        )
        bad_df = _make_pl_df(
            "BAD",
            ["2021-03", "2022-03"],
            [100, 110],
            [10, 11],
            [1.0, 1.1],
        )
        combined = pd.concat([good_df, bad_df], ignore_index=True)

        original = compute_company_cagr.__wrapped__ if hasattr(
            compute_company_cagr, "__wrapped__"
        ) else compute_company_cagr

        def side_effect(group: pd.DataFrame):
            cid = group["company_id"].iloc[0]
            if cid == "BAD":
                raise RuntimeError("Simulated failure")
            return original(group)

        with mock.patch(
            "src.analytics.cagr.compute_company_cagr", side_effect=side_effect
        ):
            result = compute_all_cagrs(combined)

        # GOOD company results should still be present.
        assert "GOOD" in result["company_id"].values

    def test_all_companies_raise_returns_empty_with_schema(self) -> None:
        """If all groups raise, result is empty DataFrame with correct columns."""
        import unittest.mock as mock

        df = _make_pl_df(
            "ERR",
            ["2021-03", "2022-03"],
            [100, 110],
            [10, 11],
            [1.0, 1.1],
        )
        with mock.patch(
            "src.analytics.cagr.compute_company_cagr",
            side_effect=RuntimeError("fail"),
        ):
            result = compute_all_cagrs(df)

        assert result.empty
        assert list(result.columns) == OUTPUT_COLUMNS