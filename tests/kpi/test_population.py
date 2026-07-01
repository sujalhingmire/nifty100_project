from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root (parent of "tests" and "src") is importable, regardless
# of whether this file is run via pytest, `python -m`, or directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from src.analytics.populate_financial_ratios import (
    EXPECTED_MIN_ROWS,
    QUALITY_WEIGHTS,
    TABLE_NAME,
    ValidationReport,
    _configure_file_logging,
    _normalize_year,
    _winsorize_and_scale,
    calculate_quality_score,
    compute_all_kpis,
    generate_validation_report,
    load_data,
    merge_financial_data,
    run_pipeline,
    save_to_sqlite,
    validate_dataframe,
    verify_database,
)

# ---------------------------------------------------------------------------
# Fixtures — synthetic raw datasets (fast, deterministic, no I/O dependency)
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_pl() -> pd.DataFrame:
    """8-year P&L history for two companies, normalised year strings."""
    years = [
        "2017-03", "2018-03", "2019-03", "2020-03",
        "2021-03", "2022-03", "2023-03", "2024-03",
    ]
    rows = []
    for cid, base_sales, base_profit in (("TCS", 100_000, 20_000), ("INFY", 60_000, 12_000)):
        for i, yr in enumerate(years):
            rows.append({
                "company_id": cid,
                "year": yr,
                "sales": base_sales * (1.10 ** i),
                "expenses": base_sales * (1.10 ** i) * 0.7,
                "operating_profit": base_sales * (1.10 ** i) * 0.3,
                "opm_percentage": 30.0,
                "other_income": 500.0,
                "interest": 100.0,
                "depreciation": 1000.0,
                "profit_before_tax": base_profit * (1.10 ** i),
                "tax_percentage": 25.0,
                "net_profit": base_profit * (1.10 ** i),
                "eps": 50.0 * (1.10 ** i),
                "dividend_payout": 40.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_bs(synthetic_pl: pd.DataFrame) -> pd.DataFrame:
    """Balance sheet aligned to synthetic_pl years/companies."""
    rows = []
    for cid, year in synthetic_pl[["company_id", "year"]].drop_duplicates().itertuples(index=False):
        rows.append({
            "company_id": cid,
            "year": year,
            "equity_capital": 100.0,
            "reserves": 50_000.0,
            "borrowings": 5_000.0,
            "other_liabilities": 10_000.0,
            "total_liabilities": 65_100.0,
            "fixed_assets": 30_000.0,
            "cwip": 1_000.0,
            "investments": 2_000.0,
            "other_asset": 32_100.0,
            "total_assets": 65_100.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_cf(synthetic_pl: pd.DataFrame) -> pd.DataFrame:
    """Cash flow aligned to synthetic_pl years/companies."""
    rows = []
    for cid, year in synthetic_pl[["company_id", "year"]].drop_duplicates().itertuples(index=False):
        rows.append({
            "company_id": cid,
            "year": year,
            "operating_activity": 18_000.0,
            "investing_activity": -5_000.0,
            "financing_activity": -3_000.0,
            "net_cash_flow": 10_000.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_sectors() -> pd.DataFrame:
    """Sector mapping for both synthetic companies."""
    return pd.DataFrame({
        "company_id": ["TCS", "INFY"],
        "broad_sector": ["Information Technology", "Information Technology"],
    })


@pytest.fixture()
def synthetic_companies() -> pd.DataFrame:
    """Minimal companies table with face_value for book-value calc."""
    return pd.DataFrame({
        "company_id": ["TCS", "INFY"],
        "face_value": [1.0, 5.0],
        "book_value": [150.0, 200.0],
    })


@pytest.fixture()
def synthetic_datasets(
    synthetic_pl: pd.DataFrame,
    synthetic_bs: pd.DataFrame,
    synthetic_cf: pd.DataFrame,
    synthetic_sectors: pd.DataFrame,
    synthetic_companies: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Full dataset dict matching the load_data() return shape."""
    return {
        "profitandloss": synthetic_pl,
        "balancesheet": synthetic_bs,
        "cashflow": synthetic_cf,
        "companies": synthetic_companies,
        "sectors": synthetic_sectors,
        "financial_ratios": pd.DataFrame(),
    }


@pytest.fixture()
def merged_df(synthetic_datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Convenience fixture: already-merged DataFrame."""
    return merge_financial_data(synthetic_datasets)


@pytest.fixture()
def raw_excel_dir(
    tmp_path: Path,
    synthetic_pl: pd.DataFrame,
    synthetic_bs: pd.DataFrame,
    synthetic_cf: pd.DataFrame,
    synthetic_sectors: pd.DataFrame,
    synthetic_companies: pd.DataFrame,
) -> Path:
    """Write synthetic datasets to .xlsx files mimicking the real raw layout.

    Core files get an extra metadata row at index 0 (header=1 convention);
    supplementary files use header=0.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    def _write_core(df: pd.DataFrame, name: str) -> None:
        # Prepend a metadata row so header=1 reads correctly.
        path = raw_dir / f"{name}.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, startrow=1, header=True)
            # Write a metadata row at row 0.
            meta = pd.DataFrame([["METADATA"] * len(df.columns)], columns=df.columns)
            meta.to_excel(writer, index=False, header=False, startrow=0)

    _write_core(synthetic_pl, "profitandloss")
    _write_core(synthetic_bs, "balancesheet")
    _write_core(synthetic_cf, "cashflow")
    _write_core(synthetic_companies.assign(id=synthetic_companies["company_id"]), "companies")

    synthetic_sectors.to_excel(raw_dir / "sectors.xlsx", index=False)

    return raw_dir


# ===========================================================================
# TEST GROUP 1 — _normalize_year
# ===========================================================================


class TestNormalizeYear:
    """Tests for the internal year-label normaliser."""

    def test_month_name_space_year(self) -> None:
        assert _normalize_year("Mar 2014") == "2014-03"

    def test_month_name_dash_year(self) -> None:
        assert _normalize_year("Mar-2014") == "2014-03"

    def test_month_short_dash_2digit_year(self) -> None:
        assert _normalize_year("Mar-23") == "2023-03"

    def test_already_normalised_passthrough(self) -> None:
        assert _normalize_year("2023-03") == "2023-03"

    def test_december_year_end(self) -> None:
        assert _normalize_year("Dec 2012") == "2012-12"

    def test_bare_year_assumes_march(self) -> None:
        assert _normalize_year("2023") == "2023-03"

    def test_none_returns_none(self) -> None:
        assert _normalize_year(None) is None

    def test_garbage_returns_none(self) -> None:
        assert _normalize_year("garbage") is None

    def test_empty_string_returns_none(self) -> None:
        assert _normalize_year("") is None


# ===========================================================================
# TEST GROUP 2 — _winsorize_and_scale
# ===========================================================================


class TestWinsorizeAndScale:
    """Tests for the internal scaling helper used by the quality score."""

    def test_basic_scaling_range(self) -> None:
        series = pd.Series([0.0, 25.0, 50.0, 75.0, 100.0])
        scaled = _winsorize_and_scale(series)
        assert scaled.min() >= 0.0
        assert scaled.max() <= 100.0

    def test_invert_reverses_order(self) -> None:
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        normal = _winsorize_and_scale(series)
        inverted = _winsorize_and_scale(series, invert=True)
        # Highest raw value should score lowest when inverted.
        assert inverted.iloc[-1] < inverted.iloc[0]
        assert normal.iloc[-1] > normal.iloc[0]

    def test_all_nan_returns_all_nan(self) -> None:
        series = pd.Series([np.nan, np.nan, np.nan])
        scaled = _winsorize_and_scale(series)
        assert scaled.isna().all()

    def test_nan_preserved_at_same_positions(self) -> None:
        series = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])
        scaled = _winsorize_and_scale(series)
        assert scaled.isna().iloc[1]
        assert not scaled.isna().iloc[0]

    def test_constant_series_returns_neutral_score(self) -> None:
        series = pd.Series([5.0, 5.0, 5.0, 5.0])
        scaled = _winsorize_and_scale(series)
        assert (scaled == 50.0).all()


# ===========================================================================
# TEST GROUP 3 — load_data
# ===========================================================================


class TestLoadData:
    """Tests for the raw-source loader."""

    def test_loads_all_required_keys(self, raw_excel_dir: Path) -> None:
        datasets = load_data(raw_excel_dir)
        for key in ("profitandloss", "balancesheet", "cashflow", "companies", "sectors"):
            assert key in datasets

    def test_missing_optional_financial_ratios_returns_empty(
        self, raw_excel_dir: Path
    ) -> None:
        datasets = load_data(raw_excel_dir)
        assert datasets["financial_ratios"].empty

    def test_missing_required_file_raises(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            load_data(empty_dir)

    def test_loaded_profitandloss_has_expected_columns(
        self, raw_excel_dir: Path
    ) -> None:
        datasets = load_data(raw_excel_dir)
        pl = datasets["profitandloss"]
        assert "company_id" in pl.columns
        assert "sales" in pl.columns
        assert "net_profit" in pl.columns


# ===========================================================================
# TEST GROUP 4 — merge_financial_data
# ===========================================================================


class TestMergeFinancialData:
    """Tests for the dataset-merging function."""

    def test_merge_success_row_count(
        self, synthetic_datasets: dict[str, pd.DataFrame]
    ) -> None:
        merged = merge_financial_data(synthetic_datasets)
        assert len(merged) == 16  # 2 companies x 8 years

    def test_merge_contains_required_columns(
        self, merged_df: pd.DataFrame
    ) -> None:
        required = {
            "company_id", "year", "sales", "net_profit", "eps",
            "operating_activity", "investing_activity", "financing_activity",
            "equity_capital", "reserves", "borrowings", "total_assets",
            "broad_sector",
        }
        assert required.issubset(set(merged_df.columns))

    def test_merge_normalises_company_id_case(self) -> None:
        pl = pd.DataFrame({
            "company_id": [" tcs "], "year": ["2023-03"], "sales": [100],
            "expenses": [70], "operating_profit": [30], "opm_percentage": [30],
            "other_income": [0], "interest": [0], "depreciation": [0],
            "profit_before_tax": [30], "tax_percentage": [25],
            "net_profit": [20], "eps": [10], "dividend_payout": [40],
        })
        bs = pd.DataFrame({
            "company_id": ["TCS"], "year": ["2023-03"], "equity_capital": [100],
            "reserves": [500], "borrowings": [0], "other_liabilities": [0],
            "total_liabilities": [600], "fixed_assets": [300], "cwip": [0],
            "investments": [0], "other_asset": [300], "total_assets": [600],
        })
        cf = pd.DataFrame({
            "company_id": ["TCS"], "year": ["2023-03"],
            "operating_activity": [50], "investing_activity": [-20],
            "financing_activity": [-10], "net_cash_flow": [20],
        })
        merged = merge_financial_data({"profitandloss": pl, "balancesheet": bs, "cashflow": cf})
        assert merged["company_id"].iloc[0] == "TCS"

    def test_merge_missing_required_key_raises(self) -> None:
        with pytest.raises(KeyError):
            merge_financial_data({"profitandloss": pd.DataFrame()})

    def test_merge_deduplicates_company_year(self) -> None:
        pl = pd.DataFrame({
            "company_id": ["TCS", "TCS"], "year": ["2023-03", "2023-03"],
            "sales": [100, 999], "expenses": [70, 70], "operating_profit": [30, 30],
            "opm_percentage": [30, 30], "other_income": [0, 0], "interest": [0, 0],
            "depreciation": [0, 0], "profit_before_tax": [30, 30],
            "tax_percentage": [25, 25], "net_profit": [20, 20], "eps": [10, 10],
            "dividend_payout": [40, 40],
        })
        bs = pd.DataFrame({
            "company_id": ["TCS"], "year": ["2023-03"], "equity_capital": [100],
            "reserves": [500], "borrowings": [0], "other_liabilities": [0],
            "total_liabilities": [600], "fixed_assets": [300], "cwip": [0],
            "investments": [0], "other_asset": [300], "total_assets": [600],
        })
        cf = pd.DataFrame({
            "company_id": ["TCS"], "year": ["2023-03"],
            "operating_activity": [50], "investing_activity": [-20],
            "financing_activity": [-10], "net_cash_flow": [20],
        })
        merged = merge_financial_data({"profitandloss": pl, "balancesheet": bs, "cashflow": cf})
        assert len(merged) == 1
        # 'keep last' semantics -> sales should be the second row's value.
        assert merged["sales"].iloc[0] == 999


# ===========================================================================
# TEST GROUP 5 — compute_all_kpis (delegation, no duplicated formulas)
# ===========================================================================


class TestComputeAllKpis:
    """Tests verifying compute_all_kpis correctly delegates to the 3 engines."""

    def test_returns_nonempty_dataframe(self, merged_df: pd.DataFrame) -> None:
        kpis = compute_all_kpis(merged_df)
        assert not kpis.empty

    def test_row_count_matches_merged_input(self, merged_df: pd.DataFrame) -> None:
        kpis = compute_all_kpis(merged_df)
        assert len(kpis) == len(merged_df)

    def test_contains_all_required_kpi_columns(self, merged_df: pd.DataFrame) -> None:
        kpis = compute_all_kpis(merged_df)
        required = {
            "net_profit_margin_pct", "operating_profit_margin_pct",
            "return_on_equity_pct", "debt_to_equity", "interest_coverage",
            "asset_turnover", "free_cash_flow_cr", "capex_cr",
            "earnings_per_share", "book_value_per_share",
            "dividend_payout_ratio_pct", "total_debt_cr",
            "cash_from_operations_cr", "revenue_cagr_5yr", "pat_cagr_5yr",
            "eps_cagr_5yr",
        }
        assert required.issubset(set(kpis.columns))

    def test_empty_input_returns_empty_output(self) -> None:
        kpis = compute_all_kpis(pd.DataFrame())
        assert kpis.empty

    def test_net_profit_margin_matches_manual_calc(
        self, merged_df: pd.DataFrame
    ) -> None:
        """Cross-check delegated NPM against a manual calculation (no duplicate
        formula — this asserts ratios.py was actually called correctly)."""
        kpis = compute_all_kpis(merged_df)
        tcs_first = merged_df[
            (merged_df["company_id"] == "TCS") & (merged_df["year"] == "2017-03")
        ].iloc[0]
        expected_npm = (tcs_first["net_profit"] / tcs_first["sales"]) * 100.0

        kpi_row = kpis[
            (kpis["company_id"] == "TCS") & (kpis["year"] == "2017-03")
        ].iloc[0]
        assert kpi_row["net_profit_margin_pct"] == pytest.approx(expected_npm, rel=1e-6)

    def test_revenue_cagr_5yr_populated_for_later_years(
        self, merged_df: pd.DataFrame
    ) -> None:
        """8 years of history -> rows from year index 5 onward should have a
        non-null 5yr revenue CAGR (delegated to cagr.py)."""
        kpis = compute_all_kpis(merged_df)
        tcs_rows = kpis[kpis["company_id"] == "TCS"].sort_values("year")
        last_row = tcs_rows.iloc[-1]
        assert pd.notna(last_row["revenue_cagr_5yr"])

    def test_capital_allocation_pattern_present(
        self, merged_df: pd.DataFrame
    ) -> None:
        kpis = compute_all_kpis(merged_df)
        assert "capital_allocation_pattern" in kpis.columns
        assert kpis["capital_allocation_pattern"].notna().any()


# ===========================================================================
# TEST GROUP 6 — calculate_quality_score
# ===========================================================================


class TestCalculateQualityScore:
    """Tests for the composite quality score."""

    @pytest.fixture()
    def kpi_df(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        return compute_all_kpis(merged_df)

    def test_score_in_valid_range(self, kpi_df: pd.DataFrame) -> None:
        scores = calculate_quality_score(kpi_df)
        valid = scores.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 100.0).all()

    def test_score_aligned_to_input_index(self, kpi_df: pd.DataFrame) -> None:
        scores = calculate_quality_score(kpi_df)
        assert list(scores.index) == list(kpi_df.index)

    def test_weights_sum_to_one(self) -> None:
        assert sum(QUALITY_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-9)

    def test_missing_all_metric_columns_returns_all_nan(self) -> None:
        df = pd.DataFrame({"company_id": ["A", "B"], "year": ["2023-03", "2023-03"]})
        scores = calculate_quality_score(df)
        assert scores.isna().all()

    def test_partial_metrics_still_produce_score(self) -> None:
        """When only some metric columns exist, available weights should be
        re-normalised rather than producing all-NaN output."""
        df = pd.DataFrame({
            "return_on_equity_pct": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        scores = calculate_quality_score(df)
        assert scores.notna().any()


# ===========================================================================
# TEST GROUP 7 — validate_dataframe
# ===========================================================================


class TestValidateDataframe:
    """Tests for the pre-insert validation function."""

    def test_row_count_check_passes_above_threshold(self) -> None:
        df = pd.DataFrame({
            "company_id": ["X"] * (EXPECTED_MIN_ROWS + 10),
            "year": [f"{2000 + i}-03" for i in range(EXPECTED_MIN_ROWS + 10)],
        })
        report = validate_dataframe(df)
        assert report.row_count_ok is True

    def test_row_count_check_fails_below_threshold(self) -> None:
        df = pd.DataFrame({"company_id": ["X"] * 5, "year": ["2023-03"] * 5})
        report = validate_dataframe(df)
        assert report.row_count_ok is False
        assert report.passed is False

    def test_duplicate_detection(self) -> None:
        df = pd.DataFrame({
            "company_id": ["TCS", "TCS", "INFY"],
            "year": ["2023-03", "2023-03", "2023-03"],
        })
        report = validate_dataframe(df)
        assert report.duplicate_count == 2
        assert report.duplicates_ok is False

    def test_no_duplicates_passes(self) -> None:
        df = pd.DataFrame({
            "company_id": ["TCS", "INFY"], "year": ["2023-03", "2023-03"],
        })
        report = validate_dataframe(df)
        assert report.duplicates_ok is True

    def test_null_only_column_detected(self) -> None:
        df = pd.DataFrame({
            "company_id": ["TCS"], "year": ["2023-03"],
            "net_profit_margin_pct": [np.nan],
        })
        report = validate_dataframe(df)
        assert "net_profit_margin_pct" in report.null_only_columns

    def test_quality_score_stats_computed(self) -> None:
        df = pd.DataFrame({
            "company_id": ["A", "B", "C"], "year": ["2023-03"] * 3,
            "composite_quality_score": [50.0, 60.0, 70.0],
        })
        report = validate_dataframe(df)
        assert report.quality_score_stats["mean"] == pytest.approx(60.0)


# ===========================================================================
# TEST GROUP 8 — save_to_sqlite / verify_database (full DB round trip)
# ===========================================================================


class TestDatabasePersistence:
    """Tests for SQLite persistence via SQLAlchemy."""

    @pytest.fixture()
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "test_nifty100.db"

    @pytest.fixture()
    def sample_kpi_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "company_id": ["TCS", "INFY", "HDFC"],
            "year": ["2023-03", "2023-03", "2023-03"],
            "net_profit_margin_pct": [20.0, 18.0, 15.0],
            "composite_quality_score": [80.0, 70.0, 60.0],
        })

    def test_database_connection_creates_file(
        self, db_path: Path, sample_kpi_df: pd.DataFrame
    ) -> None:
        save_to_sqlite(sample_kpi_df, db_path=db_path)
        assert db_path.exists()

    def test_insert_success_row_count(
        self, db_path: Path, sample_kpi_df: pd.DataFrame
    ) -> None:
        n_inserted = save_to_sqlite(sample_kpi_df, db_path=db_path)
        assert n_inserted == 3

        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar_one()
        assert count == 3

    def test_duplicate_handling_replaces_not_appends(
        self, db_path: Path, sample_kpi_df: pd.DataFrame
    ) -> None:
        save_to_sqlite(sample_kpi_df, db_path=db_path)
        # Re-insert the same (company_id, year) rows with different values.
        updated = sample_kpi_df.copy()
        updated["net_profit_margin_pct"] = [99.0, 99.0, 99.0]
        save_to_sqlite(updated, db_path=db_path)

        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar_one()
            tcs_value = conn.execute(
                text(f"SELECT net_profit_margin_pct FROM {TABLE_NAME} "
                     f"WHERE company_id='TCS' AND year='2023-03'")
            ).scalar_one()
        assert count == 3  # not 6 — old rows were replaced, not appended.
        assert tcs_value == pytest.approx(99.0)

    def test_save_empty_dataframe_raises(self, db_path: Path) -> None:
        with pytest.raises(ValueError):
            save_to_sqlite(pd.DataFrame(), db_path=db_path)

    def test_verify_database_row_count(
        self, db_path: Path, sample_kpi_df: pd.DataFrame
    ) -> None:
        save_to_sqlite(sample_kpi_df, db_path=db_path)
        report = verify_database(db_path=db_path)
        assert report.total_rows == 3

    def test_verify_database_detects_duplicates(
        self, db_path: Path, sample_kpi_df: pd.DataFrame
    ) -> None:
        save_to_sqlite(sample_kpi_df, db_path=db_path)

        # Manually insert a true duplicate to simulate corruption, bypassing
        # the replace-before-insert logic in save_to_sqlite.
        engine = create_engine(f"sqlite:///{db_path}")
        dup_row = sample_kpi_df.iloc[[0]]
        dup_row.to_sql(TABLE_NAME, engine, if_exists="append", index=False)

        report = verify_database(db_path=db_path)
        assert report.duplicate_count > 0
        assert report.duplicates_ok is False

    def test_verify_database_below_min_rows_fails(
        self, db_path: Path, sample_kpi_df: pd.DataFrame
    ) -> None:
        save_to_sqlite(sample_kpi_df, db_path=db_path)
        report = verify_database(db_path=db_path)
        assert report.row_count_ok is False  # 3 rows << EXPECTED_MIN_ROWS
        assert report.passed is False


# ===========================================================================
# TEST GROUP 9 — generate_validation_report
# ===========================================================================


class TestGenerateValidationReport:
    """Tests for the human-readable report writer."""

    def test_report_file_created(self, tmp_path: Path) -> None:
        report = ValidationReport(
            total_rows=1200, row_count_ok=True, duplicate_count=0,
            duplicates_ok=True, passed=True,
        )
        out_path = tmp_path / "validation.txt"
        result_path = generate_validation_report(report, output_path=out_path)
        assert Path(result_path).exists()

    def test_report_contains_pass_result(self, tmp_path: Path) -> None:
        report = ValidationReport(
            total_rows=1200, row_count_ok=True, duplicate_count=0,
            duplicates_ok=True, passed=True,
        )
        out_path = tmp_path / "validation.txt"
        generate_validation_report(report, output_path=out_path)
        content = out_path.read_text(encoding="utf-8")
        assert "OVERALL VALIDATION RESULT : PASS" in content

    def test_report_contains_fail_result(self, tmp_path: Path) -> None:
        report = ValidationReport(
            total_rows=5, row_count_ok=False, duplicate_count=2,
            duplicates_ok=False, passed=False,
        )
        out_path = tmp_path / "validation.txt"
        generate_validation_report(report, output_path=out_path)
        content = out_path.read_text(encoding="utf-8")
        assert "OVERALL VALIDATION RESULT : FAIL" in content

    def test_report_includes_elapsed_time_when_given(self, tmp_path: Path) -> None:
        report = ValidationReport(total_rows=1200, row_count_ok=True, passed=True)
        out_path = tmp_path / "validation.txt"
        generate_validation_report(report, output_path=out_path, elapsed_seconds=4.567)
        content = out_path.read_text(encoding="utf-8")
        assert "4.567" in content

    def test_report_lists_null_only_columns(self, tmp_path: Path) -> None:
        report = ValidationReport(
            total_rows=1200, row_count_ok=True, duplicates_ok=True,
            null_only_columns=["revenue_cagr_5yr"], passed=True,
        )
        out_path = tmp_path / "validation.txt"
        generate_validation_report(report, output_path=out_path)
        content = out_path.read_text(encoding="utf-8")
        assert "revenue_cagr_5yr" in content


# ===========================================================================
# TEST GROUP 10 — Full pipeline integration (synthetic, in-memory)
# ===========================================================================


class TestFullPipelineIntegration:
    """End-to-end test wiring load -> merge -> compute -> score -> validate ->
    save -> verify together, using synthetic data and a temp SQLite file."""

    def test_full_pipeline_round_trip(
        self,
        synthetic_datasets: dict[str, pd.DataFrame],
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        merged = merge_financial_data(synthetic_datasets)
        kpis = compute_all_kpis(merged)
        kpis["composite_quality_score"] = calculate_quality_score(kpis)

        db_path = tmp_path / "integration.db"

        # Lower the row-count expectation for this small synthetic dataset
        # by validating manually rather than relying on the module constant.
        n_inserted = save_to_sqlite(kpis, db_path=db_path)
        assert n_inserted == len(kpis)

        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar_one()
        assert count == 16  # 2 companies x 8 years

    def test_pipeline_produces_no_duplicate_keys(
        self, synthetic_datasets: dict[str, pd.DataFrame], tmp_path: Path
    ) -> None:
        merged = merge_financial_data(synthetic_datasets)
        kpis = compute_all_kpis(merged)
        kpis["composite_quality_score"] = calculate_quality_score(kpis)

        report = validate_dataframe(kpis)
        assert report.duplicates_ok is True

    def test_pipeline_no_kpi_column_entirely_null(
        self, synthetic_datasets: dict[str, pd.DataFrame]
    ) -> None:
        merged = merge_financial_data(synthetic_datasets)
        kpis = compute_all_kpis(merged)
        kpis["composite_quality_score"] = calculate_quality_score(kpis)

        report = validate_dataframe(kpis)
        # Core profitability/leverage columns must have at least some data;
        # only CAGR columns (which need 5+ years) might legitimately have
        # partial coverage in an 8-year synthetic window, never zero.
        for col in ("net_profit_margin_pct", "return_on_equity_pct", "debt_to_equity"):
            assert col not in report.null_only_columns


# ===========================================================================
# TEST GROUP 11 — _configure_file_logging
# ===========================================================================


class TestConfigureFileLogging:
    """Tests for the rotating file-handler attachment helper."""

    def test_creates_log_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "sub" / "ratio_engine.log"
        _configure_file_logging(log_path)
        logging.getLogger("src.analytics.populate_financial_ratios").info("probe")
        assert log_path.exists()

    def test_idempotent_does_not_duplicate_handler(self, tmp_path: Path) -> None:
        log_path = tmp_path / "ratio_engine.log"
        _configure_file_logging(log_path)
        handlers_before = len(logging.getLogger().handlers)
        _configure_file_logging(log_path)
        handlers_after = len(logging.getLogger().handlers)
        assert handlers_after == handlers_before


# ===========================================================================
# TEST GROUP 12 — verify_database edge cases
# ===========================================================================


class TestVerifyDatabaseEdgeCases:
    """Tests for verify_database branches not covered by happy-path tests."""

    def test_verify_database_missing_kpi_column_in_schema(
        self, tmp_path: Path
    ) -> None:
        """When the persisted table lacks a KPI column entirely (e.g. an older
        schema version), verify_database must report it as null_only rather
        than raising."""
        db_path = tmp_path / "narrow_schema.db"
        narrow_df = pd.DataFrame({
            "company_id": ["TCS"], "year": ["2023-03"],
            "net_profit_margin_pct": [20.0],
        })
        save_to_sqlite(narrow_df, db_path=db_path)
        report = verify_database(db_path=db_path)
        assert "debt_to_equity" in report.null_only_columns

    def test_verify_database_quality_score_stats_when_present(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "with_scores.db"
        df = pd.DataFrame({
            "company_id": ["A", "B", "C"] * 400,
            "year": [f"{2000+i}-03" for i in range(1200)],
            "composite_quality_score": [50.0 + (i % 30) for i in range(1200)],
        })
        save_to_sqlite(df, db_path=db_path)
        report = verify_database(db_path=db_path)
        assert report.quality_score_stats
        assert "mean" in report.quality_score_stats


# ===========================================================================
# TEST GROUP 13 — run_pipeline (top-level orchestrator)
# ===========================================================================


class TestRunPipeline:
    """Tests for the full run_pipeline() entry point."""

    def test_run_pipeline_empty_raw_dir_returns_failed_report(
        self, tmp_path: Path
    ) -> None:
        """A raw_dir missing required files should raise FileNotFoundError
        from load_data, which propagates out of run_pipeline."""
        empty_dir = tmp_path / "empty_raw"
        empty_dir.mkdir()
        output_dir = tmp_path / "output"
        db_path = tmp_path / "out.db"

        with pytest.raises(FileNotFoundError):
            run_pipeline(
                raw_dir=empty_dir,
                db_path=db_path,
                output_dir=output_dir,
            )

    def test_run_pipeline_full_synthetic_round_trip(
        self, raw_excel_dir: Path, tmp_path: Path
    ) -> None:
        """Full pipeline against synthetic .xlsx files on disk — exercises
        load_data -> merge -> compute -> score -> validate -> save -> verify
        -> report, all in one call."""
        db_path = tmp_path / "pipeline.db"
        output_dir = tmp_path / "output"

        report = run_pipeline(
            raw_dir=raw_excel_dir,
            db_path=db_path,
            output_dir=output_dir,
        )

        assert report.total_rows == 16  # 2 companies x 8 years
        assert (output_dir / "database_validation_report.txt").exists()
        assert (output_dir / "ratio_engine.log").exists()


# ===========================================================================
# TEST GROUP 14 - Remaining branch coverage
# ===========================================================================


class TestRemainingBranchCoverage:
    """Targeted tests closing the final coverage gaps."""

    def test_normalize_year_unrecognised_month_returns_none(self) -> None:
        """A 2-part string whose first token is not a valid month abbreviation
        must return None (covers the month_num is None branch)."""
        assert _normalize_year("Xyz 2023") is None

    def test_normalize_year_non_digit_year_part_returns_none(self) -> None:
        """A 2-part string whose year token is not numeric must return None."""
        assert _normalize_year("Mar abcd") is None

    def test_load_data_csv_fallback_when_xlsx_absent(self, tmp_path: Path) -> None:
        """When only a .csv file exists (no .xlsx), load_data must read it."""
        raw_dir = tmp_path / "csv_raw"
        raw_dir.mkdir()

        pd.DataFrame({
            "company_id": ["TCS"], "year": ["2023-03"], "sales": [100],
            "expenses": [70], "operating_profit": [30], "opm_percentage": [30],
            "other_income": [0], "interest": [0], "depreciation": [0],
            "profit_before_tax": [30], "tax_percentage": [25],
            "net_profit": [20], "eps": [10], "dividend_payout": [40],
        }).to_csv(raw_dir / "profitandloss.csv", index=False)

        pd.DataFrame({
            "company_id": ["TCS"], "year": ["2023-03"], "equity_capital": [100],
            "reserves": [500], "borrowings": [0], "other_liabilities": [0],
            "total_liabilities": [600], "fixed_assets": [300], "cwip": [0],
            "investments": [0], "other_asset": [300], "total_assets": [600],
        }).to_csv(raw_dir / "balancesheet.csv", index=False)

        pd.DataFrame({
            "company_id": ["TCS"], "year": ["2023-03"],
            "operating_activity": [50], "investing_activity": [-20],
            "financing_activity": [-10], "net_cash_flow": [20],
        }).to_csv(raw_dir / "cashflow.csv", index=False)

        pd.DataFrame({"company_id": ["TCS"]}).to_csv(raw_dir / "companies.csv", index=False)
        pd.DataFrame({
            "company_id": ["TCS"], "broad_sector": ["Information Technology"],
        }).to_csv(raw_dir / "sectors.csv", index=False)

        datasets = load_data(raw_dir)
        assert len(datasets["profitandloss"]) == 1

    def test_merge_logs_warning_on_unparseable_year_rows(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Rows with unparseable year strings must be dropped with a logged
        warning (covers the n_unparsed branch in merge_financial_data)."""
        pl = pd.DataFrame({
            "company_id": ["TCS", "TCS"], "year": ["2023-03", "not-a-year"],
            "sales": [100, 200], "expenses": [70, 140],
            "operating_profit": [30, 60], "opm_percentage": [30, 30],
            "other_income": [0, 0], "interest": [0, 0], "depreciation": [0, 0],
            "profit_before_tax": [30, 60], "tax_percentage": [25, 25],
            "net_profit": [20, 40], "eps": [10, 20], "dividend_payout": [40, 40],
        })
        bs = pd.DataFrame({
            "company_id": ["TCS"], "year": ["2023-03"], "equity_capital": [100],
            "reserves": [500], "borrowings": [0], "other_liabilities": [0],
            "total_liabilities": [600], "fixed_assets": [300], "cwip": [0],
            "investments": [0], "other_asset": [300], "total_assets": [600],
        })
        cf = pd.DataFrame({
            "company_id": ["TCS"], "year": ["2023-03"],
            "operating_activity": [50], "investing_activity": [-20],
            "financing_activity": [-10], "net_cash_flow": [20],
        })
        with caplog.at_level(
            logging.WARNING, logger="src.analytics.populate_financial_ratios"
        ):
            merged = merge_financial_data(
                {"profitandloss": pl, "balancesheet": bs, "cashflow": cf}
            )
        assert len(merged) == 1
        assert "unparseable year" in caplog.text

    def test_quality_score_logs_warning_when_some_rows_missing_all_metrics(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Covers the n_missing warning branch in calculate_quality_score."""
        df = pd.DataFrame({
            "return_on_equity_pct": [10.0, 20.0, np.nan, 30.0, 40.0],
            "debt_to_equity": [1.0, 2.0, np.nan, 3.0, 4.0],
        })
        with caplog.at_level(
            logging.WARNING, logger="src.analytics.populate_financial_ratios"
        ):
            scores = calculate_quality_score(df)
        assert scores.isna().sum() >= 1
        assert "no usable metrics" in caplog.text

    def test_run_pipeline_empty_merged_dataset_aborts_gracefully(
        self, tmp_path: Path
    ) -> None:
        """When the raw files exist but produce zero mergeable rows (e.g. all
        years unparseable), run_pipeline must abort cleanly and still write a
        FAIL validation report rather than raising."""
        raw_dir = tmp_path / "raw_unparseable"
        raw_dir.mkdir()

        pd.DataFrame({
            "company_id": ["TCS"], "year": ["garbage-year"], "sales": [100],
            "expenses": [70], "operating_profit": [30], "opm_percentage": [30],
            "other_income": [0], "interest": [0], "depreciation": [0],
            "profit_before_tax": [30], "tax_percentage": [25],
            "net_profit": [20], "eps": [10], "dividend_payout": [40],
        }).to_csv(raw_dir / "profitandloss.csv", index=False)

        pd.DataFrame({
            "company_id": ["TCS"], "year": ["garbage-year"], "equity_capital": [100],
            "reserves": [500], "borrowings": [0], "other_liabilities": [0],
            "total_liabilities": [600], "fixed_assets": [300], "cwip": [0],
            "investments": [0], "other_asset": [300], "total_assets": [600],
        }).to_csv(raw_dir / "balancesheet.csv", index=False)

        pd.DataFrame({
            "company_id": ["TCS"], "year": ["garbage-year"],
            "operating_activity": [50], "investing_activity": [-20],
            "financing_activity": [-10], "net_cash_flow": [20],
        }).to_csv(raw_dir / "cashflow.csv", index=False)

        pd.DataFrame({"company_id": ["TCS"]}).to_csv(raw_dir / "companies.csv", index=False)
        pd.DataFrame({
            "company_id": ["TCS"], "broad_sector": ["Information Technology"],
        }).to_csv(raw_dir / "sectors.csv", index=False)

        db_path = tmp_path / "empty_pipeline.db"
        output_dir = tmp_path / "empty_output"

        report = run_pipeline(raw_dir=raw_dir, db_path=db_path, output_dir=output_dir)

        assert report.passed is False
        assert (output_dir / "database_validation_report.txt").exists()


# ===========================================================================
# TEST GROUP 15 - Final coverage closure
# ===========================================================================


class TestFinalCoverageClosure:
    """Tests closing the last remaining branch-coverage gaps."""

    def test_merge_companies_id_column_fallback(self) -> None:
        """When the companies frame has 'id' but not 'company_id', merge
        must derive company_id from 'id' (covers the elif branch in
        merge_financial_data)."""
        pl = pd.DataFrame({
            "company_id": ["TCS"], "year": ["2023-03"], "sales": [100],
            "expenses": [70], "operating_profit": [30], "opm_percentage": [30],
            "other_income": [0], "interest": [0], "depreciation": [0],
            "profit_before_tax": [30], "tax_percentage": [25],
            "net_profit": [20], "eps": [10], "dividend_payout": [40],
        })
        bs = pd.DataFrame({
            "company_id": ["TCS"], "year": ["2023-03"], "equity_capital": [100],
            "reserves": [500], "borrowings": [0], "other_liabilities": [0],
            "total_liabilities": [600], "fixed_assets": [300], "cwip": [0],
            "investments": [0], "other_asset": [300], "total_assets": [600],
        })
        cf = pd.DataFrame({
            "company_id": ["TCS"], "year": ["2023-03"],
            "operating_activity": [50], "investing_activity": [-20],
            "financing_activity": [-10], "net_cash_flow": [20],
        })
        # companies has 'id' but NOT 'company_id' -- exercises the fallback.
        companies = pd.DataFrame({
            "id": [" tcs "], "face_value": [1.0], "book_value": [150.0],
        })

        merged = merge_financial_data({
            "profitandloss": pl, "balancesheet": bs, "cashflow": cf,
            "companies": companies,
        })
        assert not merged.empty
        # face_value should have been joined in via the derived company_id.
        assert merged["face_value"].iloc[0] == pytest.approx(1.0)

    def test_verify_database_column_present_but_all_null(
        self, tmp_path: Path
    ) -> None:
        """When a KPI column exists in the schema but every value is NULL
        (as opposed to the column being entirely absent), verify_database
        must flag it via the total_rows-equals-null-count branch."""
        db_path = tmp_path / "all_null_column.db"
        df = pd.DataFrame({
            "company_id": [f"C{i}" for i in range(1200)],
            "year": ["2023-03"] * 1200,
            "net_profit_margin_pct": [20.0] * 1200,
            "eps_cagr_5yr": [np.nan] * 1200,  # column present, all NULL
        })
        save_to_sqlite(df, db_path=db_path)
        report = verify_database(db_path=db_path)
        assert "eps_cagr_5yr" in report.null_only_columns
        assert report.null_summary["eps_cagr_5yr"] == 1200

    def test_module_main_guard_executes_run_pipeline(self) -> None:
        """Covers the if __name__ == '__main__' guard by running the module
        as __main__ via runpy against the real bundled data fixtures. This
        exercises the exact code path a user hits via
        `python -m src.analytics.populate_financial_ratios`, confirming the
        guard triggers run_pipeline() and produces a database + report."""
        import runpy
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            runpy.run_module(
                "src.analytics.populate_financial_ratios",
                run_name="__main__",
            )

        # The real pipeline runs against data/*.xlsx and output/ under CWD;
        # a successful run produces the validation report as proof the
        # __main__ guard executed run_pipeline() end-to-end.
        assert Path("output/database_validation_report.txt").exists()