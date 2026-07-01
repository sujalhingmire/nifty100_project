from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

_THIS_FILE: Path = Path(__file__).resolve()
# populate_financial_ratios.py lives at <project_root>/src/analytics/...
PROJECT_ROOT: Path = _THIS_FILE.parents[2]

sys.path.append(str(PROJECT_ROOT))

from src.analytics import cagr as cagr_engine
from src.analytics import cashflow_kpis as cf_engine
from src.analytics import ratios as ratios_engine

__all__ = [
    "DB_PATH",
    "TABLE_NAME",
    "EXPECTED_MIN_ROWS",
    "QUALITY_WEIGHTS",
    "ValidationReport",
    "load_data",
    "merge_financial_data",
    "compute_all_kpis",
    "calculate_quality_score",
    "validate_dataframe",
    "save_to_sqlite",
    "verify_database",
    "generate_validation_report",
    "run_pipeline",
]

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path / configuration constants (overridable via environment variables —
# never hardcoded inside business logic).
# ---------------------------------------------------------------------------
_DATA_DIR: Path = Path(os.environ.get("NIFTY_DATA_DIR", str(PROJECT_ROOT / "data")))
_RAW_DIR: Path = Path(os.environ.get("NIFTY_RAW_DIR", str(_DATA_DIR)))
_PROCESSED_DIR: Path = _DATA_DIR / "processed"
_OUTPUT_DIR: Path = Path(os.environ.get("NIFTY_OUTPUT_DIR", str(PROJECT_ROOT / "output")))

DB_PATH: Path = Path(os.environ.get("NIFTY_DB_PATH", str(_DATA_DIR / "nifty100.db")))
TABLE_NAME: str = "financial_ratios"
EXPECTED_MIN_ROWS: int = 1_100

VALIDATION_REPORT_PATH: Path = _OUTPUT_DIR / "database_validation_report.txt"
RATIO_ENGINE_LOG_PATH: Path = _OUTPUT_DIR / "ratio_engine.log"

# Source filenames (accepts .xlsx with header offsets matching the project
# document; .csv with header=0 is also supported transparently).
_SOURCE_FILES: dict[str, str] = {
    "profitandloss": "profitandloss",
    "balancesheet": "balancesheet",
    "cashflow": "cashflow",
    "companies": "companies",
    "sectors": "sectors",
    "financial_ratios": "financial_ratios",  # optional pre-existing extract
}

# Core source files use header=1 per the project data dictionary
# (row 0 is metadata).  Supplementary files use header=0.
_CORE_HEADER_FILES: frozenset[str] = frozenset(
    {"profitandloss", "balancesheet", "cashflow", "companies"}
)

# ---------------------------------------------------------------------------
# Composite Quality Score configuration
# ---------------------------------------------------------------------------

# Reusable weighted score.  Keys are the normalised metric names; values sum
# to 1.0.  Each metric is normalised to [0, 100] before weighting, so the
# composite score formula is purely mechanical — extend this dict to add a
# new factor without touching any function body.
QUALITY_WEIGHTS: dict[str, float] = {
    "roe": 0.20,
    "debt_to_equity": 0.15,          # inverted: lower D/E -> higher score
    "interest_coverage": 0.10,
    "revenue_cagr_5yr": 0.15,
    "pat_cagr_5yr": 0.15,
    "cfo_quality_score": 0.15,
    "fcf_conversion_rate": 0.10,
}

# Winsorisation bounds (percentile) applied before min-max scaling, so a
# handful of extreme outliers cannot dominate the composite score.
_WINSOR_LOW_PCT: float = 0.10
_WINSOR_HIGH_PCT: float = 0.90

_ZERO_TOL: float = 1e-9


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------


def _configure_file_logging(log_path: Path = RATIO_ENGINE_LOG_PATH) -> None:
    """Attach a rotating file handler to the root logger for this run.

    Idempotent: if a handler writing to *log_path* is already attached, a
    duplicate handler is not added.

    Args:
        log_path: Destination path for ``ratio_engine.log``.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()

    for handler in root_logger.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == log_path.resolve()
        ):
            return  # already configured

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Structured outcome of a full database validation pass.

    Attributes:
        total_rows:            Row count in the persisted table.
        row_count_ok:          True when total_rows >= EXPECTED_MIN_ROWS.
        duplicate_count:       Number of duplicate (company_id, year) pairs.
        duplicates_ok:         True when duplicate_count == 0.
        null_only_columns:     Columns where every value is NULL.
        null_summary:          Mapping of column name -> NULL count.
        missing_kpi_counts:    Mapping of KPI column name -> missing count.
        quality_score_stats:   Descriptive statistics for composite_quality_score.
        passed:                Overall pass/fail (all checks green).
    """

    total_rows: int = 0
    row_count_ok: bool = False
    duplicate_count: int = 0
    duplicates_ok: bool = False
    null_only_columns: list[str] = field(default_factory=list)
    null_summary: dict[str, int] = field(default_factory=dict)
    missing_kpi_counts: dict[str, int] = field(default_factory=dict)
    quality_score_stats: dict[str, float] = field(default_factory=dict)
    passed: bool = False


# ---------------------------------------------------------------------------
# Helpers shared across this module
# ---------------------------------------------------------------------------


def _normalize_year(raw_year: object) -> Optional[str]:
    """Normalise a raw year label to the canonical 'YYYY-MM' format.

    Accepts formats observed in the raw source files, e.g.::

        'Mar 2014'  -> '2014-03'
        'Dec 2012'  -> '2012-12'
        'Mar-23'    -> '2023-03'
        '2023-03'   -> '2023-03'   (already normalised; passthrough)
        2023        -> '2023-03'  (bare year -> assume March FY close)

    Args:
        raw_year: Raw value from the source 'year' column.

    Returns:
        Normalised 'YYYY-MM' string, or None when unparseable.
    """
    if raw_year is None:
        return None

    text_val = str(raw_year).strip()
    if not text_val:
        return None

    # Already normalised.
    if len(text_val) == 7 and text_val[4] == "-":
        return text_val

    month_map = {
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "may": "05", "jun": "06", "jul": "07", "aug": "08",
        "sep": "09", "oct": "10", "nov": "11", "dec": "12",
    }

    # 'Mar 2014', 'Mar-2014', 'Mar-23'
    parts = text_val.replace("-", " ").split()
    if len(parts) == 2:
        month_str, year_str = parts
        month_key = month_str.strip().lower()[:3]
        month_num = month_map.get(month_key)
        if month_num is None:
            return None
        year_str = year_str.strip()
        if len(year_str) == 2:
            year_str = f"20{year_str}"
        if not year_str.isdigit():
            return None
        return f"{year_str}-{month_num}"

    # Bare year, e.g. '2023' -> assume March FY close.
    if text_val.isdigit() and len(text_val) == 4:
        return f"{text_val}-03"

    logger.warning("normalize_year: unparseable year value '%s'.", raw_year)
    return None


def _winsorize_and_scale(series: pd.Series, *, invert: bool = False) -> pd.Series:
    """Winsorise a Series at configured percentiles, then min-max scale to [0, 100].

    Args:
        series: Raw numeric values (may contain NaN).
        invert: When True, lower raw values map to higher scaled scores
                (used for metrics where "lower is better", e.g. D/E).

    Returns:
        Series of the same index, scaled to [0, 100]; NaN preserved for
        rows that had NaN input.
    """
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)

    low = valid.quantile(_WINSOR_LOW_PCT)
    high = valid.quantile(_WINSOR_HIGH_PCT)

    clipped = series.clip(lower=low, upper=high)

    value_range = high - low
    if abs(value_range) < _ZERO_TOL:
        # All values effectively identical -> neutral mid-score.
        scaled = pd.Series(50.0, index=series.index)
        scaled[series.isna()] = np.nan
        return scaled

    scaled = (clipped - low) / value_range * 100.0
    if invert:
        scaled = 100.0 - scaled

    scaled[series.isna()] = np.nan
    return scaled.clip(lower=0.0, upper=100.0)


# ---------------------------------------------------------------------------
# 1. load_data
# ---------------------------------------------------------------------------


def _find_source_file(raw_dir: Path, stem: str) -> Optional[Path]:
    """Locate a source file for ``stem``, tolerant of folder and case.

    Looks in ``raw_dir`` itself plus common sibling locations
    (``raw_dir/raw``, ``raw_dir.parent``), and matches filenames
    case-insensitively so e.g. ``ProfitAndLoss.XLSX`` is still found.
    """
    candidate_dirs = [raw_dir, raw_dir / "raw", raw_dir.parent]
    for d in candidate_dirs:
        if not d.exists():
            continue
        for ext in (".xlsx", ".csv"):
            exact = d / f"{stem}{ext}"
            if exact.exists():
                return exact
        # Case-insensitive fallback scan.
        try:
            for f in d.iterdir():
                if f.is_file() and f.stem.lower() == stem.lower() and f.suffix.lower() in (".xlsx", ".csv"):
                    return f
        except OSError:
            continue
    return None


def load_data(raw_dir: Path = _RAW_DIR) -> dict[str, pd.DataFrame]:
    """Load all raw source datasets required for the ratio engine.

    Supports both ``.xlsx`` (with the header-row offsets documented in the
    project data dictionary) and ``.csv`` (header=0) transparently — whichever
    extension is found on disk takes precedence, with ``.xlsx`` checked first.
    Filenames are matched case-insensitively, and ``raw_dir``, ``raw_dir/raw``,
    and ``raw_dir.parent`` are all searched.

    The ``financial_ratios`` source is optional; when absent an empty
    DataFrame is returned for that key and a WARNING is logged.

    Args:
        raw_dir: Directory containing the raw source files.

    Returns:
        Dictionary mapping logical dataset name -> loaded DataFrame.

    Raises:
        FileNotFoundError: When a *required* source file is missing.
    """
    datasets: dict[str, pd.DataFrame] = {}

    for key, stem in _SOURCE_FILES.items():
        header_row = 1 if key in _CORE_HEADER_FILES else 0
        found = _find_source_file(raw_dir, stem)

        if found is not None and found.suffix.lower() == ".xlsx":
            df = pd.read_excel(found, header=header_row)
            logger.info("load_data: loaded %s (%d rows) from %s.", key, len(df), found)
        elif found is not None and found.suffix.lower() == ".csv":
            df = pd.read_csv(found, header=0)
            logger.info("load_data: loaded %s (%d rows) from %s.", key, len(df), found)
        else:
            if key == "financial_ratios":
                logger.warning(
                    "load_data: optional source '%s' not found; "
                    "continuing without pre-existing extract.",
                    key,
                )
                df = pd.DataFrame()
            else:
                raise FileNotFoundError(
                    f"load_data: required source '{key}' not found. Looked in "
                    f"{raw_dir}, {raw_dir / 'raw'}, and {raw_dir.parent} "
                    f"for '{stem}.xlsx' or '{stem}.csv' (any case)."
                )

        datasets[key] = df

    return datasets


# ---------------------------------------------------------------------------
# 2. merge_financial_data
# ---------------------------------------------------------------------------


def merge_financial_data(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge profitandloss, balancesheet, and cashflow on (company_id, year).

    Year columns are normalised to 'YYYY-MM' before joining.  ``company_id``
    is normalised to a stripped, upper-case ticker on every input frame.
    Sector information is left-joined in (1:1 on company_id) for the
    Financials-sector carve-out used by ``ratios.py``.

    Args:
        datasets: Dictionary as returned by :func:`load_data`.

    Returns:
        Merged DataFrame with one row per (company_id, year), containing
        every column needed by ``ratios.py``, ``cagr.py``, and
        ``cashflow_kpis.py``.

    Raises:
        KeyError: When a required dataset key is absent from *datasets*.
    """
    required_keys = {"profitandloss", "balancesheet", "cashflow"}
    missing_keys = required_keys - set(datasets.keys())
    if missing_keys:
        raise KeyError(f"merge_financial_data: missing dataset keys: {missing_keys}")

    pl = datasets["profitandloss"].copy()
    bs = datasets["balancesheet"].copy()
    cf = datasets["cashflow"].copy()
    sectors = datasets.get("sectors", pd.DataFrame()).copy()
    companies = datasets.get("companies", pd.DataFrame()).copy()

    for frame in (pl, bs, cf, sectors, companies):
        if "company_id" in frame.columns:
            frame["company_id"] = (
                frame["company_id"].astype(str).str.strip().str.upper()
            )
        elif "id" in frame.columns and frame is companies:
            frame["company_id"] = frame["id"].astype(str).str.strip().str.upper()

    for frame in (pl, bs, cf):
        if "year" in frame.columns:
            frame["year"] = frame["year"].apply(_normalize_year)
            n_unparsed = frame["year"].isna().sum()
            if n_unparsed:
                logger.warning(
                    "merge_financial_data: %d rows with unparseable year dropped.",
                    n_unparsed,
                )
                frame.dropna(subset=["year"], inplace=True)

    # Deduplicate on (company_id, year), keeping the last occurrence.
    for name, frame in (("profitandloss", pl), ("balancesheet", bs), ("cashflow", cf)):
        before = len(frame)
        frame.drop_duplicates(subset=["company_id", "year"], keep="last", inplace=True)
        after = len(frame)
        if before != after:
            logger.warning(
                "merge_financial_data: %s — dropped %d duplicate (company_id, year) rows.",
                name,
                before - after,
            )

    merged = pl.merge(
        bs, on=["company_id", "year"], how="outer", suffixes=("", "_bs")
    )
    merged = merged.merge(
        cf, on=["company_id", "year"], how="outer", suffixes=("", "_cf")
    )

    if not sectors.empty and "broad_sector" in sectors.columns:
        merged = merged.merge(
            sectors[["company_id", "broad_sector"]],
            on="company_id",
            how="left",
        )
        merged["broad_sector"] = merged["broad_sector"].fillna("")
    else:
        merged["broad_sector"] = ""

    if not companies.empty and "company_id" in companies.columns:
        company_cols = [
            c for c in ("company_id", "face_value", "book_value")
            if c in companies.columns
        ]
        merged = merged.merge(
            companies[company_cols], on="company_id", how="left"
        )

    merged.sort_values(["company_id", "year"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    logger.info(
        "merge_financial_data: merged dataset has %d rows across %d companies.",
        len(merged),
        merged["company_id"].nunique(),
    )
    return merged


# ---------------------------------------------------------------------------
# 3. compute_all_kpis
# ---------------------------------------------------------------------------


def _run_ratios_engine(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute profitability/leverage/efficiency ratios via ``ratios.py``.

    Calls :func:`src.analytics.ratios.compute_all_ratios` row-by-row — the
    function this module owns and never re-implements.

    Args:
        merged: Merged company-year DataFrame.

    Returns:
        Flat DataFrame with one row per (company_id, year) and the columns
        required by the ``financial_ratios`` table schema.
    """
    records: list[dict] = []

    for _, row in merged.iterrows():
        row_dict = row.to_dict()
        broad_sector = str(row_dict.get("broad_sector", "") or "")

        result = ratios_engine.compute_all_ratios(row_dict, broad_sector=broad_sector)

        records.append({
            "company_id": result.company_id,
            "year": result.year,
            "net_profit_margin_pct": result.profitability.net_profit_margin,
            "operating_profit_margin_pct": result.profitability.operating_profit_margin,
            "return_on_equity_pct": result.profitability.return_on_equity,
            "debt_to_equity": result.leverage.debt_to_equity,
            "interest_coverage": result.leverage.interest_coverage_ratio,
            "asset_turnover": result.efficiency.asset_turnover,
            "total_debt_cr": row_dict.get("borrowings"),
        })

    return pd.DataFrame(records)


def _run_cagr_engine(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute Revenue/PAT/EPS CAGR via ``cagr.py``.

    Calls :func:`src.analytics.cagr.compute_all_cagrs` — never re-implements
    the CAGR formula or edge-case rules.

    Args:
        merged: Merged company-year DataFrame.  Must contain ``sales``,
                 ``net_profit``, and ``eps``.

    Returns:
        DataFrame with company_id, year, and every *_cagr_*yr / *_flag column.
    """
    cagr_input = merged[["company_id", "year", "sales", "net_profit", "eps"]].copy()
    return cagr_engine.compute_all_cagrs(cagr_input)


def _run_cashflow_engine(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute FCF / CFO Quality / CapEx / Capital Allocation via ``cashflow_kpis.py``.

    ``cashflow_kpis.py`` expects an integer ``year`` and a ``pat`` column
    (renamed from ``net_profit``).  This function performs that renaming
    only — it never recomputes any formula itself.

    Args:
        merged: Merged company-year DataFrame.

    Returns:
        Flat DataFrame with company_id, year (as 'YYYY-MM' string for join
        compatibility), fcf, cfo_quality_score, cfo_quality_label,
        capex_intensity, capex_label, fcf_conversion_rate, and capital
        allocation sign/pattern columns.
    """
    cf_input = merged[[
        "company_id", "year", "operating_activity", "investing_activity",
        "financing_activity", "sales", "net_profit", "operating_profit",
    ]].rename(columns={"net_profit": "pat"}).copy()

    # cashflow_kpis.py requires an integer year; encode 'YYYY-MM' as
    # YYYY*100+MM so ordering and uniqueness are both preserved, then map
    # back to the canonical string afterwards.
    year_str = cf_input["year"].astype(str)
    year_int = (
        year_str.str.slice(0, 4).astype(int) * 100
        + year_str.str.slice(5, 7).astype(int)
    )
    int_to_str = dict(zip(year_int, year_str))
    cf_input["year"] = year_int

    # Drop rows with any missing required numeric field — cashflow_kpis.py
    # validates internally, but pre-filtering avoids float(...) on NaN
    # inside the CashFlowRow construction path.
    numeric_cols = [
        "operating_activity", "investing_activity", "financing_activity",
        "sales", "pat", "operating_profit",
    ]
    for col in numeric_cols:
        cf_input[col] = pd.to_numeric(cf_input[col], errors="coerce")

    all_results, _csv_path = cf_engine.process_all_companies(cf_input)

    records: list[dict] = []
    for company_result in all_results:
        cfo_score = (
            company_result.cfo_quality.score if company_result.cfo_quality else None
        )
        cfo_label = (
            company_result.cfo_quality.label if company_result.cfo_quality else None
        )

        fcf_by_year = {r.year: r.fcf for r in company_result.fcf_results}
        capex_by_year = {
            r.year: (r.value, r.label) for r in company_result.capex_results
        }
        conv_by_year = {
            r.year: r.value for r in company_result.fcf_conversion_results
        }
        alloc_by_year = {
            r.year: r for r in company_result.capital_allocation_rows
        }

        all_years = set(fcf_by_year) | set(capex_by_year) | set(conv_by_year) | set(alloc_by_year)

        for yr_int in all_years:
            capex_val, capex_label = capex_by_year.get(yr_int, (None, None))
            alloc = alloc_by_year.get(yr_int)

            records.append({
                "company_id": company_result.company_id,
                "year": int_to_str.get(yr_int, str(yr_int)),
                "free_cash_flow_cr": fcf_by_year.get(yr_int),
                "cfo_quality_score": cfo_score,
                "cfo_quality_label": cfo_label,
                "capex_cr": (
                    abs(capex_val) if capex_val is not None else None
                ),
                "capex_intensity_pct": capex_val,
                "capex_label": capex_label,
                "fcf_conversion_rate": conv_by_year.get(yr_int),
                "cfo_sign": alloc.cfo_sign if alloc else None,
                "cfi_sign": alloc.cfi_sign if alloc else None,
                "cff_sign": alloc.cff_sign if alloc else None,
                "capital_allocation_pattern": alloc.pattern_label if alloc else None,
            })

    return pd.DataFrame(records)


def compute_all_kpis(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute every KPI by delegating to the three existing analytics engines.

    This function performs NO formula computation itself — it only calls
    ``ratios.py``, ``cagr.py``, and ``cashflow_kpis.py``, then joins their
    outputs on (company_id, year).

    Args:
        merged: Output of :func:`merge_financial_data`.

    Returns:
        Single flat DataFrame, one row per (company_id, year), containing
        every column required by the ``financial_ratios`` table (excluding
        the composite quality score, added separately by
        :func:`calculate_quality_score`).
    """
    if merged.empty:
        logger.warning("compute_all_kpis: received an empty merged DataFrame.")
        return pd.DataFrame()

    ratio_df = _run_ratios_engine(merged)
    cagr_df = _run_cagr_engine(merged)
    cf_df = _run_cashflow_engine(merged)

    combined = ratio_df.merge(
        cagr_df[["company_id", "year", "revenue_cagr_5yr", "pat_cagr_5yr", "eps_cagr_5yr"]],
        on=["company_id", "year"],
        how="left",
    )
    combined = combined.merge(cf_df, on=["company_id", "year"], how="left")

    # Source-derived passthrough columns that do not require new formulas.
    passthrough = merged[[
        "company_id", "year", "eps", "dividend_payout",
        "equity_capital", "reserves", "face_value",
    ]].copy()
    passthrough["book_value_per_share"] = np.where(
        (passthrough["equity_capital"].notna())
        & (passthrough["face_value"].notna())
        & (passthrough["face_value"] != 0),
        (passthrough["equity_capital"] + passthrough["reserves"].fillna(0))
        / (passthrough["equity_capital"] / passthrough["face_value"]),
        np.nan,
    )

    combined = combined.merge(
        passthrough[["company_id", "year", "eps", "dividend_payout", "book_value_per_share"]],
        on=["company_id", "year"],
        how="left",
    )
    combined.rename(
        columns={
            "eps": "earnings_per_share",
            "dividend_payout": "dividend_payout_ratio_pct",
        },
        inplace=True,
    )
    combined["cash_from_operations_cr"] = merged.set_index(
        ["company_id", "year"]
    )["operating_activity"].reindex(
        pd.MultiIndex.from_frame(combined[["company_id", "year"]])
    ).to_numpy()

    logger.info(
        "compute_all_kpis: produced %d combined rows from 3 engines.", len(combined)
    )
    return combined.drop_duplicates(subset=["company_id", "year"], keep="last").reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# 4. calculate_quality_score
# ---------------------------------------------------------------------------


def calculate_quality_score(
    df: pd.DataFrame,
    weights: dict[str, float] = QUALITY_WEIGHTS,
) -> pd.Series:
    """Compute a reusable, normalised Composite Quality Score in [0, 100].

    Every input metric is winsorised at the 10th/90th percentile (computed
    across the full universe in *df*) and min-max scaled to [0, 100] before
    being combined with its configured weight.  Metrics where a lower raw
    value is structurally better (debt_to_equity) are inverted during
    scaling.  Rows missing a metric simply exclude that metric's weight from
    the denominator (re-normalised), so partial data never silently zeroes
    out the score.

    Args:
        df:      DataFrame containing (at minimum) the columns named as keys
                 in *weights*: ``return_on_equity_pct`` (mapped to "roe"),
                 ``debt_to_equity``, ``interest_coverage``,
                 ``revenue_cagr_5yr``, ``pat_cagr_5yr``,
                 ``cfo_quality_score``, ``fcf_conversion_rate``.
        weights: Mapping of metric key -> weight (sums to 1.0 by convention;
                 re-normalised per-row when metrics are missing).

    Returns:
        Series of composite scores in [0, 100], aligned to df.index. NaN when
        no metric is available for a row.
    """
    metric_column_map: dict[str, str] = {
        "roe": "return_on_equity_pct",
        "debt_to_equity": "debt_to_equity",
        "interest_coverage": "interest_coverage",
        "revenue_cagr_5yr": "revenue_cagr_5yr",
        "pat_cagr_5yr": "pat_cagr_5yr",
        "cfo_quality_score": "cfo_quality_score",
        "fcf_conversion_rate": "fcf_conversion_rate",
    }
    invert_metrics = {"debt_to_equity"}

    scaled_frame = pd.DataFrame(index=df.index)
    for metric_key, col in metric_column_map.items():
        if col not in df.columns:
            logger.warning(
                "calculate_quality_score: column '%s' for metric '%s' not "
                "found; metric excluded from scoring.",
                col,
                metric_key,
            )
            continue
        scaled_frame[metric_key] = _winsorize_and_scale(
            df[col], invert=metric_key in invert_metrics
        )

    if scaled_frame.empty:
        logger.error(
            "calculate_quality_score: no usable metric columns found; "
            "returning all-NaN score."
        )
        return pd.Series(np.nan, index=df.index)

    weight_row = pd.Series(
        {k: weights.get(k, 0.0) for k in scaled_frame.columns}
    )

    available_mask = scaled_frame.notna()
    weighted_sum = (scaled_frame.fillna(0.0) * weight_row).sum(axis=1)
    weight_total = available_mask.mul(weight_row, axis=1).sum(axis=1)

    score = weighted_sum / weight_total.replace(0.0, np.nan)
    score = score.clip(lower=0.0, upper=100.0).round(2)

    n_missing = score.isna().sum()
    if n_missing:
        logger.warning(
            "calculate_quality_score: %d rows have no usable metrics "
            "(composite_quality_score = NaN).",
            n_missing,
        )

    logger.info(
        "calculate_quality_score: computed scores for %d rows "
        "(mean=%.2f, median=%.2f).",
        score.notna().sum(),
        score.mean(skipna=True) if score.notna().any() else float("nan"),
        score.median(skipna=True) if score.notna().any() else float("nan"),
    )
    return score


# ---------------------------------------------------------------------------
# 5. validate_dataframe
# ---------------------------------------------------------------------------

_KPI_COLUMNS_FOR_VALIDATION: list[str] = [
    "net_profit_margin_pct", "operating_profit_margin_pct",
    "return_on_equity_pct", "debt_to_equity", "interest_coverage",
    "asset_turnover", "free_cash_flow_cr", "capex_cr",
    "earnings_per_share", "book_value_per_share",
    "dividend_payout_ratio_pct", "total_debt_cr",
    "cash_from_operations_cr", "revenue_cagr_5yr", "pat_cagr_5yr",
    "eps_cagr_5yr", "composite_quality_score",
]


def validate_dataframe(df: pd.DataFrame) -> ValidationReport:
    """Run all data-quality checks on the final KPI DataFrame (pre-insert).

    Checks performed:
        * Row count >= EXPECTED_MIN_ROWS.
        * No duplicate (company_id, year) pairs.
        * No KPI column is entirely NULL.
        * Per-column missing-value counts for every KPI column.
        * Descriptive statistics for composite_quality_score.

    Args:
        df: Final flat DataFrame about to be persisted to SQLite.

    Returns:
        :class:`ValidationReport` with every check populated. ``passed`` is
        True only when row count and duplicate checks both succeed; an
        entirely-NULL KPI column produces a WARNING but does not, on its
        own, fail the overall pipeline (it is surfaced in the report for
        analyst review).
    """
    total_rows = len(df)
    row_count_ok = total_rows >= EXPECTED_MIN_ROWS

    duplicate_mask = df.duplicated(subset=["company_id", "year"], keep=False)
    duplicate_count = int(duplicate_mask.sum())
    duplicates_ok = duplicate_count == 0

    null_summary: dict[str, int] = {}
    null_only_columns: list[str] = []
    missing_kpi_counts: dict[str, int] = {}

    for col in _KPI_COLUMNS_FOR_VALIDATION:
        if col not in df.columns:
            logger.warning("validate_dataframe: expected KPI column '%s' missing.", col)
            null_summary[col] = total_rows
            null_only_columns.append(col)
            missing_kpi_counts[col] = total_rows
            continue

        n_null = int(df[col].isna().sum())
        null_summary[col] = n_null
        missing_kpi_counts[col] = n_null

        if total_rows > 0 and n_null == total_rows:
            null_only_columns.append(col)
            logger.error(
                "validate_dataframe: column '%s' is entirely NULL.", col
            )

    quality_stats: dict[str, float] = {}
    if "composite_quality_score" in df.columns:
        qs = df["composite_quality_score"].dropna()
        if not qs.empty:
            quality_stats = {
                "count": float(len(qs)),
                "mean": float(qs.mean()),
                "median": float(qs.median()),
                "std": float(qs.std()) if len(qs) > 1 else 0.0,
                "min": float(qs.min()),
                "max": float(qs.max()),
            }

    passed = row_count_ok and duplicates_ok

    if not row_count_ok:
        logger.error(
            "validate_dataframe: row count %d below expected minimum %d.",
            total_rows,
            EXPECTED_MIN_ROWS,
        )
    if not duplicates_ok:
        logger.error(
            "validate_dataframe: found %d duplicate (company_id, year) rows.",
            duplicate_count,
        )

    return ValidationReport(
        total_rows=total_rows,
        row_count_ok=row_count_ok,
        duplicate_count=duplicate_count,
        duplicates_ok=duplicates_ok,
        null_only_columns=null_only_columns,
        null_summary=null_summary,
        missing_kpi_counts=missing_kpi_counts,
        quality_score_stats=quality_stats,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# 6. save_to_sqlite
# ---------------------------------------------------------------------------


def _get_engine(db_path: Path = DB_PATH) -> Engine:
    """Create (or reuse) a SQLAlchemy Engine bound to the SQLite database.

    Args:
        db_path: Filesystem path to the SQLite database file.

    Returns:
        A SQLAlchemy :class:`Engine` instance.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", future=True)


def save_to_sqlite(
    df: pd.DataFrame,
    *,
    db_path: Path = DB_PATH,
    table_name: str = TABLE_NAME,
) -> int:
    """Persist *df* to the SQLite ``financial_ratios`` table, replacing
    duplicate (company_id, year) records.

    Strategy: delete any existing rows in the target table whose
    (company_id, year) matches a row in *df*, then append *df*. This
    achieves idempotent upsert semantics without requiring SQLite
    ``ON CONFLICT`` clauses to be hand-written for every column.

    Args:
        df:         Final DataFrame to persist.
        db_path:    Path to the SQLite database file.
        table_name: Destination table name.

    Returns:
        Number of rows inserted.

    Raises:
        ValueError: When *df* is empty.
    """
    if df.empty:
        raise ValueError("save_to_sqlite: cannot persist an empty DataFrame.")

    engine = _get_engine(db_path)

    with engine.begin() as conn:
        inspector = inspect(conn)
        table_exists = table_name in inspector.get_table_names()

        if table_exists:
            existing_cols = {c["name"] for c in inspector.get_columns(table_name)}
            missing_cols = [c for c in df.columns if c not in existing_cols]
            for col in missing_cols:
                sample = df[col].dropna()
                if sample.empty:
                    sql_type = "TEXT"
                elif pd.api.types.is_bool_dtype(df[col]):
                    sql_type = "INTEGER"
                elif pd.api.types.is_integer_dtype(df[col]):
                    sql_type = "INTEGER"
                elif pd.api.types.is_float_dtype(df[col]):
                    sql_type = "REAL"
                else:
                    sql_type = "TEXT"
                conn.execute(text(f'ALTER TABLE {table_name} ADD COLUMN "{col}" {sql_type}'))
                logger.info(
                    "save_to_sqlite: added missing column '%s' (%s) to '%s'.",
                    col, sql_type, table_name,
                )

            keys = list(df[["company_id", "year"]].itertuples(index=False, name=None))
            placeholders = ", ".join(f"(:cid_{i}, :yr_{i})" for i in range(len(keys)))
            params = {}
            for i, (cid, yr) in enumerate(keys):
                params[f"cid_{i}"] = cid
                params[f"yr_{i}"] = yr

            delete_sql = text(
                f"DELETE FROM {table_name} "
                f"WHERE (company_id, year) IN ({placeholders})"
            )
            result = conn.execute(delete_sql, params)
            logger.info(
                "save_to_sqlite: removed %d pre-existing duplicate rows "
                "before insert.",
                result.rowcount if result.rowcount is not None else 0,
            )

    df.to_sql(table_name, engine, if_exists="append", index=False)

    logger.info(
        "save_to_sqlite: inserted %d rows into '%s' at %s.",
        len(df),
        table_name,
        db_path,
    )
    return len(df)


# ---------------------------------------------------------------------------
# 7. verify_database
# ---------------------------------------------------------------------------


def verify_database(
    *,
    db_path: Path = DB_PATH,
    table_name: str = TABLE_NAME,
) -> ValidationReport:
    """Re-query the persisted SQLite table and validate it independently.

    Executes ``SELECT COUNT(*) FROM financial_ratios`` plus duplicate and
    NULL checks directly against the database (not against the in-memory
    DataFrame), so this function detects any insert-time corruption.

    Args:
        db_path:    Path to the SQLite database file.
        table_name: Table to verify.

    Returns:
        :class:`ValidationReport` reflecting the persisted state.
    """
    engine = _get_engine(db_path)

    with engine.connect() as conn:
        total_rows = conn.execute(
            text(f"SELECT COUNT(*) FROM {table_name}")
        ).scalar_one()

        dup_query = text(
            f"""
            SELECT COUNT(*) FROM (
                SELECT company_id, year, COUNT(*) AS cnt
                FROM {table_name}
                GROUP BY company_id, year
                HAVING cnt > 1
            )
            """
        )
        duplicate_groups = conn.execute(dup_query).scalar_one()

        null_summary: dict[str, int] = {}
        null_only_columns: list[str] = []
        for col in _KPI_COLUMNS_FOR_VALIDATION:
            col_check = conn.execute(
                text(f"SELECT COUNT(*) FROM pragma_table_info('{table_name}') "
                     f"WHERE name = :col"),
                {"col": col},
            ).scalar_one()
            if col_check == 0:
                null_summary[col] = total_rows
                null_only_columns.append(col)
                continue

            n_null = conn.execute(
                text(f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL")
            ).scalar_one()
            null_summary[col] = n_null
            if total_rows > 0 and n_null == total_rows:
                null_only_columns.append(col)

        quality_stats: dict[str, float] = {}
        if "composite_quality_score" not in null_only_columns:
            stats_row = conn.execute(
                text(
                    "SELECT COUNT(composite_quality_score), "
                    "AVG(composite_quality_score), "
                    "MIN(composite_quality_score), "
                    "MAX(composite_quality_score) "
                    f"FROM {table_name} WHERE composite_quality_score IS NOT NULL"
                )
            ).one()
            count, avg, min_v, max_v = stats_row
            if count:
                quality_stats = {
                    "count": float(count),
                    "mean": float(avg) if avg is not None else float("nan"),
                    "min": float(min_v) if min_v is not None else float("nan"),
                    "max": float(max_v) if max_v is not None else float("nan"),
                }

    row_count_ok = total_rows >= EXPECTED_MIN_ROWS
    duplicates_ok = duplicate_groups == 0
    passed = row_count_ok and duplicates_ok

    if not row_count_ok:
        logger.error(
            "verify_database: row count %d below expected minimum %d.",
            total_rows,
            EXPECTED_MIN_ROWS,
        )
    if not duplicates_ok:
        logger.error(
            "verify_database: found %d duplicate (company_id, year) groups.",
            duplicate_groups,
        )
    if null_only_columns:
        logger.warning(
            "verify_database: columns entirely NULL: %s.", null_only_columns
        )

    logger.info(
        "verify_database: total_rows=%d | duplicate_groups=%d | passed=%s.",
        total_rows,
        duplicate_groups,
        passed,
    )

    return ValidationReport(
        total_rows=total_rows,
        row_count_ok=row_count_ok,
        duplicate_count=duplicate_groups,
        duplicates_ok=duplicates_ok,
        null_only_columns=null_only_columns,
        null_summary=null_summary,
        missing_kpi_counts=null_summary,
        quality_score_stats=quality_stats,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Validation report writer
# ---------------------------------------------------------------------------


def generate_validation_report(
    report: ValidationReport,
    *,
    output_path: Path = VALIDATION_REPORT_PATH,
    elapsed_seconds: Optional[float] = None,
) -> Path:
    """Write a human-readable validation report to disk.

    Args:
        report:          The :class:`ValidationReport` to render.
        output_path:     Destination .txt file.
        elapsed_seconds: Optional total pipeline execution time to include.

    Returns:
        The resolved path of the written report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("NIFTY 100 FINANCIAL INTELLIGENCE PLATFORM")
    lines.append("Database Validation Report — financial_ratios table")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Total Rows               : {report.total_rows}")
    lines.append(f"Expected Minimum Rows    : {EXPECTED_MIN_ROWS}")
    lines.append(f"Row Count Check          : {'PASS' if report.row_count_ok else 'FAIL'}")
    lines.append("")
    lines.append(f"Duplicate (company_id, year) Rows : {report.duplicate_count}")
    lines.append(
        f"Duplicate Check                   : "
        f"{'PASS' if report.duplicates_ok else 'FAIL'}"
    )
    lines.append("")
    lines.append("-" * 78)
    lines.append("Missing KPI Counts (NULL per column)")
    lines.append("-" * 78)
    for col, n_null in report.missing_kpi_counts.items():
        flag = "  <-- ALL NULL" if col in report.null_only_columns else ""
        lines.append(f"  {col:<40s}: {n_null:>6d}{flag}")
    lines.append("")
    lines.append("-" * 78)
    lines.append("NULL Summary")
    lines.append("-" * 78)
    for col, n_null in report.null_summary.items():
        pct = (n_null / report.total_rows * 100.0) if report.total_rows else 0.0
        lines.append(f"  {col:<40s}: {n_null:>6d} ({pct:5.1f}%)")
    lines.append("")
    lines.append("-" * 78)
    lines.append("Quality Score Statistics")
    lines.append("-" * 78)
    if report.quality_score_stats:
        for stat_name, stat_value in report.quality_score_stats.items():
            lines.append(f"  {stat_name:<10s}: {stat_value:.4f}")
    else:
        lines.append("  No quality score data available.")
    lines.append("")
    lines.append("-" * 78)
    lines.append("Columns Entirely NULL")
    lines.append("-" * 78)
    if report.null_only_columns:
        for col in report.null_only_columns:
            lines.append(f"  - {col}")
    else:
        lines.append("  None")
    lines.append("")
    if elapsed_seconds is not None:
        lines.append(f"Execution Time (seconds) : {elapsed_seconds:.3f}")
        lines.append("")
    lines.append("=" * 78)
    lines.append(f"OVERALL VALIDATION RESULT : {'PASS' if report.passed else 'FAIL'}")
    lines.append("=" * 78)

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("generate_validation_report: report written to %s.", output_path)
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Top-level pipeline orchestrator
# ---------------------------------------------------------------------------


def run_pipeline(
    *,
    raw_dir: Path = _RAW_DIR,
    db_path: Path = DB_PATH,
    table_name: str = TABLE_NAME,
    output_dir: Path = _OUTPUT_DIR,
) -> ValidationReport:
    """Execute the full load -> merge -> compute -> validate -> persist -> verify pipeline.

    Args:
        raw_dir:    Directory containing raw source files.
        db_path:    Destination SQLite database path.
        table_name: Destination table name.
        output_dir: Directory for the validation report and log file.

    Returns:
        :class:`ValidationReport` from the post-insert database verification.
    """
    _configure_file_logging(output_dir / "ratio_engine.log")
    start_time = time.monotonic()

    logger.info("run_pipeline: starting Nifty 100 ratio engine pipeline.")
    logger.info("run_pipeline: database target = %s.", db_path)

    datasets = load_data(raw_dir)
    merged = merge_financial_data(datasets)

    if merged.empty:
        logger.error("run_pipeline: merged dataset is empty; aborting.")
        report = ValidationReport(passed=False)
        generate_validation_report(
            report,
            output_path=output_dir / "database_validation_report.txt",
            elapsed_seconds=time.monotonic() - start_time,
        )
        return report

    kpi_df = compute_all_kpis(merged)
    kpi_df["composite_quality_score"] = calculate_quality_score(kpi_df)

    pre_insert_report = validate_dataframe(kpi_df)
    logger.info(
        "run_pipeline: pre-insert validation passed=%s.", pre_insert_report.passed
    )

    save_to_sqlite(kpi_df, db_path=db_path, table_name=table_name)

    final_report = verify_database(db_path=db_path, table_name=table_name)
    elapsed = time.monotonic() - start_time

    generate_validation_report(
        final_report,
        output_path=output_dir / "database_validation_report.txt",
        elapsed_seconds=elapsed,
    )

    logger.info(
        "run_pipeline: completed in %.3f seconds | passed=%s.",
        elapsed,
        final_report.passed,
    )
    return final_report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline()