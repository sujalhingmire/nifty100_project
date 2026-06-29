"""
cagr.py — CAGR (Compound Annual Growth Rate) Engine
=====================================================
Sprint 2 · Day 10 · Nifty 100 Financial Intelligence Platform

Computes Revenue, PAT (Net Profit), and EPS CAGR for 3-year, 5-year, and
10-year windows for every company-year combination in the platform dataset.

All six edge cases mandated by the project specification are handled:
    NORMAL          — both values positive; CAGR computed.
    DECLINE_TO_LOSS — start > 0, end < 0; CAGR undefined.
    TURNAROUND      — start < 0, end > 0; CAGR undefined.
    BOTH_NEGATIVE   — both values negative; CAGR undefined.
    ZERO_BASE       — start == 0; division impossible.
    INSUFFICIENT    — fewer data points than the requested window.

Follows:
    PEP 8 · SOLID · DRY · type hints throughout · logging (no print)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

__all__ = [
    "CAGRFlag",
    "CAGRResult",
    "CAGRRecord",
    "PERIODS",
    "METRIC_COLUMNS",
    "OUTPUT_COLUMNS",
    "calculate_cagr",
    "determine_cagr_flag",
    "validate_cagr_inputs",
    "compute_growth_metric",
    "compute_company_cagr",
    "compute_all_cagrs",
]

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CAGR windows (years) — extend this tuple to add new periods without code changes.
PERIODS: tuple[int, ...] = (3, 5, 10)

# Source DataFrame column names for each metric.
METRIC_COLUMNS: dict[str, str] = {
    "revenue": "sales",
    "pat": "net_profit",
    "eps": "eps",
}

# All output columns produced by compute_all_cagrs().
OUTPUT_COLUMNS: list[str] = (
    ["company_id", "year"]
    + [
        col
        for metric in METRIC_COLUMNS
        for n in PERIODS
        for col in (f"{metric}_cagr_{n}yr", f"{metric}_cagr_{n}yr_flag")
    ]
)

# Precision for CAGR values (decimal places).
_CAGR_PRECISION: int = 2

# Tolerance for floating-point zero comparison.
_ZERO_TOL: float = 1e-9


# ---------------------------------------------------------------------------
# Flag enumeration
# ---------------------------------------------------------------------------


class CAGRFlag(str, Enum):
    """Categorical labels describing why a CAGR value is valid or undefined.

    Using ``str`` as a mixin ensures direct SQLite/pandas string compatibility
    without needing an extra `.value` accessor.
    """

    NORMAL = "NORMAL"
    DECLINE_TO_LOSS = "DECLINE_TO_LOSS"
    TURNAROUND = "TURNAROUND"
    BOTH_NEGATIVE = "BOTH_NEGATIVE"
    ZERO_BASE = "ZERO_BASE"
    INSUFFICIENT = "INSUFFICIENT"
    MISSING = "MISSING"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CAGRResult:
    """Immutable result for a single CAGR computation.

    Attributes:
        value: Rounded CAGR percentage, or None when undefined.
        flag:  :class:`CAGRFlag` describing the computation outcome.
    """

    value: Optional[float]
    flag: CAGRFlag


@dataclass
class CAGRRecord:
    """All CAGR results for one company-year row.

    Attribute names mirror the output column schema so the dataclass can be
    converted directly to a DataFrame row via ``dataclasses.asdict()``.
    """

    company_id: str = ""
    year: str = ""

    revenue_cagr_3yr: Optional[float] = None
    revenue_cagr_3yr_flag: str = CAGRFlag.MISSING
    revenue_cagr_5yr: Optional[float] = None
    revenue_cagr_5yr_flag: str = CAGRFlag.MISSING
    revenue_cagr_10yr: Optional[float] = None
    revenue_cagr_10yr_flag: str = CAGRFlag.MISSING

    pat_cagr_3yr: Optional[float] = None
    pat_cagr_3yr_flag: str = CAGRFlag.MISSING
    pat_cagr_5yr: Optional[float] = None
    pat_cagr_5yr_flag: str = CAGRFlag.MISSING
    pat_cagr_10yr: Optional[float] = None
    pat_cagr_10yr_flag: str = CAGRFlag.MISSING

    eps_cagr_3yr: Optional[float] = None
    eps_cagr_3yr_flag: str = CAGRFlag.MISSING
    eps_cagr_5yr: Optional[float] = None
    eps_cagr_5yr_flag: str = CAGRFlag.MISSING
    eps_cagr_10yr: Optional[float] = None
    eps_cagr_10yr_flag: str = CAGRFlag.MISSING


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _is_invalid(value: object) -> bool:
    """Return True when *value* is None, non-numeric, NaN, or ±Infinity.

    Args:
        value: Any scalar from a pandas cell or Python dict.

    Returns:
        True when the value cannot be used safely in arithmetic.
    """
    if value is None:
        return True
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return True
    return math.isnan(f) or math.isinf(f)


def _safe_float(value: object, label: str = "") -> Optional[float]:
    """Convert *value* to a finite float; return None and log on failure.

    Args:
        value: Raw value from a pandas Series or dict.
        label: Field name for log context.

    Returns:
        Finite float, or None when conversion fails.
    """
    if _is_invalid(value):
        if label:
            logger.debug("Missing or invalid value for field '%s'.", label)
        return None
    return float(value)  # type: ignore[arg-type]


def _is_zero(value: float) -> bool:
    """Return True when *value* is within floating-point zero tolerance.

    Args:
        value: Finite float already validated by _safe_float.

    Returns:
        True when abs(value) < _ZERO_TOL.
    """
    return abs(value) < _ZERO_TOL


# ---------------------------------------------------------------------------
# Core CAGR logic
# ---------------------------------------------------------------------------


def determine_cagr_flag(start: float, end: float) -> CAGRFlag:
    """Classify a (start, end) pair and return the appropriate :class:`CAGRFlag`.

    This is the single source of truth for all six edge-case rules.

    Args:
        start: Base-period value (already validated as finite).
        end:   End-period value (already validated as finite).

    Returns:
        A :class:`CAGRFlag` indicating how this pair should be treated.
    """
    if _is_zero(start):
        return CAGRFlag.ZERO_BASE
    if start > 0 and end > 0:
        return CAGRFlag.NORMAL
    if start > 0 and end < 0:
        return CAGRFlag.DECLINE_TO_LOSS
    if start < 0 and end > 0:
        return CAGRFlag.TURNAROUND
    # start < 0 and end < 0  (also covers end == 0 with negative start)
    return CAGRFlag.BOTH_NEGATIVE


def validate_cagr_inputs(
    start: object,
    end: object,
    years: int,
    *,
    metric: str = "",
    company_id: str = "",
    year: str = "",
) -> tuple[Optional[float], Optional[float], Optional[CAGRFlag]]:
    """Validate raw inputs and return usable floats or an early-exit flag.

    Args:
        start:      Base-period value (raw, possibly None/NaN).
        end:        End-period value (raw, possibly None/NaN).
        years:      Number of years in the CAGR window (must be >= 1).
        metric:     Metric name for log context.
        company_id: Company identifier for log context.
        year:       Current year label for log context.

    Returns:
        Tuple of (start_float, end_float, early_flag):
            - When (start_float, end_float) are both not None → caller computes CAGR.
            - When early_flag is not None → return CAGRResult(None, early_flag).
    """
    ctx = f"metric={metric} | company={company_id} | year={year}"

    s_val = _safe_float(start, metric)
    e_val = _safe_float(end, metric)

    if s_val is None or e_val is None:
        logger.warning("CAGR skipped — missing value | %s.", ctx)
        return None, None, CAGRFlag.MISSING

    if years < 1:
        logger.warning("CAGR skipped — invalid years=%d | %s.", years, ctx)
        return None, None, CAGRFlag.INSUFFICIENT

    return s_val, e_val, None


def calculate_cagr(
    start: object,
    end: object,
    years: int,
    *,
    metric: str = "",
    company_id: str = "",
    year: str = "",
) -> CAGRResult:
    """Compute CAGR for a single (start, end, years) triple.

    Formula::

        CAGR = ((end / start) ** (1 / years) - 1) × 100

    All six edge cases are handled via :func:`determine_cagr_flag`.

    Args:
        start:      Base-period value.  May be None/NaN.
        end:        End-period value.  May be None/NaN.
        years:      Length of the CAGR window (e.g. 3, 5, 10).
        metric:     Metric name for log context.
        company_id: For log context.
        year:       For log context.

    Returns:
        :class:`CAGRResult` with (value, flag).  value is None for all
        non-NORMAL flags.
    """
    ctx = f"metric={metric} | company={company_id} | year={year} | years={years}"

    s_val, e_val, early_flag = validate_cagr_inputs(
        start, end, years,
        metric=metric, company_id=company_id, year=year,
    )
    if early_flag is not None:
        return CAGRResult(value=None, flag=early_flag)

    # s_val and e_val are both finite floats here.
    flag = determine_cagr_flag(s_val, e_val)  # type: ignore[arg-type]

    if flag is not CAGRFlag.NORMAL:
        logger.warning(
            "CAGR undefined — flag=%s | %s | start=%.4g | end=%.4g.",
            flag,
            ctx,
            s_val,
            e_val,
        )
        return CAGRResult(value=None, flag=flag)

    cagr_raw = ((e_val / s_val) ** (1.0 / years) - 1.0) * 100.0
    cagr_rounded = round(cagr_raw, _CAGR_PRECISION)

    logger.info(
        "CAGR computed — %.2f%% | %s.", cagr_rounded, ctx
    )
    return CAGRResult(value=cagr_rounded, flag=CAGRFlag.NORMAL)


# ---------------------------------------------------------------------------
# Per-metric window helper
# ---------------------------------------------------------------------------


def compute_growth_metric(
    series: pd.Series,
    end_idx: int,
    years: int,
    *,
    metric: str = "",
    company_id: str = "",
    year: str = "",
) -> CAGRResult:
    """Compute CAGR for one metric/window given a sorted value Series.

    Handles the INSUFFICIENT case when the Series does not have enough history
    to reach back *years* positions from *end_idx*.

    Args:
        series:     Pandas Series of metric values sorted ascending by year,
                    index reset to 0..N-1.
        end_idx:    Integer position of the current (end) row in *series*.
        years:      Number of years to look back.
        metric:     Metric name for log context.
        company_id: For log context.
        year:       For log context.

    Returns:
        :class:`CAGRResult`.
    """
    start_idx = end_idx - years

    if start_idx < 0:
        logger.warning(
            "CAGR skipped — insufficient history (need %d more years) "
            "| metric=%s | company=%s | year=%s.",
            abs(start_idx),
            metric,
            company_id,
            year,
        )
        return CAGRResult(value=None, flag=CAGRFlag.INSUFFICIENT)

    start_val = series.iloc[start_idx]
    end_val = series.iloc[end_idx]

    return calculate_cagr(
        start_val,
        end_val,
        years,
        metric=metric,
        company_id=company_id,
        year=year,
    )


# ---------------------------------------------------------------------------
# Company-level orchestrator
# ---------------------------------------------------------------------------


def compute_company_cagr(group: pd.DataFrame) -> list[CAGRRecord]:
    """Compute all CAGR metrics for every row in one company's time-series.

    Args:
        group: DataFrame slice for a single company, containing at minimum
               the columns ``company_id``, ``year``, ``sales``, ``net_profit``,
               and ``eps``.  Rows must be sorted ascending by year (handled
               internally).

    Returns:
        List of :class:`CAGRRecord` — one per row in *group*.
    """
    # Defensive sort: deduplicate year, keep last occurrence, sort ascending.
    group = (
        group
        .drop_duplicates(subset=["year"], keep="last")
        .sort_values("year")
        .reset_index(drop=True)
    )

    company_id: str = str(group["company_id"].iloc[0])
    records: list[CAGRRecord] = []

    for idx in range(len(group)):
        row = group.iloc[idx]
        year_label: str = str(row["year"])

        rec = CAGRRecord(company_id=company_id, year=year_label)

        for metric_key, col in METRIC_COLUMNS.items():
            series: pd.Series = group[col]
            for n in PERIODS:
                result = compute_growth_metric(
                    series,
                    end_idx=idx,
                    years=n,
                    metric=metric_key,
                    company_id=company_id,
                    year=year_label,
                )
                attr_val = f"{metric_key}_cagr_{n}yr"
                attr_flag = f"{metric_key}_cagr_{n}yr_flag"
                setattr(rec, attr_val, result.value)
                setattr(rec, attr_flag, str(result.flag))

        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------


def compute_all_cagrs(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Revenue, PAT, and EPS CAGR for all companies in *df*.

    Accepts the raw merged ``profitandloss`` DataFrame (as loaded from SQLite
    or Excel) and returns a clean output DataFrame ready for insertion into the
    ``financial_ratios`` table or export to CSV.

    Required input columns:
        ``company_id``, ``year``, ``sales``, ``net_profit``, ``eps``

    Args:
        df: Merged P&L DataFrame with one row per (company_id, year).

    Returns:
        DataFrame with columns defined by :data:`OUTPUT_COLUMNS`, one row per
        (company_id, year).  Suitable for ``df.to_sql()`` or ``df.to_csv()``.

    Raises:
        ValueError: When any required column is absent from *df*.
    """
    required = {"company_id", "year"} | set(METRIC_COLUMNS.values())
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"compute_all_cagrs: input DataFrame missing columns: {missing_cols}"
        )

    if df.empty:
        logger.warning("compute_all_cagrs received an empty DataFrame.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    logger.info(
        "compute_all_cagrs: processing %d rows across %d companies.",
        len(df),
        df["company_id"].nunique(),
    )

    all_records: list[CAGRRecord] = []

    for company_id, group in df.groupby("company_id", sort=False):
        logger.debug("Processing CAGR for company_id=%s.", company_id)
        try:
            records = compute_company_cagr(group)
            all_records.extend(records)
        except Exception:
            logger.error(
                "Unexpected error computing CAGR for company_id=%s.",
                company_id,
                exc_info=True,
            )

    if not all_records:
        logger.warning("compute_all_cagrs produced no records.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    import dataclasses

    result_df = pd.DataFrame(
        [dataclasses.asdict(r) for r in all_records]
    )

    # Cast dtypes explicitly for SQLite / pandas 2+ compatibility.
    # Flag columns -> object (plain str); value columns -> float64.
    flag_cols = [c for c in OUTPUT_COLUMNS if c.endswith("_flag")]
    value_cols = [
        c for c in OUTPUT_COLUMNS
        if c not in ("company_id", "year") and not c.endswith("_flag")
    ]
    result_df[flag_cols] = result_df[flag_cols].astype(object)
    result_df[value_cols] = result_df[value_cols].astype(float)

    logger.info(
        "compute_all_cagrs: completed — %d output rows.", len(result_df)
    )
    return result_df[OUTPUT_COLUMNS].reset_index(drop=True)