"""
ratio_edge_cases.py — Day 13 Edge-Case & Cross-Validation Engine
==================================================================
Sprint 2 · Day 13 · Nifty 100 Financial Intelligence Platform

Extends ``src.analytics.ratios`` with:

1. A Financial-sector carve-out helper (``is_financial_company``) that wraps
   the existing ``ratios._is_financial_sector`` so the high_leverage_flag
   rule already implemented in :func:`ratios.calculate_debt_to_equity`
   continues to be the single source of truth — no formula duplication.
2. Cross-validation of computed ROCE / ROE against the source
   ``companies.xlsx`` values (``roce_percentage`` / ``roe_percentage``).
3. Anomaly categorisation into exactly three categories:
   ``DATA_SOURCE_ISSUE``, ``VERSION_DIFFERENCE``, ``FORMULA_DISCREPANCY``.
4. Structured logging of every anomaly (> 5% difference) to
   ``output/ratio_edge_cases.log``.

IMPORTANT: This module does NOT recompute or override any ratio. Computed
values always come from ``src.analytics.ratios``; source values from
``companies.xlsx`` are used for comparison/display only and are NEVER
written back into the computed analytics fields.

Follows:
    PEP 8 · SOLID · DRY · type hints throughout · logging (no print)
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Bootstrap: ensure the project root (which contains the ``src`` package) is
# on sys.path. This is required when this file is executed directly
# (e.g. ``python src/analytics/ratio_edge_cases.py``) rather than as a
# module (``python -m src.analytics.ratio_edge_cases``), since in the
# former case Python only adds the script's own directory to sys.path,
# not the project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_BOOTSTRAP))

from src.analytics.ratios import _is_financial_sector, _safe_float

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROCE_DIFF_THRESHOLD: float = 5.0  # percentage points
ROE_DIFF_THRESHOLD: float = 5.0   # percentage points

# Anomaly categories — must match exactly (Task 4).
CATEGORY_DATA_SOURCE_ISSUE: str = "DATA_SOURCE_ISSUE"
CATEGORY_VERSION_DIFFERENCE: str = "VERSION_DIFFERENCE"
CATEGORY_FORMULA_DISCREPANCY: str = "FORMULA_DISCREPANCY"

VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        CATEGORY_DATA_SOURCE_ISSUE,
        CATEGORY_VERSION_DIFFERENCE,
        CATEGORY_FORMULA_DISCREPANCY,
    }
)

# Heuristic thresholds used to discriminate between anomaly categories.
_VERSION_DIFFERENCE_BAND: tuple[float, float] = (5.0, 15.0)
_DATA_SOURCE_ISSUE_MULTIPLE: float = 5.0  # computed >= N x source (or vice versa)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_OUTPUT_DIR: Path = Path(
    os.environ.get("NIFTY_OUTPUT_DIR", str(PROJECT_ROOT / "output"))
)
RATIO_EDGE_CASES_LOG_PATH: Path = _OUTPUT_DIR / "ratio_edge_cases.log"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnomalyRecord:
    """Structured representation of a single cross-validation anomaly."""

    company_id: str
    year: str
    metric: str
    computed_value: float
    source_value: float
    difference: float
    category: str
    severity: str
    explanation: str
    recommended_action: str
    timestamp: str


# ---------------------------------------------------------------------------
# Task 6 — helper functions
# ---------------------------------------------------------------------------


def is_financial_company(broad_sector: str) -> bool:
    """Return True when *broad_sector* identifies a Financials-sector company.

    Thin, explicitly-named wrapper around
    :func:`src.analytics.ratios._is_financial_sector` so the carve-out logic
    is defined exactly once and reused everywhere (DRY). Banks, NBFCs, and
    Insurance companies all roll up to ``broad_sector == "Financials"``.

    Args:
        broad_sector: Value from ``sectors.broad_sector`` / ``companies.xlsx``.

    Returns:
        True if the company is a Financial-sector company; False otherwise.
    """
    if not broad_sector:
        return False
    return _is_financial_sector(broad_sector)


def categorise_anomaly(
    computed_value: float,
    source_value: float,
    difference: float,
) -> tuple[str, str]:
    """Classify a ROCE/ROE anomaly into exactly one of the three categories.

    Classification heuristic (Task 4):

    * ``FORMULA_DISCREPANCY`` — the computed and source values are of a
      similar order of magnitude (difference is modest relative to the
      computed value), suggesting a methodology gap, e.g. different
      EBIT/Capital-Employed definitions.
    * ``DATA_SOURCE_ISSUE`` — source value is implausibly small/large versus
      computed (e.g. off by a scale/unit factor or a known bad source cell),
      such as the documented TCS ROE = 0.52% case.
    * ``VERSION_DIFFERENCE`` — difference falls in a mid-band that is
      consistent with the source being computed against a prior fiscal
      restatement / different reporting period.

    Args:
        computed_value: Ratio computed by this platform (percentage).
        source_value:   Ratio reported in ``companies.xlsx`` (percentage).
        difference:     ``abs(computed_value - source_value)``.

    Returns:
        Tuple of (category, human-readable reason).
    """
    abs_computed = abs(computed_value)
    abs_source = abs(source_value)

    # Source value near zero (or far smaller in magnitude) while computed is
    # substantial → classic bad/stale source cell (e.g. TCS ROE = 0.52).
    if abs_source < 1.0 or (
        abs_source > 0 and abs_computed >= _DATA_SOURCE_ISSUE_MULTIPLE * abs_source
    ):
        return (
            CATEGORY_DATA_SOURCE_ISSUE,
            "Source value is implausibly small relative to the computed "
            "value, indicating a stale, mis-scaled, or erroneous source cell.",
        )

    # Source value far larger than computed by the same multiple test.
    if abs_computed > 0 and abs_source >= _DATA_SOURCE_ISSUE_MULTIPLE * abs_computed:
        return (
            CATEGORY_DATA_SOURCE_ISSUE,
            "Source value is implausibly large relative to the computed "
            "value, indicating a stale, mis-scaled, or erroneous source cell.",
        )

    low, high = _VERSION_DIFFERENCE_BAND
    if low <= difference <= high:
        return (
            CATEGORY_VERSION_DIFFERENCE,
            "Moderate difference consistent with source data computed "
            "against a different fiscal restatement or reporting version.",
        )

    return (
        CATEGORY_FORMULA_DISCREPANCY,
        "Computed and source values are of comparable magnitude; the gap "
        "likely stems from differing ratio formula definitions/components.",
    )


def _severity_for_difference(difference: float) -> str:
    """Map an absolute percentage-point difference to a severity label.

    Args:
        difference: ``abs(computed_value - source_value)``.

    Returns:
        One of "LOW", "MEDIUM", "HIGH", "CRITICAL".
    """
    if difference > 50.0:
        return "CRITICAL"
    if difference > 20.0:
        return "HIGH"
    if difference > 10.0:
        return "MEDIUM"
    return "LOW"


def _recommended_action_for(category: str) -> str:
    """Return a standard recommended action string for a given category.

    Args:
        category: One of the three valid anomaly categories.

    Returns:
        Human-readable recommended next step for an analyst/data engineer.
    """
    actions = {
        CATEGORY_DATA_SOURCE_ISSUE: (
            "Verify and correct the source companies.xlsx cell; do not "
            "overwrite computed analytics value."
        ),
        CATEGORY_VERSION_DIFFERENCE: (
            "Confirm fiscal year / restatement alignment between source "
            "and computed datasets; reconcile reporting periods."
        ),
        CATEGORY_FORMULA_DISCREPANCY: (
            "Review ratio formula components against source methodology "
            "documentation; document the definitional difference."
        ),
    }
    return actions[category]


def log_ratio_anomaly(record: AnomalyRecord) -> None:
    """Write one structured anomaly entry to ``output/ratio_edge_cases.log``.

    The log line is structured (pipe-delimited, parseable) and contains:
    Timestamp, Company, Year, Metric, Computed Value, Source Value,
    Difference, Category, Severity, Explanation, Recommended Action.

    Args:
        record: Fully populated :class:`AnomalyRecord`.

    Returns:
        None.
    """
    file_logger = _get_edge_case_file_logger()
    file_logger.warning(
        "Timestamp=%s | Company=%s | Year=%s | Metric=%s | "
        "Computed=%.4f | Source=%.4f | Difference=%.4f | "
        "Category=%s | Severity=%s | Explanation=%s | "
        "Recommended Action=%s",
        record.timestamp,
        record.company_id,
        record.year,
        record.metric,
        record.computed_value,
        record.source_value,
        record.difference,
        record.category,
        record.severity,
        record.explanation,
        record.recommended_action,
    )


_edge_case_logger: Optional[logging.Logger] = None


def _get_edge_case_file_logger(
    log_path: Optional[Path] = None,
) -> logging.Logger:
    """Return a dedicated, idempotently-configured logger for edge cases.

    Uses a distinct logger (rather than the root logger) so edge-case
    entries always land in ``ratio_edge_cases.log`` regardless of how the
    caller has configured root/application logging elsewhere.

    Args:
        log_path: Destination path for ``ratio_edge_cases.log``. When None
            (the default), the current value of the module-level
            ``RATIO_EDGE_CASES_LOG_PATH`` is read dynamically so test
            fixtures / environment overrides applied after import are
            honoured (avoids Python's mutable-default late-binding trap).

    Returns:
        Configured ``logging.Logger`` instance.
    """
    global _edge_case_logger

    if log_path is None:
        log_path = RATIO_EDGE_CASES_LOG_PATH

    if _edge_case_logger is not None:
        return _edge_case_logger

    log_path.parent.mkdir(parents=True, exist_ok=True)
    edge_logger = logging.getLogger("ratio_edge_cases")
    edge_logger.setLevel(logging.WARNING)
    edge_logger.propagate = False

    already_configured = any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == log_path.resolve()
        for handler in edge_logger.handlers
    )
    if not already_configured:
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        edge_logger.addHandler(file_handler)

    _edge_case_logger = edge_logger
    return edge_logger


def _build_and_log_anomaly(
    *,
    company_id: str,
    year: str,
    metric: str,
    computed_value: float,
    source_value: float,
    difference: float,
) -> AnomalyRecord:
    """Categorise an anomaly, build its record, and persist it to the log.

    Shared by :func:`validate_roce` and :func:`validate_roe` to avoid
    duplicating the categorise → severity → log sequence (DRY).

    Args:
        company_id:     Company identifier.
        year:           Fiscal year.
        metric:         "ROCE" or "ROE".
        computed_value: Platform-computed ratio (percentage).
        source_value:   companies.xlsx ratio (percentage).
        difference:     abs(computed_value - source_value).

    Returns:
        The persisted :class:`AnomalyRecord`.
    """
    category, explanation = categorise_anomaly(
        computed_value, source_value, difference
    )
    severity = _severity_for_difference(difference)
    record = AnomalyRecord(
        company_id=company_id,
        year=year,
        metric=metric,
        computed_value=computed_value,
        source_value=source_value,
        difference=difference,
        category=category,
        severity=severity,
        explanation=explanation,
        recommended_action=_recommended_action_for(category),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    log_ratio_anomaly(record)
    logger.warning(
        "Ratio anomaly detected | metric=%s | company=%s | year=%s | "
        "category=%s | severity=%s | diff=%.4f",
        metric,
        company_id,
        year,
        category,
        severity,
        difference,
    )
    return record


def validate_roce(
    computed_roce: object,
    source_roce: object,
    *,
    company_id: str = "",
    year: str = "",
    threshold: float = ROCE_DIFF_THRESHOLD,
) -> Optional[AnomalyRecord]:
    """Cross-check computed ROCE against the ``companies.xlsx`` source value.

    Computed ROCE (from ``ratios.calculate_return_on_capital_employed``) is
    always the value used for analytics; this function only flags
    discrepancies for review — it never mutates the computed value.

    Args:
        computed_roce: ROCE percentage computed by the ratio engine.
        source_roce:   ``roce_percentage`` from companies.xlsx.
        company_id:    For log context.
        year:          For log context.
        threshold:     Absolute percentage-point difference that triggers a
                        logged anomaly (default 5.0, per Task 2).

    Returns:
        An :class:`AnomalyRecord` if the difference exceeds *threshold*,
        otherwise None.
    """
    computed_val = _safe_float(computed_roce, "computed_roce")
    source_val = _safe_float(source_roce, "source_roce")

    if computed_val is None or source_val is None:
        logger.debug(
            "ROCE cross-check skipped — missing value | company=%s | year=%s.",
            company_id,
            year,
        )
        return None

    difference = abs(computed_val - source_val)
    if difference <= threshold:
        return None

    return _build_and_log_anomaly(
        company_id=company_id,
        year=year,
        metric="ROCE",
        computed_value=computed_val,
        source_value=source_val,
        difference=difference,
    )


def validate_roe(
    computed_roe: object,
    source_roe: object,
    *,
    company_id: str = "",
    year: str = "",
    threshold: float = ROE_DIFF_THRESHOLD,
) -> Optional[AnomalyRecord]:
    """Cross-check computed ROE against the ``companies.xlsx`` source value.

    Computed ROE (from ``ratios.calculate_return_on_equity``) is always the
    value used for analytics. Known anomalies (e.g. TCS source ROE = 0.52
    while computed ROE is far higher) are logged for review but the source
    value is NEVER used to overwrite the computed value; it is retained for
    display purposes only.

    Args:
        computed_roe: ROE percentage computed by the ratio engine.
        source_roe:   ``roe_percentage`` from companies.xlsx.
        company_id:   For log context.
        year:         For log context.
        threshold:    Absolute percentage-point difference that triggers a
                       logged anomaly (default 5.0, per Task 3).

    Returns:
        An :class:`AnomalyRecord` if the difference exceeds *threshold*,
        otherwise None.
    """
    computed_val = _safe_float(computed_roe, "computed_roe")
    source_val = _safe_float(source_roe, "source_roe")

    if computed_val is None or source_val is None:
        logger.debug(
            "ROE cross-check skipped — missing value | company=%s | year=%s.",
            company_id,
            year,
        )
        return None

    difference = abs(computed_val - source_val)
    if difference <= threshold:
        return None

    return _build_and_log_anomaly(
        company_id=company_id,
        year=year,
        metric="ROE",
        computed_value=computed_val,
        source_value=source_val,
        difference=difference,
    )


def process_validation(
    company_id: str,
    year: str,
    broad_sector: str,
    computed_roce: object,
    source_roce: object,
    computed_roe: object,
    source_roe: object,
) -> list[AnomalyRecord]:
    """Run the full Day-13 edge-case validation pipeline for one company-year.

    Orchestrates the Financial-sector carve-out awareness plus ROCE/ROE
    cross-validation, returning every anomaly raised so callers (e.g. the
    population pipeline) can persist or report on them without re-deriving
    any ratio logic (DRY — delegates to ``ratios.py`` and the validators
    above for all computation).

    Note: the high_leverage_flag carve-out itself is enforced inside
    ``ratios.calculate_debt_to_equity`` / ``compute_all_ratios``; this
    function only exposes ``is_financial_company`` so calling code can make
    the same determination consistently when assembling display rows.

    Args:
        company_id:    Company identifier.
        year:           Fiscal year.
        broad_sector:   Sector string from sectors / companies.xlsx.
        computed_roce:  Computed ROCE percentage.
        source_roce:    Source ROCE percentage from companies.xlsx.
        computed_roe:   Computed ROE percentage.
        source_roe:     Source ROE percentage from companies.xlsx.

    Returns:
        List of AnomalyRecord objects logged during this validation pass
        (zero, one, or two entries).
    """
    anomalies: list[AnomalyRecord] = []

    if is_financial_company(broad_sector):
        logger.debug(
            "Financial-sector company — high_leverage_flag carve-out "
            "applies | company=%s | year=%s.",
            company_id,
            year,
        )

    roce_anomaly = validate_roce(
        computed_roce, source_roce, company_id=company_id, year=year
    )
    if roce_anomaly is not None:
        anomalies.append(roce_anomaly)

    roe_anomaly = validate_roe(
        computed_roe, source_roe, company_id=company_id, year=year
    )
    if roe_anomaly is not None:
        anomalies.append(roe_anomaly)

    return anomalies


__all__ = [
    "AnomalyRecord",
    "CATEGORY_DATA_SOURCE_ISSUE",
    "CATEGORY_VERSION_DIFFERENCE",
    "CATEGORY_FORMULA_DISCREPANCY",
    "ROCE_DIFF_THRESHOLD",
    "ROE_DIFF_THRESHOLD",
    "RATIO_EDGE_CASES_LOG_PATH",
    "is_financial_company",
    "categorise_anomaly",
    "validate_roce",
    "validate_roe",
    "log_ratio_anomaly",
    "process_validation",
]