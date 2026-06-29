"""
cashflow_kpis.py
================
Nifty 100 Financial Intelligence Platform
------------------------------------------
Production-grade Cash-Flow KPI engine.

KPIs implemented
----------------
1. Free Cash Flow (FCF)
2. CFO Quality Score
3. CapEx Intensity
4. FCF Conversion Rate
5. Capital Allocation Pattern

Author : Senior Python / CFA / Data Engineer
Python : 3.10+
"""

from __future__ import annotations

import csv
import logging
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants  (never hard-coded literals in business logic)
# ---------------------------------------------------------------------------
_OUTPUT_DIR = Path(os.getenv("NIFTY_OUTPUT_DIR", "output"))
_CAPITAL_ALLOCATION_CSV = _OUTPUT_DIR / "capital_allocation.csv"
_EDGE_CASE_LOG = _OUTPUT_DIR / "ratio_edge_cases.log"

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
_CFO_QUALITY_WINDOW: int = 5          # years for rolling average
_CAPEX_THRESHOLD_LIGHT: float = 3.0   # %
_CAPEX_THRESHOLD_HEAVY: float = 8.0   # %
_CFO_QUALITY_HIGH: float = 1.0
_CFO_QUALITY_MODERATE_LOW: float = 0.5

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CashFlowRow:
    """Single-year cash-flow record for one company."""
    company_id: str
    year: int
    operating_activity: float   # CFO
    investing_activity: float   # CFI
    financing_activity: float   # CFF
    sales: float
    pat: float                  # Profit After Tax
    operating_profit: float


@dataclass
class FCFResult:
    """Free Cash Flow output."""
    company_id: str
    year: int
    fcf: Optional[float]


@dataclass
class CFOQualityResult:
    """CFO Quality Score output."""
    company_id: str
    score: Optional[float]
    label: Optional[str]


@dataclass
class CapExIntensityResult:
    """CapEx Intensity output."""
    company_id: str
    year: int
    value: Optional[float]
    label: Optional[str]


@dataclass
class FCFConversionResult:
    """FCF Conversion Rate output."""
    company_id: str
    year: int
    value: Optional[float]


@dataclass
class CapitalAllocationRow:
    """Capital Allocation Pattern record (one per company-year)."""
    company_id: str
    year: int
    cfo_sign: str
    cfi_sign: str
    cff_sign: str
    pattern_label: str


@dataclass
class CompanyKPIResult:
    """Aggregated KPI results for a single company."""
    company_id: str
    fcf_results: list[FCFResult] = field(default_factory=list)
    cfo_quality: Optional[CFOQualityResult] = None
    capex_results: list[CapExIntensityResult] = field(default_factory=list)
    fcf_conversion_results: list[FCFConversionResult] = field(default_factory=list)
    capital_allocation_rows: list[CapitalAllocationRow] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Edge-case logger
# ---------------------------------------------------------------------------

def write_edge_case_log(
    company_id: str,
    year: Optional[int],
    reason: str,
    severity: str = "WARNING",
) -> None:
    """
    Append a structured edge-case entry to *ratio_edge_cases.log*.

    Parameters
    ----------
    company_id : str
        Identifier of the company being processed.
    year : int | None
        Fiscal year; ``None`` when the issue spans multiple years.
    reason : str
        Human-readable description of the anomaly.
    severity : str
        One of ``"INFO"``, ``"WARNING"``, ``"ERROR"``.
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    year_str = str(year) if year is not None else "N/A"
    line = (
        f"{timestamp} | SEVERITY={severity} | company={company_id} | "
        f"year={year_str} | reason={reason}\n"
    )
    with _EDGE_CASE_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line)

    # Mirror to the Python logging system at the matching level
    _level = getattr(logging, severity.upper(), logging.WARNING)
    logger.log(_level, "Edge-case [%s | %s]: %s", company_id, year_str, reason)


# ---------------------------------------------------------------------------
# Helper: sign string
# ---------------------------------------------------------------------------

def _sign_str(value: float) -> str:
    """Return ``'+'`` or ``'-'`` for a finite numeric value."""
    return "+" if value >= 0 else "-"


def _is_invalid(value: float, company_id: str, year: int, field_name: str) -> bool:
    """
    Return ``True`` and log if *value* is NaN, None-like, or infinite.

    Parameters
    ----------
    value : float
        The numeric value to validate.
    company_id : str
        Company being checked.
    year : int
        Fiscal year being checked.
    field_name : str
        Name of the field for log context.
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        write_edge_case_log(
            company_id,
            year,
            f"Invalid value for {field_name}: {value}",
            severity="WARNING",
        )
        return True
    return False


# ---------------------------------------------------------------------------
# KPI 1 – Free Cash Flow
# ---------------------------------------------------------------------------

def calculate_fcf(row: CashFlowRow) -> FCFResult:
    """
    Calculate Free Cash Flow for a single company-year.

    Formula
    -------
    FCF = operating_activity + investing_activity

    Negative FCF is valid and is never forced positive.

    Parameters
    ----------
    row : CashFlowRow
        A single year's cash-flow record.

    Returns
    -------
    FCFResult
        Contains ``fcf`` which may be ``None`` on invalid input.
    """
    cfo, cfi = row.operating_activity, row.investing_activity

    if _is_invalid(cfo, row.company_id, row.year, "operating_activity"):
        return FCFResult(company_id=row.company_id, year=row.year, fcf=None)
    if _is_invalid(cfi, row.company_id, row.year, "investing_activity"):
        return FCFResult(company_id=row.company_id, year=row.year, fcf=None)

    fcf = cfo + cfi
    logger.debug("FCF | %s | %d | FCF=%.2f", row.company_id, row.year, fcf)
    return FCFResult(company_id=row.company_id, year=row.year, fcf=fcf)


# ---------------------------------------------------------------------------
# KPI 2 – CFO Quality Score
# ---------------------------------------------------------------------------

def calculate_cfo_quality(
    rows: Sequence[CashFlowRow],
) -> CFOQualityResult:
    """
    Calculate CFO Quality Score as a rolling 5-year average of CFO / PAT.

    Rules
    -----
    * Uses the most-recent ``_CFO_QUALITY_WINDOW`` years (sorted ascending).
    * If PAT == 0 for any year the ratio for that year is ``None``; only
      valid ratios contribute to the average.
    * If no valid ratios exist, returns ``score=None, label=None``.

    Classification
    --------------
    * average > 1.0  → "High Quality"
    * 0.5 ≤ average ≤ 1.0 → "Moderate"
    * average < 0.5  → "Accrual Risk"

    Parameters
    ----------
    rows : Sequence[CashFlowRow]
        All available annual rows for one company (any order).

    Returns
    -------
    CFOQualityResult
        Both ``score`` (float | None) and ``label`` (str | None).
    """
    if not rows:
        return CFOQualityResult(company_id="unknown", score=None, label=None)

    company_id = rows[0].company_id
    sorted_rows = sorted(rows, key=lambda r: r.year)
    window = sorted_rows[-_CFO_QUALITY_WINDOW:]

    ratios: list[float] = []
    for r in window:
        if _is_invalid(r.operating_activity, company_id, r.year, "operating_activity"):
            continue
        if _is_invalid(r.pat, company_id, r.year, "pat"):
            continue
        if r.pat == 0:
            write_edge_case_log(
                company_id,
                r.year,
                "PAT is zero; CFO/PAT ratio skipped for this year",
                severity="WARNING",
            )
            continue
        if r.pat < 0:
            write_edge_case_log(
                company_id,
                r.year,
                f"PAT is negative ({r.pat}); ratio included but flagged",
                severity="INFO",
            )
        ratio = r.operating_activity / r.pat
        ratios.append(ratio)
        logger.debug(
            "CFO Quality | %s | %d | CFO=%.2f PAT=%.2f ratio=%.4f",
            company_id, r.year, r.operating_activity, r.pat, ratio,
        )

    if not ratios:
        write_edge_case_log(
            company_id,
            None,
            "No valid CFO/PAT ratios; CFO Quality Score is None",
            severity="WARNING",
        )
        return CFOQualityResult(company_id=company_id, score=None, label=None)

    avg = sum(ratios) / len(ratios)

    if avg > _CFO_QUALITY_HIGH:
        label = "High Quality"
    elif avg >= _CFO_QUALITY_MODERATE_LOW:
        label = "Moderate"
    else:
        label = "Accrual Risk"

    logger.info(
        "CFO Quality | %s | avg=%.4f | label=%s", company_id, avg, label
    )
    return CFOQualityResult(company_id=company_id, score=round(avg, 6), label=label)


# ---------------------------------------------------------------------------
# KPI 3 – CapEx Intensity
# ---------------------------------------------------------------------------

def calculate_capex_intensity(row: CashFlowRow) -> CapExIntensityResult:
    """
    Calculate CapEx Intensity for a single company-year.

    Formula
    -------
    CapEx Intensity = abs(investing_activity) / sales * 100

    Classification
    --------------
    * < 3%   → "Asset Light"
    * 3–8%   → "Moderate"
    * ≥ 8%   → "Capital Intensive"

    Parameters
    ----------
    row : CashFlowRow

    Returns
    -------
    CapExIntensityResult
        ``value`` and ``label`` are ``None`` when sales == 0.
    """
    if _is_invalid(row.investing_activity, row.company_id, row.year, "investing_activity"):
        return CapExIntensityResult(
            company_id=row.company_id, year=row.year, value=None, label=None
        )
    if _is_invalid(row.sales, row.company_id, row.year, "sales"):
        return CapExIntensityResult(
            company_id=row.company_id, year=row.year, value=None, label=None
        )
    if row.sales == 0:
        write_edge_case_log(
            row.company_id,
            row.year,
            "Sales is zero; CapEx Intensity cannot be calculated (division by zero)",
            severity="WARNING",
        )
        return CapExIntensityResult(
            company_id=row.company_id, year=row.year, value=None, label=None
        )
    if row.sales < 0:
        write_edge_case_log(
            row.company_id,
            row.year,
            f"Sales is negative ({row.sales}); CapEx Intensity flagged",
            severity="WARNING",
        )

    intensity = abs(row.investing_activity) / row.sales * 100.0

    if intensity < _CAPEX_THRESHOLD_LIGHT:
        label = "Asset Light"
    elif intensity < _CAPEX_THRESHOLD_HEAVY:
        label = "Moderate"
    else:
        label = "Capital Intensive"

    logger.debug(
        "CapEx Intensity | %s | %d | %.4f%% | %s",
        row.company_id, row.year, intensity, label,
    )
    return CapExIntensityResult(
        company_id=row.company_id,
        year=row.year,
        value=round(intensity, 6),
        label=label,
    )


# ---------------------------------------------------------------------------
# KPI 4 – FCF Conversion Rate
# ---------------------------------------------------------------------------

def calculate_fcf_conversion(
    row: CashFlowRow, fcf: Optional[float]
) -> FCFConversionResult:
    """
    Calculate FCF Conversion Rate for a single company-year.

    Formula
    -------
    FCF Conversion = FCF / operating_profit * 100

    Parameters
    ----------
    row : CashFlowRow
    fcf : float | None
        Pre-computed FCF value for this row.

    Returns
    -------
    FCFConversionResult
        ``value`` is ``None`` when operating_profit == 0 or fcf is invalid.
    """
    if fcf is None:
        return FCFConversionResult(company_id=row.company_id, year=row.year, value=None)

    if _is_invalid(row.operating_profit, row.company_id, row.year, "operating_profit"):
        return FCFConversionResult(company_id=row.company_id, year=row.year, value=None)

    if row.operating_profit == 0:
        write_edge_case_log(
            row.company_id,
            row.year,
            "operating_profit is zero; FCF Conversion Rate undefined (division by zero)",
            severity="WARNING",
        )
        return FCFConversionResult(company_id=row.company_id, year=row.year, value=None)

    conversion = fcf / row.operating_profit * 100.0
    logger.debug(
        "FCF Conversion | %s | %d | FCF=%.2f EBIT=%.2f rate=%.4f%%",
        row.company_id, row.year, fcf, row.operating_profit, conversion,
    )
    return FCFConversionResult(
        company_id=row.company_id, year=row.year, value=round(conversion, 6)
    )


# ---------------------------------------------------------------------------
# KPI 5 – Capital Allocation Pattern
# ---------------------------------------------------------------------------

_PATTERN_MAP: dict[tuple[str, str, str], str] = {
    ("+", "-", "-"): "Reinvestor",
    ("+", "+", "-"): "Liquidating Assets",
    ("-", "+", "+"): "Distress Signal",
    ("-", "-", "+"): "Growth Funded by Debt",
    ("+", "+", "+"): "Cash Accumulator",
    ("-", "-", "-"): "Pre-Revenue",
    ("+", "-", "+"): "Mixed",
}

# Override when CFO Quality is High
_SHAREHOLDER_RETURNS_SIGNS = ("+", "-", "-")


def classify_capital_allocation(
    row: CashFlowRow,
    cfo_quality_label: Optional[str] = None,
) -> CapitalAllocationRow:
    """
    Classify the Capital Allocation Pattern using CFO / CFI / CFF signs.

    Classification Rules
    --------------------
    Sign triplet → label mapping defined in ``_PATTERN_MAP``.
    Special override: ``(+, -, -)`` with CFO Quality == "High Quality"
    → "Shareholder Returns".

    Parameters
    ----------
    row : CashFlowRow
    cfo_quality_label : str | None
        Label from :func:`calculate_cfo_quality`; used for the
        "Shareholder Returns" override.

    Returns
    -------
    CapitalAllocationRow
    """
    for field_name, val in [
        ("operating_activity", row.operating_activity),
        ("investing_activity", row.investing_activity),
        ("financing_activity", row.financing_activity),
    ]:
        if _is_invalid(val, row.company_id, row.year, field_name):
            write_edge_case_log(
                row.company_id,
                row.year,
                f"Capital allocation anomaly: invalid {field_name}",
                severity="ERROR",
            )
            return CapitalAllocationRow(
                company_id=row.company_id,
                year=row.year,
                cfo_sign="?",
                cfi_sign="?",
                cff_sign="?",
                pattern_label="Unknown",
            )

    cfo_sign = _sign_str(row.operating_activity)
    cfi_sign = _sign_str(row.investing_activity)
    cff_sign = _sign_str(row.financing_activity)

    signs = (cfo_sign, cfi_sign, cff_sign)
    label = _PATTERN_MAP.get(signs, "Mixed")

    # Override: (+, -, -) + High Quality CFO → Shareholder Returns
    if signs == _SHAREHOLDER_RETURNS_SIGNS and cfo_quality_label == "High Quality":
        label = "Shareholder Returns"

    logger.debug(
        "Capital Allocation | %s | %d | signs=%s | label=%s",
        row.company_id, row.year, signs, label,
    )
    return CapitalAllocationRow(
        company_id=row.company_id,
        year=row.year,
        cfo_sign=cfo_sign,
        cfi_sign=cfi_sign,
        cff_sign=cff_sign,
        pattern_label=label,
    )


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def generate_capital_allocation_csv(rows: list[CapitalAllocationRow]) -> Path:
    """
    Write *capital_allocation.csv* from a list of :class:`CapitalAllocationRow`.

    Parameters
    ----------
    rows : list[CapitalAllocationRow]
        All capital allocation rows across all companies.

    Returns
    -------
    Path
        Absolute path of the written file.
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["company_id", "year", "cfo_sign", "cfi_sign", "cff_sign", "pattern_label"]

    with _CAPITAL_ALLOCATION_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    logger.info(
        "Capital allocation CSV written: %s (%d rows)",
        _CAPITAL_ALLOCATION_CSV,
        len(rows),
    )
    return _CAPITAL_ALLOCATION_CSV.resolve()


# ---------------------------------------------------------------------------
# Per-company orchestration
# ---------------------------------------------------------------------------

def _validate_and_clean_rows(
    rows: Sequence[CashFlowRow],
) -> list[CashFlowRow]:
    """
    Validate, deduplicate, and sort rows for one company.

    Handles
    -------
    * Duplicate year entries (keep first occurrence, log warning)
    * Unsorted years (sort ascending)
    * Logs any anomalies

    Parameters
    ----------
    rows : Sequence[CashFlowRow]

    Returns
    -------
    list[CashFlowRow]
        Cleaned, sorted rows.
    """
    if not rows:
        return []

    company_id = rows[0].company_id
    seen_years: dict[int, CashFlowRow] = {}

    for r in rows:
        if r.year in seen_years:
            write_edge_case_log(
                company_id,
                r.year,
                "Duplicate year detected; keeping first occurrence",
                severity="WARNING",
            )
        else:
            seen_years[r.year] = r

    cleaned = sorted(seen_years.values(), key=lambda x: x.year)
    logger.debug("Validated %d rows for company %s", len(cleaned), company_id)
    return cleaned


def process_company_cashflows(
    rows: Sequence[CashFlowRow],
) -> CompanyKPIResult:
    """
    Compute all five KPIs for a single company.

    Parameters
    ----------
    rows : Sequence[CashFlowRow]
        All annual cash-flow rows for one company.

    Returns
    -------
    CompanyKPIResult
    """
    if not rows:
        logger.warning("process_company_cashflows called with empty rows")
        return CompanyKPIResult(company_id="unknown")

    company_id = rows[0].company_id
    clean_rows = _validate_and_clean_rows(rows)

    if not clean_rows:
        return CompanyKPIResult(company_id=company_id)

    # KPI 2 first so we have the quality label for capital allocation
    cfo_quality = calculate_cfo_quality(clean_rows)

    fcf_results: list[FCFResult] = []
    capex_results: list[CapExIntensityResult] = []
    fcf_conversion_results: list[FCFConversionResult] = []
    capital_allocation_rows: list[CapitalAllocationRow] = []

    for row in clean_rows:
        # KPI 1
        fcf_result = calculate_fcf(row)
        fcf_results.append(fcf_result)

        # KPI 3
        capex_results.append(calculate_capex_intensity(row))

        # KPI 4
        fcf_conversion_results.append(
            calculate_fcf_conversion(row, fcf_result.fcf)
        )

        # KPI 5
        capital_allocation_rows.append(
            classify_capital_allocation(row, cfo_quality.label)
        )

    logger.info(
        "KPIs computed for company=%s | years=%d",
        company_id,
        len(clean_rows),
    )
    return CompanyKPIResult(
        company_id=company_id,
        fcf_results=fcf_results,
        cfo_quality=cfo_quality,
        capex_results=capex_results,
        fcf_conversion_results=fcf_conversion_results,
        capital_allocation_rows=capital_allocation_rows,
    )


# ---------------------------------------------------------------------------
# Batch orchestration (92 companies × multiple years)
# ---------------------------------------------------------------------------

def process_all_companies(
    df: pd.DataFrame,
) -> tuple[list[CompanyKPIResult], Path]:
    """
    Process the full Nifty-100 dataset.

    Expected DataFrame columns
    --------------------------
    ``company_id``, ``year``, ``operating_activity``, ``investing_activity``,
    ``financing_activity``, ``sales``, ``pat``, ``operating_profit``

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data; may contain string numbers, nulls, duplicates.

    Returns
    -------
    tuple[list[CompanyKPIResult], Path]
        All company results and the path to the capital allocation CSV.
    """
    required_cols = {
        "company_id", "year", "operating_activity", "investing_activity",
        "financing_activity", "sales", "pat", "operating_profit",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Coerce numeric columns; invalid strings → NaN
    numeric_cols = list(required_cols - {"company_id"})
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Log rows with any NaN after coercion
    nan_mask = df[numeric_cols].isna().any(axis=1)
    for _, bad_row in df[nan_mask].iterrows():
        write_edge_case_log(
            str(bad_row.get("company_id", "unknown")),
            int(bad_row["year"]) if pd.notna(bad_row.get("year")) else None,
            "Row contains NaN / non-numeric values after coercion",
            severity="WARNING",
        )

    all_results: list[CompanyKPIResult] = []
    all_capital_rows: list[CapitalAllocationRow] = []

    for company_id, group in df.groupby("company_id", sort=True):
        rows: list[CashFlowRow] = []
        for _, r in group.iterrows():
            # Skip entirely-NaN rows
            if any(
                pd.isna(r[c]) for c in numeric_cols
            ):
                continue
            rows.append(
                CashFlowRow(
                    company_id=str(company_id),
                    year=int(r["year"]),
                    operating_activity=float(r["operating_activity"]),
                    investing_activity=float(r["investing_activity"]),
                    financing_activity=float(r["financing_activity"]),
                    sales=float(r["sales"]),
                    pat=float(r["pat"]),
                    operating_profit=float(r["operating_profit"]),
                )
            )

        result = process_company_cashflows(rows)
        all_results.append(result)
        all_capital_rows.extend(result.capital_allocation_rows)

    csv_path = generate_capital_allocation_csv(all_capital_rows)
    logger.info(
        "Batch processing complete | companies=%d | capital_allocation_rows=%d",
        len(all_results),
        len(all_capital_rows),
    )
    return all_results, csv_path