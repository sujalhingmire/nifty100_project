"""
ratios.py — Financial Ratio Engine
====================================
Sprint 2 · Days 08–09 · Nifty 100 Financial Intelligence Platform

Computes profitability, leverage, and efficiency ratios for all company-year
combinations loaded from the SQLite database.  All edge cases (zero denominator,
negative equity, debt-free companies, Financial-sector carve-outs, missing values,
NaN, and Infinity) are handled explicitly.

Follows:
    PEP 8 · SOLID principles · DRY · type hints throughout · logging (no print)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel / threshold constants
# ---------------------------------------------------------------------------
HIGH_LEVERAGE_THRESHOLD: float = 5.0
OPM_MISMATCH_TOLERANCE: float = 1.0   # percentage points
FINANCIAL_SECTORS: frozenset[str] = frozenset(
    {"Financials", "financials", "FINANCIALS"}
)
ICR_WARNING_THRESHOLD: float = 1.5
DEBT_FREE_LABEL: str = "Debt Free"


# ---------------------------------------------------------------------------
# Result dataclasses — one per ratio family
# ---------------------------------------------------------------------------


@dataclass
class ProfitabilityRatios:
    """Container for Day-08 profitability KPIs."""

    net_profit_margin: Optional[float] = None
    operating_profit_margin: Optional[float] = None
    opm_mismatch: bool = False
    return_on_equity: Optional[float] = None
    return_on_capital_employed: Optional[float] = None
    return_on_assets: Optional[float] = None


@dataclass
class LeverageRatios:
    """Container for Day-09 leverage KPIs."""

    debt_to_equity: Optional[float] = None
    high_leverage_flag: bool = False
    interest_coverage_ratio: Optional[float] = None
    icr_label: str = ""
    icr_warning: bool = False
    net_debt: Optional[float] = None


@dataclass
class EfficiencyRatios:
    """Container for Day-09 efficiency KPIs."""

    asset_turnover: Optional[float] = None


@dataclass
class FinancialRatios:
    """Aggregate container holding all ratio families for a single company-year."""

    company_id: str = ""
    year: str = ""
    broad_sector: str = ""
    profitability: ProfitabilityRatios = field(
        default_factory=ProfitabilityRatios
    )
    leverage: LeverageRatios = field(default_factory=LeverageRatios)
    efficiency: EfficiencyRatios = field(default_factory=EfficiencyRatios)


# ---------------------------------------------------------------------------
# Internal guard helpers
# ---------------------------------------------------------------------------


def _is_invalid(value: object) -> bool:
    """Return True when *value* is None, NaN, or ±Infinity.

    Args:
        value: Any scalar that may arrive from a pandas cell or raw dict.

    Returns:
        True  → value cannot be safely used in arithmetic.
        False → value is a finite, usable number.
    """
    if value is None:
        return True
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return True
    return math.isnan(f) or math.isinf(f)


def _safe_float(value: object, label: str = "") -> Optional[float]:
    """Convert *value* to float; return None and log on failure.

    Args:
        value: Raw value from a data row.
        label: Human-readable field name used in log messages.

    Returns:
        Finite float, or None if conversion fails or value is invalid.
    """
    if _is_invalid(value):
        if label:
            logger.debug("Missing or invalid value for field '%s'.", label)
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        logger.warning("Cannot convert '%s' to float for field '%s'.", value, label)
        return None
    if math.isnan(result) or math.isinf(result):
        logger.warning(
            "Non-finite float (%.6g) encountered for field '%s'.", result, label
        )
        return None
    return result


def _safe_divide(
    numerator: float,
    denominator: float,
    *,
    company_id: str = "",
    year: str = "",
    ratio_name: str = "",
) -> Optional[float]:
    """Perform division and return None (with a log) when denominator ≤ 0 or 0.

    Args:
        numerator:   The top of the fraction.
        denominator: The bottom of the fraction.
        company_id:  For log context.
        year:        For log context.
        ratio_name:  Human-readable ratio name for log messages.

    Returns:
        numerator / denominator, or None on invalid denominator.
    """
    if denominator == 0:
        logger.warning(
            "Zero denominator for '%s' | company=%s | year=%s.",
            ratio_name,
            company_id,
            year,
        )
        return None
    return numerator / denominator


def _is_financial_sector(broad_sector: str) -> bool:
    """Return True when the company belongs to the Financials broad sector.

    Args:
        broad_sector: Value from sectors.broad_sector column.

    Returns:
        True if company is in Financials; False otherwise.
    """
    return broad_sector.strip() in FINANCIAL_SECTORS


# ---------------------------------------------------------------------------
# Day-08 profitability ratios
# ---------------------------------------------------------------------------


def calculate_net_profit_margin(
    net_profit: object,
    sales: object,
    *,
    company_id: str = "",
    year: str = "",
) -> Optional[float]:
    """Compute Net Profit Margin (NPM).

    Formula::

        NPM = (net_profit / sales) × 100

    Args:
        net_profit:  PAT (Profit After Tax) in ₹ Crore.
        sales:       Net revenue / total income in ₹ Crore.
        company_id:  For log context.
        year:        For log context.

    Returns:
        NPM as a percentage, or None if sales ≤ 0 or inputs are invalid.

    Examples:
        >>> calculate_net_profit_margin(100, 500)
        20.0
        >>> calculate_net_profit_margin(100, 0) is None
        True
    """
    np_val = _safe_float(net_profit, "net_profit")
    s_val = _safe_float(sales, "sales")

    if np_val is None or s_val is None:
        logger.warning(
            "NPM skipped — missing value | company=%s | year=%s.",
            company_id,
            year,
        )
        return None

    if s_val <= 0:
        logger.warning(
            "NPM skipped — sales ≤ 0 (sales=%.4g) | company=%s | year=%s.",
            s_val,
            company_id,
            year,
        )
        return None

    result = _safe_divide(
        np_val,
        s_val,
        company_id=company_id,
        year=year,
        ratio_name="Net Profit Margin",
    )
    return None if result is None else result * 100.0


def calculate_operating_profit_margin(
    operating_profit: object,
    sales: object,
    opm_percentage: object = None,
    *,
    company_id: str = "",
    year: str = "",
) -> Optional[float]:
    """Compute Operating Profit Margin (OPM) and cross-validate against source.

    Formula::

        OPM = (operating_profit / sales) × 100

    Cross-check: If ``opm_percentage`` is supplied and
    ``|computed_OPM − opm_percentage| > 1%``, a WARNING is logged.

    Args:
        operating_profit: EBITDA in ₹ Crore.
        sales:            Net revenue in ₹ Crore.
        opm_percentage:   Pre-computed OPM % from the source P&L row (optional).
        company_id:       For log context.
        year:             For log context.

    Returns:
        Computed OPM as a percentage, or None when sales ≤ 0 / inputs invalid.
    """
    op_val = _safe_float(operating_profit, "operating_profit")
    s_val = _safe_float(sales, "sales")

    if op_val is None or s_val is None:
        logger.warning(
            "OPM skipped — missing value | company=%s | year=%s.",
            company_id,
            year,
        )
        return None

    if s_val <= 0:
        logger.warning(
            "OPM skipped — sales ≤ 0 (sales=%.4g) | company=%s | year=%s.",
            s_val,
            company_id,
            year,
        )
        return None

    result = _safe_divide(
        op_val,
        s_val,
        company_id=company_id,
        year=year,
        ratio_name="Operating Profit Margin",
    )
    if result is None:
        return None

    computed_opm = result * 100.0

    # Cross-validate against source opm_percentage when available
    source_opm = _safe_float(opm_percentage, "opm_percentage")
    if source_opm is not None:
        diff = abs(computed_opm - source_opm)
        if diff > OPM_MISMATCH_TOLERANCE:
            logger.warning(
                "OPM mismatch | Company: %s | Year: %s | "
                "Expected (source): %.4f | Actual (computed): %.4f | Diff: %.4f",
                company_id,
                year,
                source_opm,
                computed_opm,
                diff,
            )

    return computed_opm


def calculate_return_on_equity(
    net_profit: object,
    equity_capital: object,
    reserves: object,
    *,
    company_id: str = "",
    year: str = "",
) -> Optional[float]:
    """Compute Return on Equity (ROE).

    Formula::

        ROE = net_profit / (equity_capital + reserves) × 100

    Args:
        net_profit:     PAT in ₹ Crore.
        equity_capital: Paid-up share capital in ₹ Crore.
        reserves:       Reserves & surplus in ₹ Crore.
        company_id:     For log context.
        year:           For log context.

    Returns:
        ROE as a percentage, or None when total equity ≤ 0 or inputs invalid.
    """
    np_val = _safe_float(net_profit, "net_profit")
    ec_val = _safe_float(equity_capital, "equity_capital")
    res_val = _safe_float(reserves, "reserves")

    if np_val is None:
        logger.warning(
            "ROE skipped — net_profit missing | company=%s | year=%s.",
            company_id,
            year,
        )
        return None

    # Treat missing equity components as zero when the other is present
    ec_val = ec_val if ec_val is not None else 0.0
    res_val = res_val if res_val is not None else 0.0

    total_equity = ec_val + res_val

    if total_equity <= 0:
        logger.warning(
            "ROE skipped — negative/zero equity (%.4g) | company=%s | year=%s.",
            total_equity,
            company_id,
            year,
        )
        return None

    result = _safe_divide(
        np_val,
        total_equity,
        company_id=company_id,
        year=year,
        ratio_name="Return on Equity",
    )
    return None if result is None else result * 100.0


def calculate_return_on_capital_employed(
    operating_profit: object,
    depreciation: object,
    equity_capital: object,
    reserves: object,
    borrowings: object,
    broad_sector: str = "",
    *,
    company_id: str = "",
    year: str = "",
) -> Optional[float]:
    """Compute Return on Capital Employed (ROCE).

    Formula::

        EBIT  = operating_profit − depreciation
        ROCE  = EBIT / (equity_capital + reserves + borrowings) × 100

    For companies in the **Financials** sector, ROCE is computed using the same
    formula but a sector-relative benchmark (rather than an absolute threshold)
    should be used downstream by the screener.  This function logs the sector
    context so callers are aware.

    Args:
        operating_profit: EBITDA in ₹ Crore.
        depreciation:     D&A in ₹ Crore.
        equity_capital:   Paid-up capital in ₹ Crore.
        reserves:         Reserves & surplus in ₹ Crore.
        borrowings:       Total debt in ₹ Crore.
        broad_sector:     Sector string from sectors table.
        company_id:       For log context.
        year:             For log context.

    Returns:
        ROCE as a percentage, or None when capital employed ≤ 0 or inputs invalid.
    """
    op_val = _safe_float(operating_profit, "operating_profit")
    dep_val = _safe_float(depreciation, "depreciation")
    ec_val = _safe_float(equity_capital, "equity_capital")
    res_val = _safe_float(reserves, "reserves")
    borrow_val = _safe_float(borrowings, "borrowings")

    if op_val is None:
        logger.warning(
            "ROCE skipped — operating_profit missing | company=%s | year=%s.",
            company_id,
            year,
        )
        return None

    dep_val = dep_val if dep_val is not None else 0.0
    ec_val = ec_val if ec_val is not None else 0.0
    res_val = res_val if res_val is not None else 0.0
    borrow_val = borrow_val if borrow_val is not None else 0.0

    ebit = op_val - dep_val
    capital_employed = ec_val + res_val + borrow_val

    if capital_employed <= 0:
        logger.warning(
            "ROCE skipped — capital employed ≤ 0 (%.4g) | company=%s | year=%s.",
            capital_employed,
            company_id,
            year,
        )
        return None

    if _is_financial_sector(broad_sector):
        logger.info(
            "ROCE for Financial sector company — sector-relative benchmark applies "
            "| company=%s | year=%s.",
            company_id,
            year,
        )

    result = _safe_divide(
        ebit,
        capital_employed,
        company_id=company_id,
        year=year,
        ratio_name="ROCE",
    )
    return None if result is None else result * 100.0


def calculate_return_on_assets(
    net_profit: object,
    total_assets: object,
    *,
    company_id: str = "",
    year: str = "",
) -> Optional[float]:
    """Compute Return on Assets (ROA).

    Formula::

        ROA = net_profit / total_assets × 100

    Args:
        net_profit:   PAT in ₹ Crore.
        total_assets: Sum of all asset-side items in ₹ Crore.
        company_id:   For log context.
        year:         For log context.

    Returns:
        ROA as a percentage, or None when total_assets ≤ 0 or inputs invalid.
    """
    np_val = _safe_float(net_profit, "net_profit")
    ta_val = _safe_float(total_assets, "total_assets")

    if np_val is None or ta_val is None:
        logger.warning(
            "ROA skipped — missing value | company=%s | year=%s.",
            company_id,
            year,
        )
        return None

    if ta_val <= 0:
        logger.warning(
            "ROA skipped — total_assets ≤ 0 (%.4g) | company=%s | year=%s.",
            ta_val,
            company_id,
            year,
        )
        return None

    result = _safe_divide(
        np_val,
        ta_val,
        company_id=company_id,
        year=year,
        ratio_name="Return on Assets",
    )
    return None if result is None else result * 100.0


# ---------------------------------------------------------------------------
# Day-09 leverage ratios
# ---------------------------------------------------------------------------


def calculate_debt_to_equity(
    borrowings: object,
    equity_capital: object,
    reserves: object,
    broad_sector: str = "",
    *,
    company_id: str = "",
    year: str = "",
) -> tuple[Optional[float], bool]:
    """Compute Debt-to-Equity (D/E) ratio with leverage flag.

    Formula::

        D/E = borrowings / (equity_capital + reserves)

    Special rules:

    * ``borrowings == 0``  →  return ``0.0`` (not None — company is debt-free).
    * ``D/E > 5`` AND sector is **not** Financials  →  ``high_leverage_flag = True``.

    Args:
        borrowings:    Total debt in ₹ Crore.
        equity_capital: Paid-up capital in ₹ Crore.
        reserves:      Reserves & surplus in ₹ Crore.
        broad_sector:  Sector string; Financials companies exempt from the flag.
        company_id:    For log context.
        year:          For log context.

    Returns:
        Tuple of (D/E ratio or None, high_leverage_flag bool).
    """
    borrow_val = _safe_float(borrowings, "borrowings")
    ec_val = _safe_float(equity_capital, "equity_capital")
    res_val = _safe_float(reserves, "reserves")

    if borrow_val is None:
        logger.warning(
            "D/E skipped — borrowings missing | company=%s | year=%s.",
            company_id,
            year,
        )
        return None, False

    # Debt-free shortcut — return 0 explicitly (not None)
    if borrow_val == 0:
        logger.debug(
            "D/E = 0 (debt-free) | company=%s | year=%s.", company_id, year
        )
        return 0.0, False

    ec_val = ec_val if ec_val is not None else 0.0
    res_val = res_val if res_val is not None else 0.0
    total_equity = ec_val + res_val

    if total_equity <= 0:
        logger.warning(
            "D/E skipped — negative/zero equity (%.4g) | company=%s | year=%s.",
            total_equity,
            company_id,
            year,
        )
        return None, False

    result = _safe_divide(
        borrow_val,
        total_equity,
        company_id=company_id,
        year=year,
        ratio_name="Debt to Equity",
    )
    if result is None:
        return None, False

    is_financial = _is_financial_sector(broad_sector)
    high_leverage_flag = (result > HIGH_LEVERAGE_THRESHOLD) and not is_financial

    if high_leverage_flag:
        logger.warning(
            "High leverage flag — D/E=%.4f > %.1f for non-Financial company "
            "| company=%s | year=%s.",
            result,
            HIGH_LEVERAGE_THRESHOLD,
            company_id,
            year,
        )

    return result, high_leverage_flag


def calculate_interest_coverage_ratio(
    operating_profit: object,
    other_income: object,
    interest: object,
    *,
    company_id: str = "",
    year: str = "",
) -> tuple[Optional[float], str, bool]:
    """Compute Interest Coverage Ratio (ICR).

    Formula::

        ICR = (operating_profit + other_income) / interest

    Special rules:

    * ``interest == 0``  →  return ``(None, "Debt Free", False)``.
    * ``ICR < 1.5``      →  ``icr_warning = True``.

    Args:
        operating_profit: EBITDA in ₹ Crore.
        other_income:     Non-operating income in ₹ Crore.
        interest:         Finance costs in ₹ Crore.
        company_id:       For log context.
        year:             For log context.

    Returns:
        Tuple of (ICR value or None, icr_label str, icr_warning bool).
    """
    op_val = _safe_float(operating_profit, "operating_profit")
    oi_val = _safe_float(other_income, "other_income")
    int_val = _safe_float(interest, "interest")

    if int_val is None:
        logger.warning(
            "ICR skipped — interest missing | company=%s | year=%s.",
            company_id,
            year,
        )
        return None, "", False

    # Debt-free company
    if int_val == 0:
        logger.debug(
            "ICR = Debt Free (interest=0) | company=%s | year=%s.",
            company_id,
            year,
        )
        return None, DEBT_FREE_LABEL, False

    if op_val is None:
        logger.warning(
            "ICR skipped — operating_profit missing | company=%s | year=%s.",
            company_id,
            year,
        )
        return None, "", False

    oi_val = oi_val if oi_val is not None else 0.0
    numerator = op_val + oi_val

    result = _safe_divide(
        numerator,
        int_val,
        company_id=company_id,
        year=year,
        ratio_name="Interest Coverage Ratio",
    )
    if result is None:
        return None, "", False

    icr_warning = result < ICR_WARNING_THRESHOLD
    if icr_warning:
        logger.warning(
            "ICR warning — ICR=%.4f < %.1f | company=%s | year=%s.",
            result,
            ICR_WARNING_THRESHOLD,
            company_id,
            year,
        )

    return result, "", icr_warning


def calculate_net_debt(
    borrowings: object,
    investments: object,
    *,
    company_id: str = "",
    year: str = "",
) -> Optional[float]:
    """Compute Net Debt.

    Formula::

        Net Debt = borrowings − investments

    Negative result means the company has more investments than debt
    (net cash positive).

    Args:
        borrowings:  Total debt in ₹ Crore.
        investments: Long-term investments used as liquid asset proxy (₹ Crore).
        company_id:  For log context.
        year:        For log context.

    Returns:
        Net Debt in ₹ Crore, or None when inputs are invalid.
    """
    borrow_val = _safe_float(borrowings, "borrowings")
    invest_val = _safe_float(investments, "investments")

    if borrow_val is None:
        logger.warning(
            "Net Debt skipped — borrowings missing | company=%s | year=%s.",
            company_id,
            year,
        )
        return None

    # investments may legitimately be 0; treat missing as 0
    invest_val = invest_val if invest_val is not None else 0.0
    return borrow_val - invest_val


def calculate_asset_turnover(
    sales: object,
    total_assets: object,
    *,
    company_id: str = "",
    year: str = "",
) -> Optional[float]:
    """Compute Asset Turnover ratio.

    Formula::

        Asset Turnover = sales / total_assets

    Args:
        sales:        Net revenue in ₹ Crore.
        total_assets: Sum of all assets in ₹ Crore.
        company_id:   For log context.
        year:         For log context.

    Returns:
        Asset Turnover (×), or None when total_assets ≤ 0 or inputs invalid.
    """
    s_val = _safe_float(sales, "sales")
    ta_val = _safe_float(total_assets, "total_assets")

    if s_val is None or ta_val is None:
        logger.warning(
            "Asset Turnover skipped — missing value | company=%s | year=%s.",
            company_id,
            year,
        )
        return None

    if ta_val <= 0:
        logger.warning(
            "Asset Turnover skipped — total_assets ≤ 0 (%.4g) "
            "| company=%s | year=%s.",
            ta_val,
            company_id,
            year,
        )
        return None

    return _safe_divide(
        s_val,
        ta_val,
        company_id=company_id,
        year=year,
        ratio_name="Asset Turnover",
    )


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------


def compute_all_ratios(
    row: dict,
    broad_sector: str = "",
) -> FinancialRatios:
    """Compute the full set of Day-08 and Day-09 ratios for a single data row.

    *row* is expected to contain keys matching the column names in the merged
    ``profitandloss`` + ``balancesheet`` query result.  Missing keys are treated
    as None.

    Args:
        row:          Dictionary with financial data for one company-year.
        broad_sector: Sector string from ``sectors.broad_sector``.

    Returns:
        A fully populated :class:`FinancialRatios` dataclass.
    """
    cid = str(row.get("company_id", ""))
    yr = str(row.get("year", ""))

    # ------------------------------------------------------------------ #
    # Profitability
    # ------------------------------------------------------------------ #
    npm = calculate_net_profit_margin(
        row.get("net_profit"),
        row.get("sales"),
        company_id=cid,
        year=yr,
    )

    computed_opm = calculate_operating_profit_margin(
        row.get("operating_profit"),
        row.get("sales"),
        opm_percentage=row.get("opm_percentage"),
        company_id=cid,
        year=yr,
    )

    # Detect mismatch flag for storage
    source_opm = _safe_float(row.get("opm_percentage"), "opm_percentage")
    opm_mismatch = False
    if computed_opm is not None and source_opm is not None:
        opm_mismatch = abs(computed_opm - source_opm) > OPM_MISMATCH_TOLERANCE

    roe = calculate_return_on_equity(
        row.get("net_profit"),
        row.get("equity_capital"),
        row.get("reserves"),
        company_id=cid,
        year=yr,
    )

    roce = calculate_return_on_capital_employed(
        row.get("operating_profit"),
        row.get("depreciation"),
        row.get("equity_capital"),
        row.get("reserves"),
        row.get("borrowings"),
        broad_sector=broad_sector,
        company_id=cid,
        year=yr,
    )

    roa = calculate_return_on_assets(
        row.get("net_profit"),
        row.get("total_assets"),
        company_id=cid,
        year=yr,
    )

    profitability = ProfitabilityRatios(
        net_profit_margin=npm,
        operating_profit_margin=computed_opm,
        opm_mismatch=opm_mismatch,
        return_on_equity=roe,
        return_on_capital_employed=roce,
        return_on_assets=roa,
    )

    # ------------------------------------------------------------------ #
    # Leverage
    # ------------------------------------------------------------------ #
    de_ratio, high_leverage = calculate_debt_to_equity(
        row.get("borrowings"),
        row.get("equity_capital"),
        row.get("reserves"),
        broad_sector=broad_sector,
        company_id=cid,
        year=yr,
    )

    icr, icr_label, icr_warn = calculate_interest_coverage_ratio(
        row.get("operating_profit"),
        row.get("other_income"),
        row.get("interest"),
        company_id=cid,
        year=yr,
    )

    net_debt = calculate_net_debt(
        row.get("borrowings"),
        row.get("investments"),
        company_id=cid,
        year=yr,
    )

    leverage = LeverageRatios(
        debt_to_equity=de_ratio,
        high_leverage_flag=high_leverage,
        interest_coverage_ratio=icr,
        icr_label=icr_label,
        icr_warning=icr_warn,
        net_debt=net_debt,
    )

    # ------------------------------------------------------------------ #
    # Efficiency
    # ------------------------------------------------------------------ #
    at_ratio = calculate_asset_turnover(
        row.get("sales"),
        row.get("total_assets"),
        company_id=cid,
        year=yr,
    )

    efficiency = EfficiencyRatios(asset_turnover=at_ratio)

    return FinancialRatios(
        company_id=cid,
        year=yr,
        broad_sector=broad_sector,
        profitability=profitability,
        leverage=leverage,
        efficiency=efficiency,
    )