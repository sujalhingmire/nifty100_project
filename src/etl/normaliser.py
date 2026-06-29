"""
normaliser.py
=============
Nifty 100 Financial Intelligence Platform

ETL normalisation utilities for ticker symbols and fiscal year values.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Exchange suffixes
_EXCHANGE_SUFFIX = re.compile(
    r"\.(NS|BO|EQ|NSE|BSE)\s*$",
    re.IGNORECASE,
)

# Mar 2024
_PAT_MON_SPACE_YYYY = re.compile(
    r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})$",
    re.IGNORECASE,
)

# Mar-23 or Mar-2023
_PAT_MON_DASH_YEAR = re.compile(
    r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(\d{2}|\d{4})$",
    re.IGNORECASE,
)

# 31-03-2024
_PAT_DDMMYYYY = re.compile(
    r"^\d{1,2}-\d{1,2}-(\d{4})$"
)

# 2024/03
_PAT_YYYY_SLASH_MM = re.compile(
    r"^(\d{4})/\d{1,2}$"
)

# 2024-03
_PAT_YYYY_DASH_MM = re.compile(
    r"^(\d{4})-\d{1,2}$"
)

# FY 2023 / CY 2022 / Mid 2020 / 2021A
_PAT_CONTAINS_4DIGIT_YEAR = re.compile(
    r"(?<!\d)(20\d{2})(?!\d)"
)


def _valid_year(year: int) -> bool:
    """Return True for valid fiscal years in the dataset."""
    return 2000 <= year <= 2099


def _checked(value: str) -> Optional[int]:
    """Convert a year string to an integer and validate it."""
    try:
        year = int(value)
    except ValueError:
        return None

    return year if _valid_year(year) else None

# ---------------------------------------------------------------------------
# normalize_ticker
# ---------------------------------------------------------------------------

def normalize_ticker(value: object) -> Optional[str]:
    """
    Normalize NSE/BSE ticker symbols.

    Examples
    --------
    reliance.ns -> RELIANCE
    sbin.bo -> SBIN
    tcs -> TCS
    """

    # None
    if value is None:
        return None

    # pandas NA / NaN
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except Exception:
        pass

    if not isinstance(value, str):
        return ""

    ticker = value.strip()

    if ticker == "":
        logger.debug("Empty ticker after stripping whitespace.")
        return ""

    # Remove exchange suffix
    ticker = _EXCHANGE_SUFFIX.sub("", ticker)

    ticker = ticker.strip().upper()

    if ticker == "":
        return ""

    logger.debug("normalize_ticker(%r) -> %r", value, ticker)

    return ticker

# ---------------------------------------------------------------------------
# normalize_year
# ---------------------------------------------------------------------------

def normalize_year(value: object) -> Optional[int]:
    """
    Normalize fiscal year values into a 4-digit integer.

    Supported examples
    ------------------
    2023
    2023.0
    "2023"
    "Mar-23"
    "Mar-2023"
    "Mar 2023"
    "Dec-22"
    "Jun-23"
    "31-03-2023"
    "2023/03"
    "2023-03"
    "FY 2023"
    "CY 2022"
    "Mid 2021"
    "2020A"
    """

    # -------------------------------------------------------
    # None
    # -------------------------------------------------------
    if value is None:
        return None

    # -------------------------------------------------------
    # pandas NA / NaN / NaT
    # -------------------------------------------------------
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except Exception:
        pass

    # -------------------------------------------------------
    # float
    # -------------------------------------------------------
    if isinstance(value, float):

        if math.isnan(value) or math.isinf(value):
            return None

        value = int(value)

    # -------------------------------------------------------
    # integer
    # -------------------------------------------------------
    if isinstance(value, int):

        if _valid_year(value):
            return value

        return value

    # -------------------------------------------------------
    # unsupported type
    # -------------------------------------------------------
    if not isinstance(value, str):
        return None

    raw = value.strip()

    if raw == "":
        return None

    # -------------------------------------------------------
    # Plain YYYY
    # -------------------------------------------------------
    if re.fullmatch(r"\d{4}", raw):

        year = int(raw)

        return year if _valid_year(year) else None

    # -------------------------------------------------------
    # Mar 2023
    # -------------------------------------------------------
    m = _PAT_MON_SPACE_YYYY.match(raw)

    if m:
        return _checked(m.group(2))

    # -------------------------------------------------------
    # Mar-23 / Mar-2023
    # -------------------------------------------------------
    m = _PAT_MON_DASH_YEAR.match(raw)

    if m:

        year = m.group(2)

        # --------
        # 2-digit year
        # --------
        if len(year) == 2:

            yy = int(year)

            # Dataset covers 2000 onwards
            return 2000 + yy

        # --------
        # 4-digit year
        # --------
        return _checked(year)

    # -------------------------------------------------------
    # DD-MM-YYYY
    # -------------------------------------------------------
    m = _PAT_DDMMYYYY.match(raw)

    if m:
        return _checked(m.group(1))

    # -------------------------------------------------------
    # YYYY/MM
    # -------------------------------------------------------
    m = _PAT_YYYY_SLASH_MM.match(raw)

    if m:
        return _checked(m.group(1))

    # -------------------------------------------------------
    # YYYY-MM
    # -------------------------------------------------------
    m = _PAT_YYYY_DASH_MM.match(raw)

    if m:
        return _checked(m.group(1))

    # -------------------------------------------------------
    # FY 2023
    # CY 2023
    # Mid 2021
    # 2021A
    # etc.
    # -------------------------------------------------------
    m = _PAT_CONTAINS_4DIGIT_YEAR.search(raw)

    if m:
        return _checked(m.group(1))

    logger.debug("normalize_year(%r) -> None", value)

    return None