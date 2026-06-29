"""
validator.py
============
Data Quality validation utilities for the Nifty 100 pipeline.
"""
from __future__ import annotations
import logging
from typing import Optional
import re
import math

import pandas as pd

logger = logging.getLogger(__name__)

# Re-export normalisation helpers so validator module exposes them too
from src.etl.normaliser import normalize_ticker, normalize_year  # noqa: F401


def check_dq03_fk_integrity(
    df: pd.DataFrame,
    table_name: str,
    valid_company_ids: set[str],
) -> list[dict]:
    """
    DQ-03: Detect rows whose company_id does not exist in the companies table.

    Parameters
    ----------
    df : pd.DataFrame
        Table being validated; must contain a ``company_id`` column.
    table_name : str
        Name of the table (used in the failure report).
    valid_company_ids : set[str]
        Set of known-good company IDs from the companies master table.

    Returns
    -------
    list[dict]
        One dict per failing row with keys:
        ``rule_id``, ``table``, ``company_id``, ``year``, ``reason``.
    """
    failures: list[dict] = []

    if "company_id" not in df.columns:
        logger.error("check_dq03_fk_integrity: 'company_id' column missing from %s", table_name)
        return failures

    for _, row in df.iterrows():
        cid = str(row.get("company_id", ""))
        if cid not in valid_company_ids:
            failure = {
                "rule_id": "DQ-03",
                "table": table_name,
                "company_id": cid,
                "year": row.get("year"),
                "reason": f"company_id '{cid}' not found in companies master table",
            }
            failures.append(failure)
            logger.warning("DQ-03 | %s | %s | FK violation", table_name, cid)

    return failures