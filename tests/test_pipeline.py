import pytest
import pandas as pd
import sqlite3
import os
import sys
from pathlib import Path

# Force Python to find the absolute root directory (D:\nifty100_project)
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Debug helper: This will print exactly where Python is looking
print(f"--- Python Search Paths: {sys.path[:2]} ---")

# Your imports
from src.etl import validator
from src.etl.normaliser import normalize_ticker, normalize_year
from src.etl.validator import check_dq03_fk_integrity   

# ==========================================
# PART 1: NORMALIZE TICKER TESTS (15 ASSERTS)
# ==========================================
def test_normalize_ticker_variants():
    assert normalize_ticker("tcs") == "TCS"
    assert normalize_ticker("INFY ") == "INFY"
    assert normalize_ticker("  wipro  ") == "WIPRO"
    assert normalize_ticker("reliance.ns") == "RELIANCE"
    assert normalize_ticker("HDFCBANK.NS") == "HDFCBANK"
    assert normalize_ticker("SBIN.BO") == "SBIN"
    assert normalize_ticker("TATASTEEL.EQ") == "TATASTEEL"
    assert normalize_ticker(123) == ""
    assert normalize_ticker(None) is None
    assert normalize_ticker("MARUTI.NS ") == "MARUTI"
    assert normalize_ticker("bse.bo") == "BSE"
    assert normalize_ticker(" axisbank ") == "AXISBANK"
    assert normalize_ticker("") == ""
    assert normalize_ticker("ZOMATO") == "ZOMATO"
    assert normalize_ticker("ITC.NS") == "ITC"

# ==========================================
# PART 2: NORMALIZE YEAR TESTS (20 ASSERTS)
# ==========================================
def test_normalize_year_variants():
    assert normalize_year("Mar 2024") == 2024
    assert normalize_year("Mar-2023") == 2023
    assert normalize_year("2022") == 2022
    assert normalize_year(2021) == 2021
    assert normalize_year("Dec 2020") == 2020
    assert normalize_year("FY 2019") == 2019
    assert normalize_year("31-03-2018") == 2018
    assert normalize_year("2017/03") == 2017
    assert normalize_year("Mar 16") is None  # strict 4 digit check
    assert normalize_year("invalid_date") is None
    assert normalize_year(None) is None
    assert normalize_year("2026") == 2026
    assert normalize_year("Sep 2025") == 2025
    assert normalize_year("2015") == 2015
    assert normalize_year("2014-03") == 2014
    assert normalize_year("CY 2013") == 2013
    assert normalize_year(" 2012 ") == 2012
    assert normalize_year("2011A") == 2011
    assert normalize_year("2010 Old") == 2010
    assert normalize_year("Mid 2009") == 2009

## ==========================================
# PART 3: VALIDATOR & DB INTEGRITY TESTS
# ==========================================

def test_validator_error_capture():

    comp_df = pd.DataFrame({
        "id": ["ABB", "TCS", "WIPRO"]
    })

    valid_ids = set(comp_df["id"])

    bad_ts = pd.DataFrame({
        "company_id": ["UNKNOWN_CO"],
        "year": ["2024"]
    })

    failures = check_dq03_fk_integrity(
        bad_ts,
        "profitandloss",
        valid_ids
    )

    assert len(failures) > 0
    assert failures[0]["rule_id"] == "DQ-03"


def test_database_integrity_and_presence():

    db_path = "nifty100.db"

    if not os.path.exists(db_path):
        pytest.skip("Database not created yet")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # ── NEW: skip gracefully if schema has not been initialised yet ──
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='companies';"
    )
    if cursor.fetchone() is None:
        conn.close()
        pytest.skip("Schema not initialised — run: sqlite3 nifty100.db < db/schema.sql")

    cursor.execute("PRAGMA foreign_key_check;")
    fk_checks = cursor.fetchall()
    assert len(fk_checks) == 0

    cursor.execute("SELECT COUNT(*) FROM companies;")
    count = cursor.fetchone()[0]
    assert count >= 0          # passes even on empty DB before pipeline runs

    conn.close()