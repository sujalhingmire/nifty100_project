"""
review.py — Day 06 Data Quality Review Script
==============================================
File: src/etl/review.py

Checks 5 random companies across all time-series tables.
Verifies year coverage, identifies missing years,
flags companies with less than 5 years of history.
Generates: output/review_report.csv

Usage:
    python src/etl/review.py
"""

import logging
import os
import random
import sqlite3
import sys

import pandas as pd

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DB_PATH    = os.getenv("DB_PATH",   "data/nifty100.db")
OUTPUT_DIR = os.getenv("OUTPUT_DIR","output")
REPORT_CSV = os.path.join(OUTPUT_DIR, "review_report.csv")

# Expected financial year range (normalised integer years)
EXPECTED_YEARS = set(range(2010, 2025))   # 2010–2024 = 15 years


def _get_connection() -> sqlite3.Connection:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database not found at '{DB_PATH}'. "
            "Run load_to_sqlite.py first."
        )
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def get_all_companies(conn: sqlite3.Connection) -> list[str]:
    """Return list of all company_ids from the companies table."""
    rows = conn.execute("SELECT id FROM companies ORDER BY id").fetchall()
    return [r[0] for r in rows]


def get_year_coverage(conn: sqlite3.Connection,
                      company_id: str,
                      table: str) -> dict:
    """
    Return year coverage stats for one company in one table.
    Years in the DB are stored as integer or YYYY-MM strings —
    we extract the 4-digit year part for comparison.
    """
    sql = f"SELECT year FROM {table} WHERE company_id = ? ORDER BY year"
    rows = conn.execute(sql, (company_id,)).fetchall()
    stored_years_raw = [r[0] for r in rows]

    # Parse year to integer (handles both 2023 and "2023-03")
    def _parse_yr(y):
        """Parse year from any format: 2023, 'Mar 2013', '2023-03', etc."""
        import re
        s = str(y)
        # Match any 4-digit year in the string
        m = re.search(r'(20\d{2}|19\d{2})', s)
        if m:
            return int(m.group(1))
        try:
            return int(s[:4])
        except Exception:
            return None

    stored_years = sorted(set(filter(None, [_parse_yr(y) for y in stored_years_raw])))

    if not stored_years:
        return {
            "company_id":       company_id,
            "table":            table,
            "years_in_db":      0,
            "first_year":       None,
            "last_year":        None,
            "missing_years":    str(sorted(EXPECTED_YEARS)),
            "has_min_5_years":  False,
            "coverage_note":    "No data found",
        }

    stored_set   = set(stored_years)
    missing      = sorted(EXPECTED_YEARS - stored_set)
    year_count   = len(stored_years)

    return {
        "company_id":      company_id,
        "table":           table,
        "years_in_db":     year_count,
        "first_year":      min(stored_years),
        "last_year":       max(stored_years),
        "missing_years":   str(missing) if missing else "None",
        "has_min_5_years": year_count >= 5,
        "coverage_note":   (
            f"OK — {year_count} years" if year_count >= 10
            else f"LOW — only {year_count} year(s)" if year_count < 5
            else f"PARTIAL — {year_count} years"
        ),
    }


def run_review(sample_size: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Review data quality for a random sample of companies.
    Returns a DataFrame report.
    """
    conn = _get_connection()

    all_companies = get_all_companies(conn)
    if not all_companies:
        raise ValueError("No companies found in database. Load data first.")

    logger.info("Total companies in DB: %d", len(all_companies))

    # Pick 5 random companies (reproducible with seed)
    random.seed(seed)
    sample = random.sample(all_companies, min(sample_size, len(all_companies)))
    logger.info("Reviewing companies: %s", sample)

    tables = ["profitandloss", "balancesheet", "cashflow"]
    records = []

    for company_id in sample:
        logger.info("  Checking: %s", company_id)
        for table in tables:
            rec = get_year_coverage(conn, company_id, table)
            records.append(rec)

    # Also check ALL companies for the "less than 5 years" flag
    logger.info("Checking all %d companies for minimum coverage...", len(all_companies))
    low_coverage = []
    for table in tables:
        sql = f"""
            SELECT company_id, COUNT(DISTINCT year) AS year_count
            FROM {table}
            GROUP BY company_id
            HAVING year_count < 5
        """
        rows = conn.execute(sql).fetchall()
        for row in rows:
            low_coverage.append({
                "company_id":     row[0],
                "table":          table,
                "years_in_db":    row[1],
                "first_year":     None,
                "last_year":      None,
                "missing_years":  "See full coverage check",
                "has_min_5_years": False,
                "coverage_note":  f"BELOW MINIMUM — {row[1]} year(s)",
            })

    conn.close()

    all_records = records + low_coverage
    report_df = pd.DataFrame(all_records)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_df.to_csv(REPORT_CSV, index=False)

    logger.info("Review report saved to: %s", REPORT_CSV)
    return report_df


def print_review(report_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("DAY 06 — DATA QUALITY REVIEW REPORT")
    print("=" * 70)

    # Section 1: sampled companies detail
    sampled = report_df[report_df["coverage_note"].str.startswith("OK") |
                        report_df["coverage_note"].str.startswith("PARTIAL") |
                        report_df["coverage_note"].str.startswith("LOW")]
    print(f"\n{'Company':<15} {'Table':<18} {'Years':>5} {'First':>5} {'Last':>5}  Status")
    print("-" * 70)
    for _, row in sampled.iterrows():
        print(f"{row['company_id']:<15} {row['table']:<18} "
              f"{row['years_in_db']:>5} {str(row['first_year']):>5} "
              f"{str(row['last_year']):>5}  {row['coverage_note']}")

    # Section 2: companies with < 5 years
    below = report_df[~report_df["has_min_5_years"]]
    if not below.empty:
        print(f"\n⚠️  Companies with < 5 years of data: {len(below)}")
        for _, row in below.head(20).iterrows():
            print(f"   {row['company_id']:<15} {row['table']:<18} {row['years_in_db']} yr(s)")
    else:
        print("\n✅ All companies have ≥ 5 years of data")

    print(f"\nFull report: {REPORT_CSV}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    report = run_review(sample_size=5, seed=42)
    print_review(report)