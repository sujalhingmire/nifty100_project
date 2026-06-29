"""
load_to_sqlite.py — Full Data Load into SQLite
===============================================
File: src/etl/load_to_sqlite.py

Reads all 12 Excel files, cleans and normalises data,
then loads into nifty100.db in the correct FK-safe order.
Generates: output/load_audit.csv

Usage:
    python src/etl/load_to_sqlite.py

Load order (FK-safe):
    1. companies         (parent — no FK dependencies)
    2. sectors           (FK → companies)
    3. profitandloss     (FK → companies)
    4. balancesheet      (FK → companies)
    5. cashflow          (FK → companies)
    6. analysis          (FK → companies)
    7. documents         (FK → companies)
    8. prosandcons       (FK → companies)
    9. stock_prices      (FK → companies)
   10. market_cap        (FK → companies)
   11. financial_ratios  (FK → companies)
   12. peer_groups       (FK → companies)
"""

import logging
import os
import sqlite3
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, ".")
from src.etl.normaliser import normalize_ticker, normalize_year

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (read from .env or use defaults)
# ---------------------------------------------------------------------------
DB_PATH    = os.getenv("DB_PATH",   "data/nifty100.db")
RAW_DIR    = os.getenv("RAW_DIR",   "data/raw")
SUPP_DIR   = os.getenv("SUPP_DIR",  "data/raw")   # supplementary also in raw/
OUTPUT_DIR = os.getenv("OUTPUT_DIR","output")
AUDIT_CSV  = os.path.join(OUTPUT_DIR, "load_audit.csv")
SCHEMA_SQL = "db/schema.sql"


# ===========================================================================
# Helper: load one Excel file and return a clean DataFrame + rejected count
# ===========================================================================
def _read_excel(path: str, header_row: int = 1,
                normalise_ticker_col: bool = True,
                normalise_year_col: bool = True) -> tuple[pd.DataFrame, int]:
    """
    Read an Excel file and apply standard normalisations.
    Returns (clean_df, rejected_count).
    """
    df = pd.read_excel(path, header=header_row)
    rejected = 0

    # Normalise company_id
    if normalise_ticker_col and "company_id" in df.columns:
        before = len(df)
        df["company_id"] = df["company_id"].apply(normalize_ticker)
        # Reject rows where company_id became None (was null/unparseable)
        bad = df["company_id"].isna()
        rejected += bad.sum()
        df = df[~bad].copy()
        if rejected:
            logger.warning("  Rejected %d rows with null company_id in %s",
                           rejected, os.path.basename(path))

    # Normalise year (core time-series files only)
    if normalise_year_col and "year" in df.columns:
        df["year"] = df["year"].apply(normalize_year)

    # Strip whitespace from all string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda v: str(v).strip() if pd.notna(v) else v
        )

    return df, int(rejected)


# ===========================================================================
# Helper: execute the schema SQL to create all tables
# ===========================================================================
def _apply_schema(conn: sqlite3.Connection) -> None:
    """Drop and recreate all tables from db/schema.sql."""
    logger.info("Applying schema from %s", SCHEMA_SQL)
    with open(SCHEMA_SQL, "r") as f:
        sql = f.read()
    conn.executescript(sql)
    conn.commit()
    logger.info("Schema applied — all tables created")


# ===========================================================================
# Helper: write a DataFrame to SQLite and return (rows_loaded, rows_rejected)
# ===========================================================================
def _filter_orphan_rows(conn, df, table):
    """Remove rows whose company_id is not in the companies master table."""
    if "company_id" not in df.columns:
        return df, 0
    valid_ids = {r[0] for r in conn.execute("SELECT id FROM companies").fetchall()}
    mask = df["company_id"].isin(valid_ids)
    orphan_count = int((~mask).sum())
    if orphan_count:
        orphans = df.loc[~mask, "company_id"].unique().tolist()
        logger.warning("  Filtered %d orphan rows from %s (unknown: %s)",
                       orphan_count, table, orphans)
    return df[mask].copy(), orphan_count


def _load_table(conn: sqlite3.Connection, df: pd.DataFrame,
                table: str, rejected: int) -> tuple[int, int]:
    """
    Write DataFrame to SQLite table.
    Filters FK-violating rows before insert.
    Returns (rows_written, total_rejected).
    """
    # Filter orphan rows for all child tables
    if table != "companies":
        df, orphans = _filter_orphan_rows(conn, df, table)
        rejected += orphans

    # Count rows before
    before = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    # Write
    try:
        df.to_sql(table, conn, if_exists="append", index=False,
                  method="multi", chunksize=500)
    except Exception as e:
        logger.error("  ERROR loading %s: %s", table, e)
        raise

    after = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    rows_written = after - before
    logger.info("  %-22s %5d rows loaded  |  %d rejected", table, rows_written, rejected)
    return rows_written, rejected


# ===========================================================================
# Individual dataset loaders
# ===========================================================================

def load_companies(conn: sqlite3.Connection) -> dict:
    path = f"{RAW_DIR}/companies.xlsx"
    df, rej = _read_excel(path, header_row=1,
                          normalise_ticker_col=False,   # uses 'id' not 'company_id'
                          normalise_year_col=False)
    # The PK column is 'id' (NSE ticker)
    df["id"] = df["id"].apply(normalize_ticker)
    df = df[df["id"].notna()].copy()
    # Remove the source row-id if present (we keep 'id' as NSE ticker)
    rows_in = len(df)
    loaded, rej2 = _load_table(conn, df, "companies", rej)
    return {"table": "companies", "rows_in": rows_in, "rows_loaded": loaded, "rows_rejected": rej + rej2}


def load_sectors(conn: sqlite3.Connection) -> dict:
    path = f"{SUPP_DIR}/sectors.xlsx"
    df, rej = _read_excel(path, header_row=0)
    # Drop the source 'id' column — let SQLite auto-increment
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    rows_in = len(df)
    loaded, rej2 = _load_table(conn, df, "sectors", rej)
    return {"table": "sectors", "rows_in": rows_in, "rows_loaded": loaded, "rows_rejected": rej + rej2}


def load_profitandloss(conn: sqlite3.Connection) -> dict:
    path = f"{RAW_DIR}/profitandloss.xlsx"
    df, rej = _read_excel(path, header_row=1)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    # Deduplicate on (company_id, year) — keep last
    before = len(df)
    df = df.drop_duplicates(subset=["company_id", "year"], keep="last")
    rej += before - len(df)
    rows_in = len(df)
    loaded, rej2 = _load_table(conn, df, "profitandloss", rej)
    return {"table": "profitandloss", "rows_in": rows_in, "rows_loaded": loaded, "rows_rejected": rej + rej2}


def load_balancesheet(conn: sqlite3.Connection) -> dict:
    path = f"{RAW_DIR}/balancesheet.xlsx"
    df, rej = _read_excel(path, header_row=1)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    before = len(df)
    df = df.drop_duplicates(subset=["company_id", "year"], keep="last")
    rej += before - len(df)
    rows_in = len(df)
    loaded, rej2 = _load_table(conn, df, "balancesheet", rej)
    return {"table": "balancesheet", "rows_in": rows_in, "rows_loaded": loaded, "rows_rejected": rej + rej2}


def load_cashflow(conn: sqlite3.Connection) -> dict:
    path = f"{RAW_DIR}/cashflow.xlsx"
    df, rej = _read_excel(path, header_row=1)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    before = len(df)
    df = df.drop_duplicates(subset=["company_id", "year"], keep="last")
    rej += before - len(df)
    rows_in = len(df)
    loaded, rej2 = _load_table(conn, df, "cashflow", rej)
    return {"table": "cashflow", "rows_in": rows_in, "rows_loaded": loaded, "rows_rejected": rej + rej2}


def load_analysis(conn: sqlite3.Connection) -> dict:
    path = f"{RAW_DIR}/analysis.xlsx"
    df, rej = _read_excel(path, header_row=1, normalise_year_col=False)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    rows_in = len(df)
    loaded, rej2 = _load_table(conn, df, "analysis", rej)
    return {"table": "analysis", "rows_in": rows_in, "rows_loaded": loaded, "rows_rejected": rej + rej2}


def load_documents(conn: sqlite3.Connection) -> dict:
    path = f"{RAW_DIR}/documents.xlsx"
    df, rej = _read_excel(path, header_row=1, normalise_year_col=False)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    # Cast Year to integer
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    rows_in = len(df)
    loaded, rej2 = _load_table(conn, df, "documents", rej)
    return {"table": "documents", "rows_in": rows_in, "rows_loaded": loaded, "rows_rejected": rej + rej2}


def load_prosandcons(conn: sqlite3.Connection) -> dict:
    path = f"{RAW_DIR}/prosandcons.xlsx"
    df, rej = _read_excel(path, header_row=1, normalise_year_col=False)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    rows_in = len(df)
    loaded, rej2 = _load_table(conn, df, "prosandcons", rej)
    return {"table": "prosandcons", "rows_in": rows_in, "rows_loaded": loaded, "rows_rejected": rej + rej2}


def load_stock_prices(conn: sqlite3.Connection) -> dict:
    path = f"{SUPP_DIR}/stock_prices.xlsx"
    df, rej = _read_excel(path, header_row=0, normalise_year_col=False)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    rows_in = len(df)
    loaded, rej2 = _load_table(conn, df, "stock_prices", rej)
    return {"table": "stock_prices", "rows_in": rows_in, "rows_loaded": loaded, "rows_rejected": rej + rej2}


def load_market_cap(conn: sqlite3.Connection) -> dict:
    path = f"{SUPP_DIR}/market_cap.xlsx"
    df, rej = _read_excel(path, header_row=0, normalise_year_col=False)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    rows_in = len(df)
    loaded, rej2 = _load_table(conn, df, "market_cap", rej)
    return {"table": "market_cap", "rows_in": rows_in, "rows_loaded": loaded, "rows_rejected": rej + rej2}


def load_financial_ratios(conn: sqlite3.Connection) -> dict:
    path = f"{SUPP_DIR}/financial_ratios.xlsx"
    df, rej = _read_excel(path, header_row=0)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    before = len(df)
    df = df.drop_duplicates(subset=["company_id", "year"], keep="last")
    rej += before - len(df)
    rows_in = len(df)
    loaded, rej2 = _load_table(conn, df, "financial_ratios", rej)
    return {"table": "financial_ratios", "rows_in": rows_in, "rows_loaded": loaded, "rows_rejected": rej + rej2}


def load_peer_groups(conn: sqlite3.Connection) -> dict:
    path = f"{SUPP_DIR}/peer_groups.xlsx"
    df, rej = _read_excel(path, header_row=0, normalise_year_col=False)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    rows_in = len(df)
    loaded, rej2 = _load_table(conn, df, "peer_groups", rej)
    return {"table": "peer_groups", "rows_in": rows_in, "rows_loaded": loaded, "rows_rejected": rej + rej2}


# ===========================================================================
# Main loader — runs all 12 in FK-safe order
# ===========================================================================
def run_full_load() -> pd.DataFrame:
    """
    Execute the full data load pipeline.
    Returns the load_audit DataFrame.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    logger.info("=" * 60)
    logger.info("NIFTY 100 — Full SQLite Load")
    logger.info("Database: %s", DB_PATH)
    logger.info("=" * 60)

    start_time = datetime.now()

    # Remove old DB so we start fresh (idempotent re-run)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        logger.info("Removed existing database for clean reload")

    conn = sqlite3.connect(DB_PATH)

    # Enable foreign key enforcement
    conn.execute("PRAGMA foreign_keys = ON")

    # Apply schema
    _apply_schema(conn)

    # Load tables in FK-safe order
    audit_rows = []
    loaders = [
        load_companies,
        load_sectors,
        load_profitandloss,
        load_balancesheet,
        load_cashflow,
        load_analysis,
        load_documents,
        load_prosandcons,
        load_stock_prices,
        load_market_cap,
        load_financial_ratios,
        load_peer_groups,
    ]

    for loader_fn in loaders:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = loader_fn(conn)
        result["load_timestamp"] = ts
        audit_rows.append(result)

    conn.commit()
    conn.close()

    # Build audit DataFrame
    audit_df = pd.DataFrame(audit_rows, columns=[
        "table", "rows_in", "rows_loaded", "rows_rejected", "load_timestamp"
    ])

    # Save audit CSV
    audit_df.to_csv(AUDIT_CSV, index=False)

    elapsed = (datetime.now() - start_time).total_seconds()
    total_loaded   = audit_df["rows_loaded"].sum()
    total_rejected = audit_df["rows_rejected"].sum()

    logger.info("=" * 60)
    logger.info("Load complete in %.1f seconds", elapsed)
    logger.info("Total rows loaded  : %d", total_loaded)
    logger.info("Total rows rejected: %d", total_rejected)
    logger.info("Audit saved to     : %s", AUDIT_CSV)
    logger.info("=" * 60)

    return audit_df


# ===========================================================================
# Audit summary printer
# ===========================================================================
def print_audit(audit_df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("LOAD AUDIT SUMMARY")
    print("=" * 60)
    print(f"  {'Table':<25} {'Loaded':>8} {'Rejected':>9}")
    print("  " + "-" * 44)
    for _, row in audit_df.iterrows():
        print(f"  {row['table']:<25} {row['rows_loaded']:>8,} {row['rows_rejected']:>9,}")
    print("  " + "-" * 44)
    print(f"  {'TOTAL':<25} {audit_df['rows_loaded'].sum():>8,} "
          f"{audit_df['rows_rejected'].sum():>9,}")
    print("=" * 60 + "\n")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    audit = run_full_load()
    print_audit(audit)