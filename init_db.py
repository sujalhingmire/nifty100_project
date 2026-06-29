"""
init_db.py
==========
One-time database initialisation script.
Run from project root: python init_db.py

Creates nifty100.db at the project root with the full schema from db/schema.sql
"""
import sqlite3
import pathlib
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def init_database(
    db_path: str = "nifty100.db",
    schema_path: str = "db/schema.sql",
) -> None:
    schema_file = pathlib.Path(schema_path)
    if not schema_file.exists():
        logger.error("Schema file not found: %s", schema_path)
        sys.exit(1)

    sql = schema_file.read_text(encoding="utf-8")
    logger.info("Applying schema from: %s", schema_path)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(sql)
        conn.commit()
    finally:
        conn.close()

    # Verify
    conn = sqlite3.connect(db_path)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    ).fetchall()
    conn.close()

    logger.info("Database ready: %s", db_path)
    logger.info("Tables created: %s", [t[0] for t in tables])

if __name__ == "__main__":
    init_database()