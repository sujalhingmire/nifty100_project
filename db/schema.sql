-- =============================================================================
-- NIFTY 100 FINANCIAL INTELLIGENCE PLATFORM
-- SQLite Database Schema — Version 1.0
-- File: db/schema.sql
-- Run with: sqlite3 data/nifty100.db < db/schema.sql
-- =============================================================================

-- Enable foreign key enforcement (MUST be first)
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- =============================================================================
-- DROP TABLES (safe re-run — drop in reverse FK dependency order)
-- =============================================================================
DROP TABLE IF EXISTS peer_groups;
DROP TABLE IF EXISTS stock_prices;
DROP TABLE IF EXISTS market_cap;
DROP TABLE IF EXISTS financial_ratios;
DROP TABLE IF EXISTS prosandcons;
DROP TABLE IF EXISTS documents;
DROP TABLE IF EXISTS analysis;
DROP TABLE IF EXISTS cashflow;
DROP TABLE IF EXISTS balancesheet;
DROP TABLE IF EXISTS profitandloss;
DROP TABLE IF EXISTS sectors;
DROP TABLE IF EXISTS companies;

-- =============================================================================
-- TABLE 1: companies  (Master Reference — Parent of all FK relationships)
-- =============================================================================
CREATE TABLE companies (
    id                VARCHAR(12)  PRIMARY KEY,   -- NSE Ticker (PK for all tables)
    company_logo      TEXT,                        -- URL to logo image (may 404)
    company_name      VARCHAR(200) NOT NULL,       -- Full legal company name
    chart_link        TEXT,                        -- TradingView chart URL
    about_company     TEXT,                        -- Business description
    website           TEXT,                        -- Official corporate website
    nse_profile       TEXT,                        -- NSE India equity page URL
    bse_profile       TEXT,                        -- BSE India stock page URL
    face_value        NUMERIC,                     -- Share face value in ₹
    book_value        NUMERIC,                     -- Book value per share (latest)
    roce_percentage   NUMERIC,                     -- Pre-computed ROCE %
    roe_percentage    NUMERIC                      -- Pre-computed ROE %
);

CREATE INDEX idx_companies_name ON companies(company_name);

-- =============================================================================
-- TABLE 2: sectors  (Company Sector Mapping — 1:1 with companies)
-- =============================================================================
CREATE TABLE sectors (
    id                  INTEGER      PRIMARY KEY AUTOINCREMENT,
    company_id          VARCHAR(12)  NOT NULL,
    broad_sector        VARCHAR(100) NOT NULL,     -- 11 macro sectors
    sub_sector          VARCHAR(100),              -- 33 sub-sectors
    index_weight_pct    NUMERIC,                   -- Estimated Nifty 100 weight %
    market_cap_category VARCHAR(20),               -- Large Cap / Mid Cap

    FOREIGN KEY (company_id) REFERENCES companies(id)
        ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE UNIQUE INDEX idx_sectors_company ON sectors(company_id);
CREATE INDEX idx_sectors_broad ON sectors(broad_sector);

-- =============================================================================
-- TABLE 3: profitandloss  (Annual P&L Statements)
-- =============================================================================
CREATE TABLE profitandloss (
    id                  INTEGER      PRIMARY KEY AUTOINCREMENT,
    company_id          VARCHAR(12)  NOT NULL,
    year                VARCHAR(10)  NOT NULL,     -- Normalised: YYYY-MM
    sales               NUMERIC,                   -- Net revenue (₹ Crore)
    expenses            NUMERIC,                   -- Total operating expenses
    operating_profit    NUMERIC,                   -- EBITDA (₹ Crore)
    opm_percentage      NUMERIC,                   -- Operating Profit Margin %
    other_income        NUMERIC,                   -- Non-operating income
    interest            NUMERIC,                   -- Finance costs
    depreciation        NUMERIC,                   -- D&A (₹ Crore)
    profit_before_tax   NUMERIC,                   -- PBT (₹ Crore)
    tax_percentage      NUMERIC,                   -- Effective tax rate %
    net_profit          NUMERIC,                   -- PAT — can be negative
    eps                 NUMERIC,                   -- Earnings Per Share (₹)
    dividend_payout     NUMERIC,                   -- Dividend payout ratio %

    FOREIGN KEY (company_id) REFERENCES companies(id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    UNIQUE (company_id, year)                      -- DQ-02: no duplicate year rows
);

CREATE INDEX idx_pl_company ON profitandloss(company_id);
CREATE INDEX idx_pl_year ON profitandloss(year);

-- =============================================================================
-- TABLE 4: balancesheet  (Annual Balance Sheets)
-- =============================================================================
CREATE TABLE balancesheet (
    id                INTEGER      PRIMARY KEY AUTOINCREMENT,
    company_id        VARCHAR(12)  NOT NULL,
    year              VARCHAR(10)  NOT NULL,
    equity_capital    NUMERIC,                     -- Paid-up share capital
    reserves          NUMERIC,                     -- Reserves & surplus
    borrowings        NUMERIC,                     -- Total debt (₹ Crore)
    other_liabilities NUMERIC,                     -- Trade payables + others
    total_liabilities NUMERIC,                     -- Sum of all liabilities
    fixed_assets      NUMERIC,                     -- Net fixed assets
    cwip              NUMERIC,                     -- Capital Work In Progress
    investments       NUMERIC,                     -- Long-term investments
    other_asset       NUMERIC,                     -- Current + other assets
    total_assets      NUMERIC,                     -- Sum of all assets

    FOREIGN KEY (company_id) REFERENCES companies(id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    UNIQUE (company_id, year)
);

CREATE INDEX idx_bs_company ON balancesheet(company_id);
CREATE INDEX idx_bs_year ON balancesheet(year);

-- =============================================================================
-- TABLE 5: cashflow  (Annual Cash Flow Statements)
-- =============================================================================
CREATE TABLE cashflow (
    id                  INTEGER      PRIMARY KEY AUTOINCREMENT,
    company_id          VARCHAR(12)  NOT NULL,
    year                VARCHAR(10)  NOT NULL,
    operating_activity  NUMERIC,                   -- CFO (positive = good)
    investing_activity  NUMERIC,                   -- CFI (usually negative)
    financing_activity  NUMERIC,                   -- CFF (variable)
    net_cash_flow       NUMERIC,                   -- CFO + CFI + CFF

    FOREIGN KEY (company_id) REFERENCES companies(id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    UNIQUE (company_id, year)
);

CREATE INDEX idx_cf_company ON cashflow(company_id);
CREATE INDEX idx_cf_year ON cashflow(year);

-- =============================================================================
-- TABLE 6: analysis  (Pre-Computed Growth Metrics — Partial Coverage)
-- =============================================================================
CREATE TABLE analysis (
    id                       INTEGER      PRIMARY KEY AUTOINCREMENT,
    company_id               VARCHAR(12)  NOT NULL,
    compounded_sales_growth  TEXT,                 -- Raw text: "10 Years: 21%"
    compounded_profit_growth TEXT,                 -- Raw text: "5 Years: 6%"
    stock_price_cagr         TEXT,                 -- Raw text: "10 Years: 15%"
    roe                      TEXT,                 -- Raw text: "10 Years: 17%"

    FOREIGN KEY (company_id) REFERENCES companies(id)
        ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE INDEX idx_analysis_company ON analysis(company_id);

-- =============================================================================
-- TABLE 7: documents  (Annual Report URL Repository)
-- =============================================================================
CREATE TABLE documents (
    id             INTEGER      PRIMARY KEY AUTOINCREMENT,
    company_id     VARCHAR(12)  NOT NULL,
    year           INTEGER,                        -- Calendar year of report
    Annual_Report  TEXT,                           -- BSE PDF URL

    FOREIGN KEY (company_id) REFERENCES companies(id)
        ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE INDEX idx_docs_company ON documents(company_id);
CREATE INDEX idx_docs_year ON documents(year);

-- =============================================================================
-- TABLE 8: prosandcons  (Qualitative Investment Insights)
-- =============================================================================
CREATE TABLE prosandcons (
    id         INTEGER      PRIMARY KEY AUTOINCREMENT,
    company_id VARCHAR(12)  NOT NULL,
    pros       TEXT,                               -- Positive observation
    cons       TEXT,                               -- Risk observation

    FOREIGN KEY (company_id) REFERENCES companies(id)
        ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE INDEX idx_pac_company ON prosandcons(company_id);

-- =============================================================================
-- TABLE 9: stock_prices  (Monthly OHLCV — Simulated)
-- =============================================================================
CREATE TABLE stock_prices (
    id             INTEGER      PRIMARY KEY AUTOINCREMENT,
    company_id     VARCHAR(12)  NOT NULL,
    date           VARCHAR(10)  NOT NULL,          -- YYYY-MM-DD format
    open_price     NUMERIC,
    high_price     NUMERIC,
    low_price      NUMERIC,
    close_price    NUMERIC,
    volume         INTEGER,
    adjusted_close NUMERIC,

    FOREIGN KEY (company_id) REFERENCES companies(id)
        ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE INDEX idx_sp_company ON stock_prices(company_id);
CREATE INDEX idx_sp_date ON stock_prices(date);

-- =============================================================================
-- TABLE 10: market_cap  (Annual Valuation Multiples — Simulated)
-- =============================================================================
CREATE TABLE market_cap (
    id                      INTEGER      PRIMARY KEY AUTOINCREMENT,
    company_id              VARCHAR(12)  NOT NULL,
    year                    INTEGER,               -- Calendar year
    market_cap_crore        NUMERIC,               -- Market cap (₹ Crore)
    enterprise_value_crore  NUMERIC,               -- EV (₹ Crore)
    pe_ratio                NUMERIC,               -- Price-to-Earnings
    pb_ratio                NUMERIC,               -- Price-to-Book
    ev_ebitda               NUMERIC,               -- EV/EBITDA multiple
    dividend_yield_pct      NUMERIC,               -- Dividend yield %

    FOREIGN KEY (company_id) REFERENCES companies(id)
        ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE INDEX idx_mc_company ON market_cap(company_id);
CREATE INDEX idx_mc_year ON market_cap(year);

-- =============================================================================
-- TABLE 11: financial_ratios  (Pre-Computed KPI Table)
-- =============================================================================
CREATE TABLE financial_ratios (
    id                         INTEGER      PRIMARY KEY AUTOINCREMENT,
    company_id                 VARCHAR(12)  NOT NULL,
    year                       VARCHAR(10)  NOT NULL,
    net_profit_margin_pct      NUMERIC,
    operating_profit_margin_pct NUMERIC,
    return_on_equity_pct       NUMERIC,
    debt_to_equity             NUMERIC,
    interest_coverage          NUMERIC,
    asset_turnover             NUMERIC,
    free_cash_flow_cr          NUMERIC,
    capex_cr                   NUMERIC,
    earnings_per_share         NUMERIC,
    book_value_per_share       NUMERIC,
    dividend_payout_ratio_pct  NUMERIC,
    total_debt_cr              NUMERIC,
    cash_from_operations_cr    NUMERIC,

    FOREIGN KEY (company_id) REFERENCES companies(id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    UNIQUE (company_id, year)
);

CREATE INDEX idx_fr_company ON financial_ratios(company_id);
CREATE INDEX idx_fr_year ON financial_ratios(year);
CREATE INDEX idx_fr_roe ON financial_ratios(return_on_equity_pct);

-- =============================================================================
-- TABLE 12: peer_groups  (Peer Comparison Groups)
-- =============================================================================
CREATE TABLE peer_groups (
    id              INTEGER      PRIMARY KEY AUTOINCREMENT,
    peer_group_name VARCHAR(100) NOT NULL,
    company_id      VARCHAR(12)  NOT NULL,
    is_benchmark    INTEGER      DEFAULT 0,        -- 1 = benchmark company

    FOREIGN KEY (company_id) REFERENCES companies(id)
        ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE INDEX idx_pg_group ON peer_groups(peer_group_name);
CREATE INDEX idx_pg_company ON peer_groups(company_id);

-- =============================================================================
-- VERIFICATION QUERIES (run after loading to confirm schema is correct)
-- =============================================================================
-- SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name;
-- PRAGMA foreign_key_check;
-- SELECT COUNT(*) FROM companies;