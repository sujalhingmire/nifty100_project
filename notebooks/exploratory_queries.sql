-- =============================================================================
-- exploratory_queries.sql — Sprint 1 Business Analysis Queries
-- File: db/exploratory_queries.sql
-- Run with: sqlite3 data/nifty100.db < db/exploratory_queries.sql
-- Or open in DB Browser for SQLite / VS Code SQLite extension
-- =============================================================================

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- QUERY 1: Total companies loaded — confirm 92
-- ---------------------------------------------------------------------------
SELECT
    COUNT(*)              AS total_companies,
    COUNT(DISTINCT id)    AS unique_tickers
FROM companies;

-- ---------------------------------------------------------------------------
-- QUERY 2: Top 10 companies by latest-year revenue (Sales)
-- ---------------------------------------------------------------------------
SELECT
    p.company_id,
    c.company_name,
    s.broad_sector,
    p.sales            AS revenue_cr,
    p.net_profit       AS profit_cr,
    ROUND(p.net_profit / p.sales * 100, 1) AS npm_pct
FROM profitandloss p
JOIN companies c ON c.id = p.company_id
JOIN sectors    s ON s.company_id = p.company_id
WHERE p.year = (
    SELECT MAX(year) FROM profitandloss WHERE company_id = p.company_id
)
ORDER BY p.sales DESC
LIMIT 10;

-- ---------------------------------------------------------------------------
-- QUERY 3: Top 10 companies by Return on Equity (ROE %)
-- Latest year only
-- ---------------------------------------------------------------------------
SELECT
    fr.company_id,
    c.company_name,
    s.broad_sector,
    ROUND(fr.return_on_equity_pct, 1)  AS roe_pct,
    ROUND(fr.debt_to_equity, 2)        AS de_ratio,
    ROUND(fr.net_profit_margin_pct, 1) AS npm_pct
FROM financial_ratios fr
JOIN companies c ON c.id = fr.company_id
JOIN sectors   s ON s.company_id = fr.company_id
WHERE fr.year = (
    SELECT MAX(year) FROM financial_ratios WHERE company_id = fr.company_id
)
AND fr.return_on_equity_pct IS NOT NULL
ORDER BY fr.return_on_equity_pct DESC
LIMIT 10;

-- ---------------------------------------------------------------------------
-- QUERY 4: Sector distribution — companies and avg ROE per sector
-- ---------------------------------------------------------------------------
SELECT
    s.broad_sector,
    COUNT(DISTINCT s.company_id)                AS company_count,
    ROUND(AVG(fr.return_on_equity_pct), 1)      AS avg_roe_pct,
    ROUND(AVG(fr.operating_profit_margin_pct),1) AS avg_opm_pct,
    ROUND(AVG(fr.debt_to_equity), 2)            AS avg_de_ratio
FROM sectors s
LEFT JOIN financial_ratios fr ON fr.company_id = s.company_id
    AND fr.year = (
        SELECT MAX(year) FROM financial_ratios WHERE company_id = fr.company_id
    )
GROUP BY s.broad_sector
ORDER BY company_count DESC;

-- ---------------------------------------------------------------------------
-- QUERY 5: Debt analysis — Top 10 most indebted companies (by borrowings)
-- ---------------------------------------------------------------------------
SELECT
    b.company_id,
    c.company_name,
    s.broad_sector,
    b.borrowings                                      AS total_debt_cr,
    ROUND(b.borrowings / NULLIF(b.equity_capital + b.reserves, 0), 2) AS de_ratio,
    b.year
FROM balancesheet b
JOIN companies c ON c.id = b.company_id
JOIN sectors   s ON s.company_id = b.company_id
WHERE b.year = (
    SELECT MAX(year) FROM balancesheet WHERE company_id = b.company_id
)
AND b.borrowings IS NOT NULL
ORDER BY b.borrowings DESC
LIMIT 10;

-- ---------------------------------------------------------------------------
-- QUERY 6: Profit growth trend — Revenue and PAT for last 5 years (TCS example)
-- Change 'TCS' to any ticker
-- ---------------------------------------------------------------------------
SELECT
    company_id,
    year,
    sales              AS revenue_cr,
    net_profit         AS pat_cr,
    ROUND(net_profit / NULLIF(sales, 0) * 100, 1) AS npm_pct,
    eps
FROM profitandloss
WHERE company_id = 'TCS'
ORDER BY year DESC
LIMIT 5;

-- ---------------------------------------------------------------------------
-- QUERY 7: Cash flow analysis — Companies with positive CFO every year (2020+)
-- Strong signal: consistent cash generation from operations
-- ---------------------------------------------------------------------------
SELECT
    company_id,
    COUNT(*) AS positive_cfo_years,
    ROUND(AVG(operating_activity), 0)  AS avg_cfo_cr,
    ROUND(MIN(operating_activity), 0)  AS worst_cfo_cr,
    ROUND(MAX(operating_activity), 0)  AS best_cfo_cr
FROM cashflow
WHERE year >= 2020
AND operating_activity > 0
GROUP BY company_id
HAVING COUNT(*) >= 4                  -- positive CFO in at least 4 of last 5 years
ORDER BY avg_cfo_cr DESC
LIMIT 15;

-- ---------------------------------------------------------------------------
-- QUERY 8: Top dividend companies — Highest dividend yield (latest year)
-- ---------------------------------------------------------------------------
SELECT
    mc.company_id,
    c.company_name,
    s.broad_sector,
    ROUND(mc.dividend_yield_pct, 2)    AS div_yield_pct,
    ROUND(mc.pe_ratio, 1)              AS pe_ratio,
    ROUND(mc.pb_ratio, 2)              AS pb_ratio
FROM market_cap mc
JOIN companies c ON c.id = mc.company_id
JOIN sectors   s ON s.company_id = mc.company_id
WHERE mc.year = (SELECT MAX(year) FROM market_cap WHERE company_id = mc.company_id)
AND mc.dividend_yield_pct IS NOT NULL
ORDER BY mc.dividend_yield_pct DESC
LIMIT 10;

-- ---------------------------------------------------------------------------
-- QUERY 9: Most profitable companies — by Net Profit Margin (latest year)
-- Only companies with sales > 5000 Cr (large-cap filter)
-- ---------------------------------------------------------------------------
SELECT
    p.company_id,
    c.company_name,
    s.broad_sector,
    ROUND(p.sales, 0)                              AS sales_cr,
    ROUND(p.net_profit, 0)                         AS profit_cr,
    ROUND(p.net_profit / NULLIF(p.sales,0)*100, 1) AS npm_pct,
    ROUND(p.opm_percentage, 1)                     AS opm_pct
FROM profitandloss p
JOIN companies c ON c.id = p.company_id
JOIN sectors   s ON s.company_id = p.company_id
WHERE p.year = (SELECT MAX(year) FROM profitandloss WHERE company_id = p.company_id)
AND p.sales > 5000
AND p.net_profit > 0
ORDER BY npm_pct DESC
LIMIT 10;

-- ---------------------------------------------------------------------------
-- QUERY 10: Data completeness report — Year coverage per company
-- Flags companies with < 10 years of P&L data
-- ---------------------------------------------------------------------------
SELECT
    p.company_id,
    c.company_name,
    COUNT(DISTINCT p.year)   AS pl_years,
    COUNT(DISTINCT b.year)   AS bs_years,
    COUNT(DISTINCT cf.year)  AS cf_years,
    MIN(p.year)              AS earliest_year,
    MAX(p.year)              AS latest_year,
    CASE
        WHEN COUNT(DISTINCT p.year) >= 10 THEN 'FULL'
        WHEN COUNT(DISTINCT p.year) >= 5  THEN 'PARTIAL'
        ELSE 'LOW'
    END AS coverage_status
FROM profitandloss p
JOIN companies c    ON c.id = p.company_id
LEFT JOIN balancesheet b  ON b.company_id = p.company_id
LEFT JOIN cashflow cf     ON cf.company_id = p.company_id
GROUP BY p.company_id, c.company_name
ORDER BY pl_years ASC, p.company_id;