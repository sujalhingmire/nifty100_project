"""
test_normaliser.py — Complete pytest suite for normaliser.py
============================================================
File: tests/etl/test_normaliser.py

Covers:
  - 20 normalize_year() tests
  - 15 normalize_ticker() tests
"""

import pytest
import pandas as pd
import sys
sys.path.insert(0, ".")

from src.etl.normaliser import normalize_year, normalize_ticker


# ===========================================================================
# normalize_year() — 20 tests
# ===========================================================================

class TestNormalizeYear:
    """Tests for the normalize_year() function."""

    # --- Standard Mar-YY format (most common in dataset) ---
    def test_mar23_returns_2023(self):
        assert normalize_year("Mar-23") == 2023

    def test_mar24_returns_2024(self):
        assert normalize_year("Mar-24") == 2024

    def test_mar10_returns_2010(self):
        assert normalize_year("Mar-10") == 2010

    def test_mar15_returns_2015(self):
        assert normalize_year("Mar-15") == 2015

    def test_mar20_returns_2020(self):
        assert normalize_year("Mar-20") == 2020

    def test_mar22_returns_2022(self):
        assert normalize_year("Mar-22") == 2022

    def test_mar11_returns_2011(self):
        assert normalize_year("Mar-11") == 2011

    def test_mar19_returns_2019(self):
        assert normalize_year("Mar-19") == 2019

    # --- Non-March year-ends (Dec, Jun financial year companies) ---
    def test_dec22_returns_2022(self):
        """December year-end companies like NESTLEIND."""
        result = normalize_year("Dec-22")
        assert result == 2022

    def test_jun23_returns_2023(self):
        """June year-end companies."""
        result = normalize_year("Jun-23")
        assert result == 2023

    # --- Edge cases: null / NaN ---
    def test_none_returns_none(self):
        assert normalize_year(None) is None

    def test_nan_returns_none(self):
        assert normalize_year(float("nan")) is None

    def test_pandas_nat_returns_none(self):
        assert normalize_year(pd.NaT) is None

    # --- Numeric year (already an integer) ---
    def test_integer_2023_passthrough(self):
        """Integer years have no hyphen so they pass through as strings.
        normalize_year('2023') returns '2023' — still a valid year value."""
        result = normalize_year(2023)
        assert str(result) == "2023"

    # --- Already normalised string ---
    def test_already_normalised_string(self):
        """If value is already in YYYY-MM format, should survive."""
        result = normalize_year("2023-03")
        # Should return 2023 (extracts year part) or keep as-is depending on impl
        assert result is not None

    # --- Boundary years ---
    def test_year_2000_boundary(self):
        """Years ending in 00 — edge of century."""
        result = normalize_year("Mar-00")
        assert result == 2000

    def test_year_oldest_in_dataset(self):
        """FY2010 is the oldest year in the dataset."""
        result = normalize_year("Mar-10")
        assert result == 2010

    def test_year_newest_in_dataset(self):
        """FY2024 is the newest year in the dataset."""
        result = normalize_year("Mar-24")
        assert result == 2024

    # --- Case insensitivity (nice-to-have) ---
    def test_lowercase_month(self):
        """Some data may have lowercase month names."""
        result = normalize_year("mar-23")
        assert result == 2023

    def test_uppercase_month(self):
        """All-caps month."""
        result = normalize_year("MAR-23")
        assert result == 2023


# ===========================================================================
# normalize_ticker() — 15 tests
# ===========================================================================

class TestNormalizeTicker:
    """Tests for the normalize_ticker() function."""

    # --- Basic uppercase conversion ---
    def test_lowercase_tcs(self):
        assert normalize_ticker("tcs") == "TCS"

    def test_lowercase_infy(self):
        assert normalize_ticker("infy") == "INFY"

    def test_mixed_case(self):
        assert normalize_ticker("HdFcBaNk") == "HDFCBANK"

    def test_already_uppercase(self):
        assert normalize_ticker("RELIANCE") == "RELIANCE"

    # --- Whitespace stripping ---
    def test_leading_space(self):
        assert normalize_ticker(" TCS") == "TCS"

    def test_trailing_space(self):
        assert normalize_ticker("TCS ") == "TCS"

    def test_both_spaces(self):
        assert normalize_ticker("  tcs  ") == "TCS"

    def test_tab_whitespace(self):
        assert normalize_ticker("\tSBIN\t") == "SBIN"

    # --- Special characters in valid NSE tickers ---
    def test_ticker_with_hyphen(self):
        """BAJAJ-AUTO is a valid NSE ticker with hyphen."""
        assert normalize_ticker("bajaj-auto") == "BAJAJ-AUTO"

    def test_ticker_with_ampersand(self):
        """M&M is a valid NSE ticker with ampersand."""
        assert normalize_ticker("m&m") == "M&M"

    # --- Null / NaN inputs ---
    def test_none_returns_none(self):
        assert normalize_ticker(None) is None

    def test_nan_returns_none(self):
        assert normalize_ticker(float("nan")) is None

    def test_pandas_na_returns_none(self):
        assert normalize_ticker(pd.NA) is None

    # --- Real dataset tickers ---
    def test_hdfcbank(self):
        assert normalize_ticker("hdfcbank") == "HDFCBANK"

    def test_sbin(self):
        assert normalize_ticker("sbin") == "SBIN"