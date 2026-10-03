"""
data_cleaning.py
================
Loads Clean_Data.csv, validates every column, documents all data-quality
issues, and returns a cleaned DataFrame together with a structured quality
report.

Design rules applied here:
- The original source file is never modified.
- Duplicate rows are DETECTED and REPORTED but NOT removed, so that the
  churn-model methodology stays consistent with the Masterclass 3 baseline.
- Invalid dates are flagged (Order_Date_Valid = False) but the rows are kept.
- Missing Revenue values remain NaN; they are never replaced with zero or
  an arbitrary average.
- Negative Profit values (losses) are preserved as-is.
- "Unknown Customer", "Unknown Product", and "Unknown Region" strings are
  flagged but the rows are retained for sales analytics.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CSV_PATH = "Clean_Data.csv"

VALID_CATEGORIES = {"Apparel", "Beauty", "Electronics", "Furniture", "Home"}
VALID_REGIONS = {
    "Bangalore", "Chennai", "Delhi", "Hyderabad",
    "Kolkata", "Mumbai", "Pune",
}
VALID_CUSTOMER_TYPES = {"New Customer", "Repeat Customer", "Unknown Customer"}
VALID_PROFIT_STATUSES = {"Profit", "Loss", "Break-even"}

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flag_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an `Is_Duplicate` boolean column.
    The FIRST occurrence of every duplicated row is marked False;
    all subsequent occurrences are marked True.
    Rows are never dropped — detection only.
    """
    df["Is_Duplicate"] = df.duplicated(keep="first")
    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to parse Order_Date with multiple common formats.
    Rows where parsing fails (e.g. the literal string "Invalid Date") get
    NaT in Order_Date and Order_Date_Valid = False.
    The original raw string is preserved in Order_Date_Raw.
    """
    df["Order_Date_Raw"] = df["Order_Date"].astype(str).str.strip()

    # Fast-path: remove literal "Invalid Date" so pandas doesn't hang trying to infer its format
    raw_dates = df["Order_Date_Raw"].replace("Invalid Date", np.nan)

    parsed = pd.to_datetime(raw_dates, format="%m/%d/%Y", errors="coerce")
    # Fallback: try dayfirst for any still-NaT rows
    mask_nat = parsed.isna() & raw_dates.notna()
    if mask_nat.any():
        fallback = pd.to_datetime(
            raw_dates[mask_nat], dayfirst=True, errors="coerce"
        )
        parsed[mask_nat] = fallback

    df["Order_Date"] = parsed
    df["Order_Date_Valid"] = df["Order_Date"].notna()
    return df


def _clean_revenue_profit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Revenue and Profit to numeric.
    - Blank Revenue → NaN  (never replaced with zero or a mean).
    - Zero Revenue rows are flagged with Revenue_Is_Zero.
    - Negative Profit values are preserved; they represent real losses.
    """
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
    df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce")

    df["Revenue_Missing"] = df["Revenue"].isna()
    df["Revenue_Is_Zero"] = df["Revenue"].eq(0)
    return df


def _flag_unknown_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean flag columns for anomalous string values that are kept in
    the dataset but need to be excludable for specific analyses.
    """
    df["Customer_Unknown"] = (
        df["Customer_ID"].astype(str).str.strip() == "Unknown Customer"
    )
    df["Product_Unknown"] = (
        df["Product"].astype(str).str.strip() == "Unknown Product"
    )
    df["Region_Unknown"] = (
        df["Region"].astype(str).str.strip() == "Unknown Region"
    )
    return df


def _standardise_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading/trailing whitespace from all object columns.
    Standardise Customer_Type casing to title-case.
    """
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    df["Customer_Type"] = df["Customer_Type"].str.title()
    df["Category"] = df["Category"].str.title()
    df["Region"] = df["Region"].str.title()
    df["Profit_Status"] = df["Profit_Status"].str.strip()
    return df


def _validate_profit_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-check the Profit_Status label against the actual Profit value.
    Adds a Profit_Status_Mismatch column for rows where the label disagrees
    with the sign of Profit.
    """
    def expected_status(profit):
        if pd.isna(profit):
            return None
        if profit > 0:
            return "Profit"
        if profit < 0:
            return "Loss"
        return "Break-even"

    expected = df["Profit"].apply(expected_status)
    df["Profit_Status_Mismatch"] = (
        expected.notna() & (df["Profit_Status"] != expected)
    )
    return df


def _add_month_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a numeric Month_Num column for calendar ordering.
    Rows with Month = 'Unknown' receive NaN.
    """
    month_map = {m: i + 1 for i, m in enumerate(MONTH_ORDER)}
    df["Month_Num"] = df["Month"].map(month_map)
    return df


# ---------------------------------------------------------------------------
# Quality report builder
# ---------------------------------------------------------------------------

def _build_quality_report(df: pd.DataFrame) -> dict:
    """
    Compile a structured dictionary of all data-quality issues found.
    This is displayed in the Streamlit dashboard's Data Quality expander.
    """
    total = len(df)

    report = {
        "total_rows": total,
        "total_columns": len(df.columns),

        # Date issues
        "invalid_date_count": int((~df["Order_Date_Valid"]).sum()),
        "invalid_date_pct": round(100 * (~df["Order_Date_Valid"]).mean(), 1),

        # Duplicate rows (detected, NOT removed)
        "duplicate_row_count": int(df["Is_Duplicate"].sum()),
        "duplicate_row_note": (
            "Duplicates detected and reported. NOT removed — required for "
            "methodological consistency with Masterclass 3 baseline results."
        ),

        # Revenue issues
        "missing_revenue_count": int(df["Revenue_Missing"].sum()),
        "missing_revenue_pct": round(100 * df["Revenue_Missing"].mean(), 1),
        "zero_revenue_count": int(df["Revenue_Is_Zero"].sum()),
        "zero_revenue_note": (
            "Missing Revenue values remain NaN. They are excluded from "
            "revenue calculations but the rows are preserved in the dataset."
        ),

        # Unknown / anomalous strings
        "unknown_customer_count": int(df["Customer_Unknown"].sum()),
        "unknown_product_count": int(df["Product_Unknown"].sum()),
        "unknown_region_count": int(df["Region_Unknown"].sum()),

        # Profit breakdown
        "loss_rows": int((df["Profit_Status"] == "Loss").sum()),
        "break_even_rows": int((df["Profit_Status"] == "Break-even").sum()),
        "profit_status_mismatch_count": int(df["Profit_Status_Mismatch"].sum()),

        # Month quality
        "unknown_month_count": int((df["Month"] == "Unknown").sum()),

        # Valid-date subset size (used by churn model)
        "valid_date_rows": int(df["Order_Date_Valid"].sum()),
        "valid_date_and_revenue_rows": int(
            (df["Order_Date_Valid"] & ~df["Revenue_Missing"]).sum()
        ),
    }
    return report


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_clean(filepath: str = CSV_PATH) -> tuple[pd.DataFrame, dict]:
    """
    Load Clean_Data.csv, apply all validation and cleaning steps, and return:

        df_clean      : cleaned DataFrame (all original rows retained)
        quality_report: structured dict of data-quality findings

    Parameters
    ----------
    filepath : str
        Path to the CSV file. Defaults to CSV_PATH.

    Returns
    -------
    tuple[pd.DataFrame, dict]
    """
    # --- Load ---
    df = pd.read_csv(
        filepath,
        dtype=str,           # read everything as string first; convert below
        keep_default_na=False,  # prevent pandas treating "NA"/"" as NaN yet
        skipinitialspace=True,
    )

    # Replace genuine empty strings with NaN now that we control the types
    df.replace("", np.nan, inplace=True)

    # --- Pipeline ---
    df = _standardise_text(df)
    df = _flag_duplicates(df)
    df = _parse_dates(df)
    df = _clean_revenue_profit(df)
    df = _flag_unknown_fields(df)
    df = _validate_profit_status(df)
    df = _add_month_order(df)

    # Ensure Quantity is integer-safe numeric
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

    # Build the quality report before returning
    quality_report = _build_quality_report(df)

    return df, quality_report


# ---------------------------------------------------------------------------
# Convenience: sub-filtered views used by other modules
# ---------------------------------------------------------------------------

def get_valid_date_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with a successfully parsed Order_Date."""
    return df[df["Order_Date_Valid"]].copy()


def get_churn_eligible_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows suitable for churn feature engineering:
    - Valid Order_Date
    - Non-missing Revenue
    - Known Customer_ID (excludes 'Unknown Customer')
    """
    return df[
        df["Order_Date_Valid"]
        & ~df["Revenue_Missing"]
        & ~df["Customer_Unknown"]
    ].copy()


# ---------------------------------------------------------------------------
# Quick self-test (run directly: python data_cleaning.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df_clean, report = load_and_clean()

    print("=" * 60)
    print("DATA CLEANING — SELF-TEST")
    print("=" * 60)
    print(f"  Rows loaded            : {report['total_rows']}")
    print(f"  Columns                : {report['total_columns']}")
    print()
    print("  --- Date Quality ---")
    print(f"  Invalid dates          : {report['invalid_date_count']} ({report['invalid_date_pct']}%)")
    print(f"  Valid-date rows        : {report['valid_date_rows']}")
    print()
    print("  --- Revenue Quality ---")
    print(f"  Missing Revenue        : {report['missing_revenue_count']} ({report['missing_revenue_pct']}%)")
    print(f"  Zero Revenue           : {report['zero_revenue_count']}")
    print()
    print("  --- Duplicates ---")
    print(f"  Duplicate rows detected: {report['duplicate_row_count']} (NOT removed)")
    print()
    print("  --- Unknown / Anomalous Strings ---")
    print(f"  Unknown Customer       : {report['unknown_customer_count']}")
    print(f"  Unknown Product        : {report['unknown_product_count']}")
    print(f"  Unknown Region         : {report['unknown_region_count']}")
    print()
    print("  --- Profit Quality ---")
    print(f"  Loss rows              : {report['loss_rows']}")
    print(f"  Break-even rows        : {report['break_even_rows']}")
    print(f"  Profit_Status mismatch : {report['profit_status_mismatch_count']}")
    print()
    print("  --- Churn-model eligible rows ---")
    print(f"  Valid date + Revenue   : {report['valid_date_and_revenue_rows']}")
    print()
    print("  dtypes of key columns:")
    for col in ["Order_Date", "Revenue", "Profit", "Quantity"]:
        print(f"    {col:20s}: {df_clean[col].dtype}")
    print()
    print("  Flag columns added:", [c for c in df_clean.columns if c not in [
        "Order_ID","Order_Date","Customer_ID","Product","Category",
        "Region","Quantity","Revenue","Profit","Customer_Type","Month","Profit_Status"
    ]])
    print("=" * 60)
    print("data_cleaning.py: PASS")
