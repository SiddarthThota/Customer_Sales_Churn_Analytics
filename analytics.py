"""
analytics.py
============
Descriptive sales analytics for the Customer Sales & Churn Analytics project.

All functions accept the cleaned DataFrame produced by data_cleaning.load_and_clean()
and return plain DataFrames ready for Plotly charts or KPI display in Streamlit.

Design rules:
- Revenue calculations exclude rows where Revenue is NaN (missing values are
  never filled with zero or a mean).
- Zero-Revenue rows are included in counts but their ₹0 contribution is
  honest — they are NOT excluded from aggregations.
- "Unknown Region" rows are excluded from region aggregations so charts
  show only the seven known cities.
- "Unknown Product" rows are excluded from product aggregations.
- Duplicate rows are NOT removed here; data_cleaning.py detected them but
  the Masterclass 3 methodology keeps them in the pipeline.
- Month ordering follows the calendar (January → December), not alphabetical.
- All monetary values are in Indian Rupees (₹).
"""

import pandas as pd
import numpy as np

# Calendar month order for sorting
MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


# ---------------------------------------------------------------------------
# KPI summary
# ---------------------------------------------------------------------------

def get_kpis(df: pd.DataFrame) -> dict:
    """
    Return the five top-level KPIs as a plain dict.

    - Total Revenue : sum of non-NaN Revenue values
    - Total Profit  : sum of all Profit values (negatives preserved → losses)
    - Total Orders  : count of rows (each row = one order)
    - Total Quantity: sum of Quantity
    - Unique Customers: count of distinct Customer_IDs excluding 'Unknown Customer'

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from data_cleaning.load_and_clean().

    Returns
    -------
    dict with keys: total_revenue, total_profit, total_orders,
                    total_quantity, unique_customers
    """
    revenue_rows = df[~df["Revenue_Missing"]]

    total_revenue = revenue_rows["Revenue"].sum()
    total_profit = df["Profit"].sum()
    total_orders = len(df)
    total_quantity = df["Quantity"].sum()
    unique_customers = df.loc[~df["Customer_Unknown"], "Customer_ID"].nunique()

    return {
        "total_revenue": total_revenue,
        "total_profit": total_profit,
        "total_orders": total_orders,
        "total_quantity": int(total_quantity),
        "unique_customers": unique_customers,
    }


# ---------------------------------------------------------------------------
# Monthly revenue trend
# ---------------------------------------------------------------------------

def revenue_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Revenue by calendar month using only rows that have:
      - a valid Order_Date (Order_Date_Valid == True)
      - a non-NaN Revenue value

    The returned DataFrame is sorted in calendar order and includes
    a Month_Num column so Plotly can keep the correct x-axis order.

    Returns
    -------
    pd.DataFrame with columns: Month, Month_Num, Revenue
    """
    eligible = df[df["Order_Date_Valid"] & ~df["Revenue_Missing"]].copy()

    # Extract month name from the parsed date (more reliable than the Month column
    # which contains 'Unknown' for the 1 194 invalid-date rows)
    eligible["Month_Name"] = eligible["Order_Date"].dt.strftime("%B")

    monthly = (
        eligible
        .groupby("Month_Name", as_index=False)["Revenue"]
        .sum()
        .rename(columns={"Month_Name": "Month"})
    )

    # Add sort key and enforce calendar order
    month_map = {m: i + 1 for i, m in enumerate(MONTH_ORDER)}
    monthly["Month_Num"] = monthly["Month"].map(month_map)
    monthly = monthly.sort_values("Month_Num").reset_index(drop=True)

    return monthly[["Month", "Month_Num", "Revenue"]]


# ---------------------------------------------------------------------------
# Revenue by Category
# ---------------------------------------------------------------------------

def revenue_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total Revenue by Category, excluding NaN Revenue rows.

    Returns
    -------
    pd.DataFrame with columns: Category, Revenue
        sorted descending by Revenue.
    """
    eligible = df[~df["Revenue_Missing"]].copy()

    cat = (
        eligible
        .groupby("Category", as_index=False)["Revenue"]
        .sum()
        .sort_values("Revenue", ascending=False)
        .reset_index(drop=True)
    )
    return cat


# ---------------------------------------------------------------------------
# Revenue by Region
# ---------------------------------------------------------------------------

def revenue_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total Revenue by Region.

    'Unknown Region' rows are excluded so charts show only the seven
    known cities. Missing Revenue values are also excluded.

    Returns
    -------
    pd.DataFrame with columns: Region, Revenue
        sorted descending by Revenue.
    """
    eligible = df[~df["Revenue_Missing"] & ~df["Region_Unknown"]].copy()

    reg = (
        eligible
        .groupby("Region", as_index=False)["Revenue"]
        .sum()
        .sort_values("Revenue", ascending=False)
        .reset_index(drop=True)
    )
    return reg


# ---------------------------------------------------------------------------
# Revenue by Product
# ---------------------------------------------------------------------------

def revenue_by_product(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Aggregate total Revenue by Product, returning the top N products.

    'Unknown Product' rows are excluded. Missing Revenue values are excluded.

    Parameters
    ----------
    top_n : int
        Number of top products to return (default 10).

    Returns
    -------
    pd.DataFrame with columns: Product, Revenue
        sorted descending by Revenue (highest first).
    """
    eligible = df[~df["Revenue_Missing"] & ~df["Product_Unknown"]].copy()

    prod = (
        eligible
        .groupby("Product", as_index=False)["Revenue"]
        .sum()
        .sort_values("Revenue", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return prod


# ---------------------------------------------------------------------------
# Profit by Product
# ---------------------------------------------------------------------------

def profit_by_product(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Aggregate total Profit by Product, returning the top N products
    by absolute profit contribution.

    Negative Profit values (losses) are preserved. 'Unknown Product'
    rows are excluded.

    Parameters
    ----------
    top_n : int
        Number of top products to return (default 10).

    Returns
    -------
    pd.DataFrame with columns: Product, Profit
        sorted descending by Profit.
    """
    eligible = df[~df["Product_Unknown"]].copy()

    prod = (
        eligible
        .groupby("Product", as_index=False)["Profit"]
        .sum()
        .sort_values("Profit", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return prod


# ---------------------------------------------------------------------------
# Customer Type distribution
# ---------------------------------------------------------------------------

def customer_type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count of orders by Customer_Type.

    All three types are included: New Customer, Repeat Customer,
    Unknown Customer (so the chart honestly reflects data quality).

    Returns
    -------
    pd.DataFrame with columns: Customer_Type, Count
        sorted descending by Count.
    """
    dist = (
        df
        .groupby("Customer_Type", as_index=False)
        .size()
        .rename(columns={"size": "Count"})
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )
    return dist


# ---------------------------------------------------------------------------
# Profit / Loss / Break-even distribution
# ---------------------------------------------------------------------------

def profit_loss_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count of orders by Profit_Status (Profit / Loss / Break-even).

    Returns
    -------
    pd.DataFrame with columns: Profit_Status, Count
    """
    dist = (
        df
        .groupby("Profit_Status", as_index=False)
        .size()
        .rename(columns={"size": "Count"})
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )
    return dist


# ---------------------------------------------------------------------------
# Filtered DataFrame helper (used by Streamlit sidebar filters)
# ---------------------------------------------------------------------------

def apply_filters(
    df: pd.DataFrame,
    regions: list | None = None,
    categories: list | None = None,
    customer_types: list | None = None,
) -> pd.DataFrame:
    """
    Apply sidebar filter selections to df and return a filtered copy.

    None or empty list means "no filter applied" (all values kept).

    Parameters
    ----------
    regions        : list of region strings to keep
    categories     : list of category strings to keep
    customer_types : list of customer type strings to keep

    Returns
    -------
    pd.DataFrame (filtered copy)
    """
    filtered = df.copy()

    if regions:
        filtered = filtered[filtered["Region"].isin(regions)]
    if categories:
        filtered = filtered[filtered["Category"].isin(categories)]
    if customer_types:
        filtered = filtered[filtered["Customer_Type"].isin(customer_types)]

    return filtered


# ---------------------------------------------------------------------------
# Quick self-test (run directly: python analytics.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_cleaning import load_and_clean

    df_clean, quality_report = load_and_clean()

    print("=" * 60)
    print("ANALYTICS - SELF-TEST")
    print("=" * 60)

    # --- KPIs ---
    kpis = get_kpis(df_clean)
    print("\n[KPIs]")
    print(f"  Total Revenue    : Rs.{kpis['total_revenue']:,.0f}")
    print(f"  Total Profit     : Rs.{kpis['total_profit']:,.0f}")
    print(f"  Total Orders     : {kpis['total_orders']:,}")
    print(f"  Total Quantity   : {kpis['total_quantity']:,}")
    print(f"  Unique Customers : {kpis['unique_customers']:,}")

    # --- Monthly Revenue ---
    monthly = revenue_by_month(df_clean)
    print(f"\n[Monthly Revenue - {len(monthly)} months with valid-date data]")
    print(monthly.to_string(index=False))

    # --- Revenue by Category ---
    cat = revenue_by_category(df_clean)
    print(f"\n[Revenue by Category - {len(cat)} categories]")
    print(cat.to_string(index=False))

    # --- Revenue by Region ---
    reg = revenue_by_region(df_clean)
    print(f"\n[Revenue by Region - {len(reg)} regions]")
    print(reg.to_string(index=False))

    # --- Top 10 Products by Revenue ---
    rev_prod = revenue_by_product(df_clean, top_n=10)
    print(f"\n[Top 10 Products by Revenue]")
    print(rev_prod.to_string(index=False))

    # --- Top 10 Products by Profit ---
    prof_prod = profit_by_product(df_clean, top_n=10)
    print(f"\n[Top 10 Products by Profit]")
    print(prof_prod.to_string(index=False))

    # --- Customer Type Distribution ---
    ct = customer_type_distribution(df_clean)
    print(f"\n[Customer Type Distribution]")
    print(ct.to_string(index=False))

    # --- Profit/Loss Distribution ---
    pl = profit_loss_distribution(df_clean)
    print(f"\n[Profit/Loss/Break-even Distribution]")
    print(pl.to_string(index=False))

    # --- Filter test ---
    filtered = apply_filters(df_clean, regions=["Delhi", "Mumbai"], categories=["Electronics"])
    print(f"\n[Filter test - Delhi+Mumbai, Electronics]: {len(filtered)} rows")

    print("\n" + "=" * 60)
    print("analytics.py: PASS")
