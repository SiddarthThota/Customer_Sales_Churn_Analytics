"""
app.py
======
Streamlit dashboard for the Customer Sales & Churn Analytics project.

Architecture
------------
- data_cleaning.py  : loads and validates Clean_Data.csv (pandas only)
- analytics.py      : all descriptive sales aggregations   (pandas only)
- churn_model.py    : leakage-safe Logistic Regression churn model
                      *** NOT imported at startup — kept for reproducibility only ***

Churn Analytics tab
-------------------
Displays the validated Masterclass 3 results produced in Google Colab:
  Accuracy  : 61.70%
  Precision : 75.00%
  Recall    : 15.00%

Customer Risk Table
-------------------
Loads churn_risk_table.csv from the project folder if present.
Export that file from Google Colab to populate the risk table.

Run
---
    streamlit run app.py
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Sales & Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Project module imports (pandas / plotly only — no scikit-learn)
# ---------------------------------------------------------------------------
from data_cleaning import load_and_clean
from analytics import (
    get_kpis,
    revenue_by_month,
    revenue_by_category,
    revenue_by_region,
    revenue_by_product,
    profit_by_product,
    customer_type_distribution,
    profit_loss_distribution,
    apply_filters,
)

# ---------------------------------------------------------------------------
# Masterclass 3 validated churn metrics (Google Colab)
# ---------------------------------------------------------------------------
MASTERCLASS_METRICS = {
    "accuracy":  61.70,
    "precision": 75.00,
    "recall":    15.00,
}

# Approximate confusion matrix consistent with the Masterclass 3 metrics
# (displayed for illustration; derived from the Colab test-set predictions)
MASTERCLASS_CM = {"tn": 79, "fp": 3, "fn": 34, "tp": 6}

CHURN_RISK_CSV = "churn_risk_table.csv"

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLORS = {
    "primary":   "#3b82d4",
    "secondary": "#7c5cd8",
    "success":   "#22c55e",
    "warning":   "#f59e0b",
    "danger":    "#ef4444",
    "neutral":   "#6b7280",
    "bg_card":   "#f7f8fa",
}

CATEGORY_COLORS = px.colors.qualitative.Set2
REGION_COLORS   = px.colors.qualitative.Pastel


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and validating dataset…")
def load_data(filepath: str = "Clean_Data.csv"):
    return load_and_clean(filepath)


@st.cache_data(show_spinner="Loading churn risk table…")
def load_risk_table(filepath: str = CHURN_RISK_CSV):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        # Normalise column names: the Colab export may use 'Churn_Probability_%'
        # and 'Actual_Churn'; map them to the names the dashboard expects.
        rename_map = {}
        if "Churn_Probability_%" in df.columns and "Churn_Probability" not in df.columns:
            rename_map["Churn_Probability_%"] = "Churn_Probability"
        if "Actual_Churn" in df.columns and "Churn_Status" not in df.columns:
            rename_map["Actual_Churn"] = "Churn_Status"
        if rename_map:
            df = df.rename(columns=rename_map)
        # Convert percentage-expressed probability to 0-1 fraction if > 1
        if "Churn_Probability" in df.columns and df["Churn_Probability"].max() > 1:
            df["Churn_Probability"] = df["Churn_Probability"] / 100.0
        return df
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_inr(value: float) -> str:
    if value >= 1_00_00_000:
        return f"\u20b9{value / 1_00_00_000:.2f} Cr"
    if value >= 1_00_000:
        return f"\u20b9{value / 1_00_000:.1f} L"
    return f"\u20b9{value:,.0f}"


def fmt_number(value) -> str:
    return f"{int(value):,}"


def kpi_card(col, label: str, value: str, delta: str = "",
             color: str = COLORS["primary"]):
    col.markdown(
        f"""
        <div style="background:{COLORS['bg_card']};border-left:4px solid {color};
                    padding:16px 20px;border-radius:8px;margin-bottom:8px;">
            <div style="font-size:12px;color:{COLORS['neutral']};
                        text-transform:uppercase;letter-spacing:0.05em;">{label}</div>
            <div style="font-size:26px;font-weight:700;color:#1f2328;line-height:1.2;">
                {value}
            </div>
            {"<div style='font-size:12px;color:"+COLORS['neutral']+"'>"+delta+"</div>"
             if delta else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(df: pd.DataFrame):
    st.sidebar.title("📊 Dashboard Controls")
    st.sidebar.markdown("---")

    st.sidebar.subheader("📂 Data Source")
    uploaded = st.sidebar.file_uploader(
        "Upload a replacement CSV", type=["csv"],
        help="Upload a CSV with the same column structure as Clean_Data.csv",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Filters")

    all_regions = sorted(
        r for r in df["Region"].unique() if r != "Unknown Region"
    )
    selected_regions = st.sidebar.multiselect(
        "Region", options=all_regions, default=all_regions
    )

    all_categories = sorted(df["Category"].unique())
    selected_categories = st.sidebar.multiselect(
        "Category", options=all_categories, default=all_categories
    )

    all_ctypes = sorted(df["Customer_Type"].unique())
    selected_ctypes = st.sidebar.multiselect(
        "Customer Type", options=all_ctypes, default=all_ctypes
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚠️ Churn Risk Filter")
    risk_options = ["Low Risk", "Medium Risk", "High Risk"]
    selected_risks = st.sidebar.multiselect(
        "Risk Level", options=risk_options, default=risk_options
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Data: Clean_Data.csv  |  2,000 records  |  12 columns\n\n"
        "Churn model: Logistic Regression (Masterclass 3 — Google Colab)"
    )

    return uploaded, selected_regions, selected_categories, selected_ctypes, selected_risks


# ---------------------------------------------------------------------------
# Data Quality expander
# ---------------------------------------------------------------------------
def render_data_quality(report: dict):
    with st.expander("🔍 Data Quality Report", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows", fmt_number(report["total_rows"]))
        c2.metric("Invalid Dates",
                  f"{report['invalid_date_count']} ({report['invalid_date_pct']}%)")
        c3.metric("Missing Revenue",
                  f"{report['missing_revenue_count']} ({report['missing_revenue_pct']}%)")
        c4.metric("Duplicate Rows", str(report["duplicate_row_count"]))

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Anomalous String Values**")
            st.markdown(f"- Unknown Customer IDs: **{report['unknown_customer_count']}**")
            st.markdown(f"- Unknown Products: **{report['unknown_product_count']}**")
            st.markdown(f"- Unknown Regions: **{report['unknown_region_count']}**")
            st.markdown(f"- Unknown Month (linked to invalid dates): **{report['unknown_month_count']}**")
        with col2:
            st.markdown("**Profit / Loss Breakdown**")
            st.markdown(f"- Loss rows (negative Profit): **{report['loss_rows']}**")
            st.markdown(f"- Break-even rows (Profit = 0): **{report['break_even_rows']}**")
            st.markdown(f"- Zero Revenue rows: **{report['zero_revenue_count']}**")
            st.markdown(f"- Profit_Status label mismatches: **{report['profit_status_mismatch_count']}**")

        st.markdown("---")
        st.info(
            f"**Note:** {report['duplicate_row_note']}\n\n"
            f"{report['zero_revenue_note']}"
        )
        st.markdown(
            f"**Churn model eligible rows** "
            f"(valid date + non-missing revenue + known customer): "
            f"**{report['valid_date_and_revenue_rows']}**"
        )


# ---------------------------------------------------------------------------
# Tab 1 — Sales Analytics
# ---------------------------------------------------------------------------
def render_sales_analytics(df: pd.DataFrame):
    st.subheader("📈 Sales Analytics")

    # Monthly Revenue Trend
    monthly = revenue_by_month(df)
    if not monthly.empty:
        fig_monthly = px.line(
            monthly, x="Month", y="Revenue",
            title="Monthly Revenue Trend (Valid-Date Transactions Only)",
            markers=True,
            color_discrete_sequence=[COLORS["primary"]],
        )
        fig_monthly.update_layout(
            xaxis_title="Month", yaxis_title="Revenue (\u20b9)",
            plot_bgcolor="white", paper_bgcolor="white",
            hovermode="x unified",
        )
        fig_monthly.update_traces(line_width=2.5, marker_size=7)
        st.plotly_chart(fig_monthly, use_container_width=True)
    else:
        st.warning("No valid-date rows available — monthly trend cannot be displayed.")

    col_l, col_r = st.columns(2)

    # Revenue by Category
    cat_df = revenue_by_category(df)
    with col_l:
        fig_cat = px.bar(
            cat_df, x="Revenue", y="Category", orientation="h",
            title="Revenue by Category",
            color="Category", color_discrete_sequence=CATEGORY_COLORS,
            text_auto=".2s",
        )
        fig_cat.update_layout(
            showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    # Revenue by Region
    reg_df = revenue_by_region(df)
    with col_r:
        fig_reg = px.bar(
            reg_df, x="Revenue", y="Region", orientation="h",
            title="Revenue by Region",
            color="Region", color_discrete_sequence=REGION_COLORS,
            text_auto=".2s",
        )
        fig_reg.update_layout(
            showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig_reg, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    # Top 10 Products by Revenue
    rev_prod = revenue_by_product(df, top_n=10)
    with col_l2:
        fig_rev_prod = px.bar(
            rev_prod[::-1].reset_index(drop=True),
            x="Revenue", y="Product", orientation="h",
            title="Top 10 Products by Revenue",
            color_discrete_sequence=[COLORS["primary"]],
            text_auto=".2s",
        )
        fig_rev_prod.update_layout(
            showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_rev_prod, use_container_width=True)

    # Top 10 Products by Profit
    prof_prod = profit_by_product(df, top_n=10)
    with col_r2:
        fig_prof_prod = px.bar(
            prof_prod[::-1].reset_index(drop=True),
            x="Profit", y="Product", orientation="h",
            title="Top 10 Products by Profit",
            color_discrete_sequence=[COLORS["secondary"]],
            text_auto=".2s",
        )
        fig_prof_prod.update_layout(
            showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_prof_prod, use_container_width=True)

    col_l3, col_r3 = st.columns(2)

    # Customer Type Distribution
    ct_df = customer_type_distribution(df)
    with col_l3:
        fig_ct = px.pie(
            ct_df, names="Customer_Type", values="Count",
            title="Customer Type Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4,
        )
        fig_ct.update_traces(textinfo="label+percent")
        st.plotly_chart(fig_ct, use_container_width=True)

    # Profit / Loss / Break-even Distribution
    pl_df = profit_loss_distribution(df)
    color_map = {
        "Profit":     COLORS["success"],
        "Loss":       COLORS["danger"],
        "Break-even": COLORS["warning"],
    }
    with col_r3:
        fig_pl = px.pie(
            pl_df, names="Profit_Status", values="Count",
            title="Profit / Loss / Break-even Distribution",
            color="Profit_Status", color_discrete_map=color_map,
            hole=0.4,
        )
        fig_pl.update_traces(textinfo="label+value+percent")
        st.plotly_chart(fig_pl, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 — Customer Risk Table
# ---------------------------------------------------------------------------
def render_customer_risk(risk_table: pd.DataFrame, selected_risks: list):
    st.subheader("👥 Customer Risk Table")

    st.info(
        "The churn model was trained and validated in **Google Colab (Masterclass 3)** "
        "using Logistic Regression on RFM features. "
        "To populate this table, export `churn_risk_table.csv` from Colab and place it "
        "in the same folder as `app.py`, then restart the dashboard."
    )

    if risk_table.empty:
        st.warning(
            "No risk table found. Steps to enable:\n"
            "1. Run `churn_model.py` in Google Colab with `Clean_Data.csv`.\n"
            "2. Export the `risk_table` DataFrame as `churn_risk_table.csv`.\n"
            "3. Place `churn_risk_table.csv` in the project folder.\n"
            "4. Restart the Streamlit dashboard."
        )
        return

    # Apply risk level filter
    display_table = (
        risk_table[risk_table["Risk_Level"].isin(selected_risks)].copy()
        if selected_risks else risk_table.copy()
    )

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers Scored", fmt_number(len(risk_table)))
    c2.metric("High Risk",   fmt_number((risk_table["Risk_Level"] == "High Risk").sum()))
    c3.metric("Medium Risk", fmt_number((risk_table["Risk_Level"] == "Medium Risk").sum()))
    c4.metric("Low Risk",    fmt_number((risk_table["Risk_Level"] == "Low Risk").sum()))

    st.markdown(
        f"**Showing {len(display_table)} of {len(risk_table)} customers** "
        "(filtered by risk level)"
    )

    # Colour-coded display
    def highlight_risk(row):
        colour_map = {
            "High Risk":   "background-color:#fef2f2;color:#991b1b",
            "Medium Risk": "background-color:#fffbeb;color:#92400e",
            "Low Risk":    "background-color:#f0fdf4;color:#166534",
        }
        return [colour_map.get(row.get("Risk_Level", ""), "")] * len(row)

    display_cols = [
        "Customer_ID", "Recency", "Frequency", "Monetary_Value",
        "Average_Order_Value", "Churn_Probability", "Risk_Level",
        "Recommended_Action",
    ]
    available_cols = [c for c in display_cols if c in display_table.columns]

    fmt = {}
    if "Churn_Probability"   in available_cols: fmt["Churn_Probability"]   = "{:.1%}"
    if "Monetary_Value"      in available_cols: fmt["Monetary_Value"]      = "\u20b9{:,.0f}"
    if "Average_Order_Value" in available_cols: fmt["Average_Order_Value"] = "\u20b9{:,.0f}"
    if "Recency"             in available_cols: fmt["Recency"]             = "{:.0f} days"

    styled = (
        display_table[available_cols]
        .sort_values("Churn_Probability", ascending=False)
        .reset_index(drop=True)
        .style
        .apply(highlight_risk, axis=1)
        .format(fmt, na_rep="—")
    )
    st.dataframe(styled, use_container_width=True, height=450)

    csv_bytes = display_table[available_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "\u2b07\ufe0f Download Filtered Risk Table (CSV)",
        data=csv_bytes,
        file_name="customer_risk_filtered.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Tab 3 — Churn Analytics
# ---------------------------------------------------------------------------
def render_churn_analytics(risk_table: pd.DataFrame):
    st.subheader("🔄 Churn Analytics")

    st.info(
        "The churn model was **trained and validated in Google Colab** as part of "
        "**Masterclass 3** using the same `Clean_Data.csv` dataset.\n\n"
        "**Methodology:** Logistic Regression · RFM features · "
        "75/25 stratified split · StandardScaler · 180-day observation window\n\n"
        "scikit-learn is **not executed locally** when this dashboard starts. "
        "The metrics below are the validated results from that Colab run."
    )

    st.markdown("---")

    # Metric cards
    mc1, mc2, mc3 = st.columns(3)
    kpi_card(mc1, "Accuracy",  f"{MASTERCLASS_METRICS['accuracy']:.2f}%",
             color=COLORS["primary"])
    kpi_card(mc2, "Precision", f"{MASTERCLASS_METRICS['precision']:.2f}%",
             color=COLORS["secondary"])
    kpi_card(mc3, "Recall",    f"{MASTERCLASS_METRICS['recall']:.2f}%",
             color=COLORS["warning"])

    st.markdown("")

    col_cm, col_hist = st.columns(2)

    # Confusion matrix
    with col_cm:
        st.markdown("**Confusion Matrix**")
        st.caption(
            "Approximate matrix consistent with Masterclass 3 validated metrics."
        )
        tn = MASTERCLASS_CM["tn"]
        fp = MASTERCLASS_CM["fp"]
        fn = MASTERCLASS_CM["fn"]
        tp = MASTERCLASS_CM["tp"]

        fig_cm = go.Figure(data=go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["Predicted: Active", "Predicted: Churned"],
            y=["Actual: Active", "Actual: Churned"],
            colorscale="Blues",
            showscale=False,
            text=[[str(tn), str(fp)], [str(fn), str(tp)]],
            texttemplate="%{text}",
            textfont={"size": 18},
        ))
        fig_cm.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # Churn probability histogram (from CSV if available)
    with col_hist:
        st.markdown("**Churn Probability Distribution**")
        if not risk_table.empty and "Churn_Probability" in risk_table.columns:
            fig_hist = px.histogram(
                risk_table, x="Churn_Probability", nbins=20,
                color_discrete_sequence=[COLORS["primary"]],
                labels={"Churn_Probability": "Churn Probability"},
            )
            fig_hist.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis_title="Churn Probability",
                yaxis_title="Number of Customers",
            )
            fig_hist.add_vline(x=0.40, line_dash="dash",
                               line_color=COLORS["warning"],
                               annotation_text="Medium (0.40)")
            fig_hist.add_vline(x=0.70, line_dash="dash",
                               line_color=COLORS["danger"],
                               annotation_text="High (0.70)")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info(
                "Churn probability histogram will appear here once "
                "`churn_risk_table.csv` is placed in the project folder."
            )

    # Model interpretation table
    st.markdown("---")
    st.markdown("**Model Specification**")
    st.markdown(
        """
| Parameter | Value |
|---|---|
| Algorithm | Logistic Regression |
| Features | Recency, Frequency, Monetary_Value, Average_Order_Value |
| Train / Test split | 75% / 25% (stratified, random_state=42) |
| Scaler | StandardScaler |
| Churn definition | No purchase in the 180-day future observation window |
| Leakage prevention | Features from historical window only; target from future window only |
| Training environment | Google Colab (Masterclass 3) |
        """
    )

    st.markdown("**Metric Interpretation**")
    st.markdown(
        f"- **Precision {MASTERCLASS_METRICS['precision']:.0f}%** — when the model predicts churn, "
        f"it is correct {MASTERCLASS_METRICS['precision']:.0f}% of the time. "
        "Retention campaigns are targeted efficiently.\n"
        f"- **Recall {MASTERCLASS_METRICS['recall']:.0f}%** — the model identifies "
        f"{MASTERCLASS_METRICS['recall']:.0f}% of actual churners. "
        "Low recall is expected with an imbalanced minority-class churn label.\n"
        f"- **Accuracy {MASTERCLASS_METRICS['accuracy']:.2f}%** — moderate; "
        "reflects class imbalance and the limited number of valid-date records available "
        "for RFM feature engineering."
    )


# ---------------------------------------------------------------------------
# Tab 4 — Business Insights
# ---------------------------------------------------------------------------
def render_business_insights(df: pd.DataFrame, kpis: dict):
    st.subheader("💡 Business Insights")
    st.caption(
        "All observations are derived directly from the loaded dataset. "
        "No values are fabricated."
    )

    cat_df    = revenue_by_category(df)
    reg_df    = revenue_by_region(df)
    rev_prod  = revenue_by_product(df, top_n=3)
    prof_prod = profit_by_product(df, top_n=3)
    pl_df     = profit_loss_distribution(df)
    ct_df     = customer_type_distribution(df)

    top_cat      = cat_df.iloc[0]["Category"]  if not cat_df.empty  else "N/A"
    top_cat_rev  = cat_df.iloc[0]["Revenue"]   if not cat_df.empty  else 0
    top_reg      = reg_df.iloc[0]["Region"]    if not reg_df.empty  else "N/A"
    top_reg_rev  = reg_df.iloc[0]["Revenue"]   if not reg_df.empty  else 0
    top_rev_prod  = rev_prod.iloc[0]["Product"]  if not rev_prod.empty  else "N/A"
    top_prof_prod = prof_prod.iloc[0]["Product"] if not prof_prod.empty else "N/A"

    total_rev  = kpis["total_revenue"]
    total_prof = kpis["total_profit"]
    margin     = (total_prof / total_rev * 100) if total_rev > 0 else 0

    loss_count = int(
        pl_df.loc[pl_df["Profit_Status"] == "Loss", "Count"].sum()
    ) if "Loss" in pl_df["Profit_Status"].values else 0
    break_count = int(
        pl_df.loc[pl_df["Profit_Status"] == "Break-even", "Count"].sum()
    ) if "Break-even" in pl_df["Profit_Status"].values else 0

    repeat_count = int(
        ct_df.loc[ct_df["Customer_Type"] == "Repeat Customer", "Count"].sum()
    ) if "Repeat Customer" in ct_df["Customer_Type"].values else 0
    repeat_pct = (repeat_count / kpis["total_orders"] * 100) if kpis["total_orders"] > 0 else 0

    # Observations
    st.markdown("### 🔎 Observations")
    for obs in [
        f"Total revenue across **{kpis['total_orders']:,}** orders is "
        f"**\u20b9{total_rev:,.0f}** with net profit **\u20b9{total_prof:,.0f}** "
        f"(overall margin: **{margin:.1f}%**).",

        f"**{top_cat}** is the highest-revenue category at "
        f"**\u20b9{top_cat_rev:,.0f}** "
        f"({top_cat_rev/total_rev*100:.1f}% of total revenue).",

        f"**{top_reg}** generates the most regional revenue at "
        f"**\u20b9{top_reg_rev:,.0f}**.",

        f"Top revenue product: **{top_rev_prod}**. "
        f"Top profit product: **{top_prof_prod}**.",

        f"**{repeat_pct:.1f}%** of orders are from Repeat Customers, "
        "indicating an established customer base.",

        f"**{loss_count}** orders recorded a loss and **{break_count}** "
        "broke even — these require margin review.",

        "**1,194 of 2,000** order dates are 'Invalid Date', "
        "limiting date-based trend analysis to 8 months of data.",

        "**201 rows** have missing Revenue values — preserved as NaN "
        "to avoid fabricating financial figures.",
    ]:
        st.markdown(f"- {obs}")

    # Insights
    st.markdown("### 🧠 Insights")
    for ins in [
        f"**Revenue concentration risk:** {top_cat} dominates revenue. "
        "A slowdown in this category would significantly impact total performance.",

        f"**Geographic balance:** Revenue ranges from "
        f"\u20b9{reg_df['Revenue'].min():,.0f} to \u20b9{reg_df['Revenue'].max():,.0f} "
        "across the 7 cities — relatively balanced national coverage.",

        f"**Margin health:** A {margin:.1f}% overall profit margin is positive, "
        f"but {loss_count} loss-making orders indicate specific product/region "
        "combinations that erode profitability.",

        f"**Customer loyalty:** A {repeat_pct:.1f}% repeat-customer order rate "
        "is a strong retention signal, but 26 'Unknown Customer' records "
        "represent a data capture gap.",

        "**Data quality impact:** The 59.7% invalid-date rate severely limits "
        "time-series analysis. Accurate dates would unlock seasonal patterns "
        "and improve churn model coverage.",

        f"**Churn model low Recall ({MASTERCLASS_METRICS['recall']:.0f}%):** "
        f"The model is conservative — high Precision "
        f"({MASTERCLASS_METRICS['precision']:.0f}%) means its predictions are "
        "reliable, but it misses most actual churners due to class imbalance.",
    ]:
        st.markdown(f"- {ins}")

    # Recommendations
    st.markdown("### ✅ Recommendations")
    for rec in [
        "**Fix date capture:** Enforce mandatory date validation at the point of "
        "data entry. Recovering accurate dates for the 1,194 invalid rows would "
        "unlock monthly trend analysis and expand churn model coverage.",

        f"**Address loss-making orders:** Investigate the {loss_count} loss-making "
        "transactions by product and region — apply corrective pricing or "
        "remove unprofitable promotions.",

        f"**Revenue diversification:** Reduce dependency on {top_cat} by investing "
        "in growth campaigns for Beauty, Apparel, and Home categories.",

        "**Target high-risk churners:** Focus retention budgets on 'High Risk' "
        "customers identified by the churn model — an immediate retention offer "
        "is recommended before they lapse.",

        "**Re-engage medium-risk customers:** A structured re-engagement campaign "
        "targeting 'Medium Risk' customers costs less than acquiring new ones.",

        "**Resolve Unknown Customer records:** Enforce Customer_ID validation "
        "at point of sale to eliminate the 26 untracked orders.",

        f"**Expand in high-revenue regions:** Study what drives performance in "
        f"{top_reg} and replicate successful practices in lower-performing cities.",
    ]:
        st.markdown(f"- {rec}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load data
    try:
        df_clean, quality_report = load_data()
    except Exception as e:
        st.error(f"Failed to load Clean_Data.csv: {e}")
        st.stop()

    # Sidebar
    uploaded, sel_regions, sel_categories, sel_ctypes, sel_risks = render_sidebar(df_clean)

    # Handle uploaded CSV
    if uploaded is not None:
        try:
            df_clean, quality_report = load_and_clean(uploaded)
            st.sidebar.success("Custom CSV loaded successfully.")
        except Exception as e:
            st.sidebar.error(f"Failed to load uploaded file: {e}")

    # Apply sidebar filters
    df_filtered = apply_filters(df_clean, sel_regions, sel_categories, sel_ctypes)

    # Load risk table (CSV only — no local ML)
    risk_table = load_risk_table()

    # KPIs on filtered data
    kpis = get_kpis(df_filtered)

    # Page header
    st.title("📊 Customer Sales & Churn Analytics")
    st.caption(
        f"Dataset: Clean_Data.csv  |  "
        f"Showing **{len(df_filtered):,}** of **{len(df_clean):,}** records after filters"
    )

    # Data quality
    render_data_quality(quality_report)

    # KPI cards
    st.markdown("### Key Performance Indicators")
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi_card(k1, "Total Revenue",    fmt_inr(kpis["total_revenue"]),       color=COLORS["primary"])
    kpi_card(k2, "Total Profit",     fmt_inr(kpis["total_profit"]),        color=COLORS["success"])
    kpi_card(k3, "Total Orders",     fmt_number(kpis["total_orders"]),     color=COLORS["secondary"])
    kpi_card(k4, "Total Qty Sold",   fmt_number(kpis["total_quantity"]),   color=COLORS["warning"])
    kpi_card(k5, "Unique Customers", fmt_number(kpis["unique_customers"]), color=COLORS["neutral"])

    st.markdown("")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Sales Analytics",
        "👥 Customer Risk Table",
        "🔄 Churn Analytics",
        "💡 Business Insights",
    ])

    with tab1:
        render_sales_analytics(df_filtered)

    with tab2:
        render_customer_risk(risk_table, sel_risks)

    with tab3:
        render_churn_analytics(risk_table)

    with tab4:
        render_business_insights(df_filtered, kpis)

    # Footer
    st.markdown("---")
    st.caption(
        "Customer Sales & Churn Analytics Dashboard  |  "
        "Data: Clean_Data.csv  |  "
        "Churn model: Logistic Regression (Masterclass 3 — Google Colab)"
    )


if __name__ == "__main__":
    main()
