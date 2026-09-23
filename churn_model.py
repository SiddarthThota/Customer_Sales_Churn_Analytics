"""
churn_model.py
==============
Leakage-safe customer churn model for the Customer Sales & Churn Analytics project.

Methodology (consistent with Masterclass 3):
---------------------------------------------
1. Use only transactions with BOTH a valid Order_Date AND non-missing Revenue.
   Exclude 'Unknown Customer' rows — they cannot represent an identifiable customer.

2. Determine the observation window:
       latest_date        = maximum valid Order_Date in the eligible set
       historical_cutoff  = latest_date - 180 days
   Historical window : Order_Date <= historical_cutoff   (features built here)
   Future window     : Order_Date >  historical_cutoff   (target defined here)

3. Build customer-level RFM features from the HISTORICAL window only:
       Recency           = days from historical_cutoff to customer's last purchase
       Frequency         = number of orders in the historical window
       Monetary_Value    = total Revenue in the historical window
       Average_Order_Value = Monetary_Value / Frequency

4. Define Churn_Status from the FUTURE window only:
       Customer has a purchase in future window  ->  Churn_Status = 0 (active)
       Customer has NO purchase in future window ->  Churn_Status = 1 (churned)

5. Train / evaluate model:
       StandardScaler  ->  LogisticRegression(max_iter=1000)
       train_test_split(test_size=0.25, random_state=42, stratify=y)
       Metrics: Accuracy, Precision, Recall, Confusion Matrix

6. Score ALL customers in the historical window (not just the test split):
       Churn_Probability -> Risk_Level -> Recommended_Action
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# ---------------------------------------------------------------------------
# Risk-level thresholds
# ---------------------------------------------------------------------------
RISK_LOW_MAX    = 0.40   # probability < 0.40  -> Low Risk
RISK_MEDIUM_MAX = 0.70   # 0.40 <= prob < 0.70 -> Medium Risk
                          # prob >= 0.70         -> High Risk

RECOMMENDED_ACTIONS = {
    "Low Risk":    "Monitor regularly",
    "Medium Risk": "Send re-engagement campaign",
    "High Risk":   "Immediate retention offer required",
}

FEATURES = ["Recency", "Frequency", "Monetary_Value", "Average_Order_Value"]


# ---------------------------------------------------------------------------
# Step 1 – Filter eligible rows
# ---------------------------------------------------------------------------

def _get_eligible(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows that are usable for churn modelling:
      - valid Order_Date
      - non-missing Revenue
      - known Customer_ID (not 'Unknown Customer')
    """
    return df[
        df["Order_Date_Valid"]
        & ~df["Revenue_Missing"]
        & ~df["Customer_Unknown"]
    ].copy()


# ---------------------------------------------------------------------------
# Step 2 – Determine cutoff dates
# ---------------------------------------------------------------------------

def _get_cutoffs(eligible: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Returns (historical_cutoff, latest_date).
    historical_cutoff = latest_date - 180 days
    """
    latest_date = eligible["Order_Date"].max()
    historical_cutoff = latest_date - pd.Timedelta(days=180)
    return historical_cutoff, latest_date


# ---------------------------------------------------------------------------
# Step 3 – Build customer-level RFM features (historical window only)
# ---------------------------------------------------------------------------

def _build_rfm(
    eligible: pd.DataFrame,
    historical_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build one row per Customer_ID using only transactions on or before
    historical_cutoff.

    Columns returned:
        Customer_ID, Recency, Frequency, Monetary_Value, Average_Order_Value
    """
    hist = eligible[eligible["Order_Date"] <= historical_cutoff].copy()

    if hist.empty:
        return pd.DataFrame(
            columns=["Customer_ID"] + FEATURES
        )

    rfm = hist.groupby("Customer_ID").agg(
        last_purchase=("Order_Date", "max"),
        Frequency=("Order_ID", "count"),
        Monetary_Value=("Revenue", "sum"),
    ).reset_index()

    # Recency: days between the customer's last purchase and the cutoff
    rfm["Recency"] = (historical_cutoff - rfm["last_purchase"]).dt.days

    rfm["Average_Order_Value"] = rfm["Monetary_Value"] / rfm["Frequency"]

    return rfm[["Customer_ID", "Recency", "Frequency",
                "Monetary_Value", "Average_Order_Value"]]


# ---------------------------------------------------------------------------
# Step 4 – Create churn target (future window only)
# ---------------------------------------------------------------------------

def _build_churn_target(
    eligible: pd.DataFrame,
    historical_cutoff: pd.Timestamp,
    rfm: pd.DataFrame,
) -> pd.DataFrame:
    """
    For every customer in rfm, check whether they appear in the future window
    (Order_Date > historical_cutoff).

    Churn_Status:
        0 = customer made a purchase in the future window (active)
        1 = customer made NO purchase in the future window (churned)
    """
    future = eligible[eligible["Order_Date"] > historical_cutoff]
    active_customers = set(future["Customer_ID"].unique())

    rfm = rfm.copy()
    rfm["Churn_Status"] = rfm["Customer_ID"].apply(
        lambda cid: 0 if cid in active_customers else 1
    )
    return rfm


# ---------------------------------------------------------------------------
# Step 5 – Train Logistic Regression model
# ---------------------------------------------------------------------------

def _train_model(
    rfm_with_target: pd.DataFrame,
) -> tuple:
    """
    Train a Logistic Regression model on the RFM features.

    Returns
    -------
    scaler      : fitted StandardScaler
    model       : fitted LogisticRegression
    metrics     : dict with accuracy, precision, recall, confusion_matrix
    X_test_df   : test-set feature DataFrame (for the confusion matrix display)
    y_test      : test-set true labels
    y_pred      : test-set predictions
    """
    X = rfm_with_target[FEATURES].values
    y = rfm_with_target["Churn_Status"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)

    metrics = {
        "accuracy":         round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision":        round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "recall":           round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    return scaler, model, metrics, y_test, y_pred


# ---------------------------------------------------------------------------
# Step 6 – Score all historical customers → risk table
# ---------------------------------------------------------------------------

def _build_risk_table(
    rfm_with_target: pd.DataFrame,
    scaler: StandardScaler,
    model: LogisticRegression,
) -> pd.DataFrame:
    """
    Apply the trained model to ALL customers in the historical window to
    produce churn probabilities, risk levels, and recommended actions.

    Returns a customer-level DataFrame with columns:
        Customer_ID, Recency, Frequency, Monetary_Value,
        Average_Order_Value, Churn_Status, Churn_Probability,
        Risk_Level, Recommended_Action
    """
    X_all = rfm_with_target[FEATURES].values
    X_all_sc = scaler.transform(X_all)

    proba = model.predict_proba(X_all_sc)[:, 1]   # probability of churn (class 1)

    risk_table = rfm_with_target.copy()
    risk_table["Churn_Probability"] = proba.round(4)

    def assign_risk(p):
        if p < RISK_LOW_MAX:
            return "Low Risk"
        elif p < RISK_MEDIUM_MAX:
            return "Medium Risk"
        return "High Risk"

    risk_table["Risk_Level"] = risk_table["Churn_Probability"].apply(assign_risk)
    risk_table["Recommended_Action"] = risk_table["Risk_Level"].map(RECOMMENDED_ACTIONS)

    return risk_table[[
        "Customer_ID", "Recency", "Frequency", "Monetary_Value",
        "Average_Order_Value", "Churn_Status", "Churn_Probability",
        "Risk_Level", "Recommended_Action",
    ]]


# ---------------------------------------------------------------------------
# Public API – single entry point used by app.py
# ---------------------------------------------------------------------------

def run_churn_model(df: pd.DataFrame) -> dict:
    """
    Full churn-model pipeline.  Accepts the cleaned DataFrame from
    data_cleaning.load_and_clean() and returns a results dict.

    Returns
    -------
    dict with keys:
        eligible_row_count   : int  – rows used after filtering
        historical_cutoff    : pd.Timestamp
        latest_date          : pd.Timestamp
        historical_row_count : int  – transactions in historical window
        future_row_count     : int  – transactions in future window
        customer_count       : int  – customers with historical data
        churn_rate           : float (0-1)
        metrics              : dict (accuracy, precision, recall, confusion_matrix)
        y_test               : np.ndarray
        y_pred               : np.ndarray
        risk_table           : pd.DataFrame
        error                : str or None  – populated if pipeline fails
    """
    result = {
        "eligible_row_count": 0,
        "historical_cutoff": None,
        "latest_date": None,
        "historical_row_count": 0,
        "future_row_count": 0,
        "customer_count": 0,
        "churn_rate": None,
        "metrics": None,
        "y_test": None,
        "y_pred": None,
        "risk_table": pd.DataFrame(),
        "error": None,
    }

    try:
        # 1. Filter eligible rows
        eligible = _get_eligible(df)
        result["eligible_row_count"] = len(eligible)

        if eligible.empty:
            result["error"] = "No eligible rows found for churn modelling."
            return result

        # 2. Cutoff dates
        historical_cutoff, latest_date = _get_cutoffs(eligible)
        result["historical_cutoff"] = historical_cutoff
        result["latest_date"] = latest_date

        hist_rows = eligible[eligible["Order_Date"] <= historical_cutoff]
        future_rows = eligible[eligible["Order_Date"] > historical_cutoff]
        result["historical_row_count"] = len(hist_rows)
        result["future_row_count"] = len(future_rows)

        # 3. RFM features
        rfm = _build_rfm(eligible, historical_cutoff)
        result["customer_count"] = len(rfm)

        if len(rfm) < 10:
            result["error"] = (
                f"Only {len(rfm)} customers in the historical window — "
                "too few to train a reliable model."
            )
            return result

        # 4. Churn target
        rfm_with_target = _build_churn_target(eligible, historical_cutoff, rfm)
        churn_rate = rfm_with_target["Churn_Status"].mean()
        result["churn_rate"] = round(float(churn_rate), 4)

        # Guard: stratified split requires both classes to be present
        class_counts = rfm_with_target["Churn_Status"].value_counts()
        if len(class_counts) < 2:
            result["error"] = (
                "Churn target has only one class — cannot train a classifier. "
                "Try adjusting the observation window."
            )
            return result

        # 5. Train model
        scaler, model, metrics, y_test, y_pred = _train_model(rfm_with_target)
        result["metrics"] = metrics
        result["y_test"] = y_test
        result["y_pred"] = y_pred

        # 6. Risk table
        risk_table = _build_risk_table(rfm_with_target, scaler, model)
        result["risk_table"] = risk_table

    except Exception as exc:
        result["error"] = f"Churn model error: {exc}"

    return result


# ---------------------------------------------------------------------------
# Quick self-test (run directly: python churn_model.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_cleaning import load_and_clean

    df_clean, _ = load_and_clean()
    results = run_churn_model(df_clean)

    print("=" * 60)
    print("CHURN MODEL - SELF-TEST")
    print("=" * 60)

    if results["error"]:
        print(f"ERROR: {results['error']}")
    else:
        print(f"\n[Data windows]")
        print(f"  Eligible rows (valid date + revenue + known customer): {results['eligible_row_count']}")
        print(f"  Latest valid date     : {results['latest_date'].date()}")
        print(f"  Historical cutoff     : {results['historical_cutoff'].date()}")
        print(f"  Historical window rows: {results['historical_row_count']}")
        print(f"  Future window rows    : {results['future_row_count']}")
        print(f"  Customers in model    : {results['customer_count']}")
        print(f"  Overall churn rate    : {results['churn_rate'] * 100:.1f}%")

        m = results["metrics"]
        print(f"\n[Model Metrics - Logistic Regression, 75/25 split, stratified]")
        print(f"  Accuracy  : {m['accuracy']:.2f}%")
        print(f"  Precision : {m['precision']:.2f}%")
        print(f"  Recall    : {m['recall']:.2f}%")

        cm = m["confusion_matrix"]
        print(f"\n[Confusion Matrix]")
        print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
        print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

        rt = results["risk_table"]
        risk_counts = rt["Risk_Level"].value_counts()
        print(f"\n[Risk Level Distribution]")
        for level in ["Low Risk", "Medium Risk", "High Risk"]:
            count = risk_counts.get(level, 0)
            print(f"  {level:12s}: {count} customers")

        print(f"\n[Sample Risk Table - first 5 rows]")
        print(rt.head(5).to_string(index=False))

        churn_pct_preview = rt["Churn_Probability"].describe()
        print(f"\n[Churn Probability Summary]")
        print(churn_pct_preview.to_string())

    print("\n" + "=" * 60)
    print("churn_model.py: PASS" if not results["error"] else "churn_model.py: FAIL")
