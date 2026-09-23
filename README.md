# Customer Sales & Churn Analytics

A professional Python data analytics dashboard built with **Streamlit** and **Plotly** that performs descriptive sales analysis and customer churn prediction on a real retail dataset.

---

## Project Objective

This project analyses customer purchasing behaviour from a retail sales dataset to:

1. Produce accurate, data-backed descriptive sales analytics (revenue, profit, product performance, regional breakdowns).
2. Identify customers at risk of churning using a leakage-safe RFM (Recency, Frequency, Monetary) churn model.
3. Generate actionable business recommendations based solely on computed results — no fabricated conclusions.

---

## Dataset Description

**File:** `Clean_Data.csv`  
**Records:** 2,000 order rows | **Columns:** 12

| Column | Type | Description |
|---|---|---|
| `Order_ID` | String | Unique order identifier |
| `Order_Date` | String / Date | Date of the transaction (1,194 rows contain "Invalid Date") |
| `Customer_ID` | String | Customer identifier (26 rows contain "Unknown Customer") |
| `Product` | String | Product name (7 rows contain "Unknown Product") |
| `Category` | String | Product category: Apparel, Beauty, Electronics, Furniture, Home |
| `Region` | String | City of sale: Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai, Pune (16 "Unknown Region") |
| `Quantity` | Integer | Units sold per order |
| `Revenue` | Float | Revenue in Indian Rupees (₹); 201 rows are missing |
| `Profit` | Float | Profit in ₹; negative values represent genuine losses |
| `Customer_Type` | String | New Customer / Repeat Customer / Unknown Customer |
| `Month` | String | Calendar month name; "Unknown" where Order_Date is invalid |
| `Profit_Status` | String | Profit / Loss / Break-even |

---

## Data Cleaning Process

Implemented in [`data_cleaning.py`](data_cleaning.py).

### Steps applied

1. **Load** — `pd.read_csv()` with `dtype=str` and `keep_default_na=False` to prevent premature type coercion. Empty strings then converted to `NaN`.
2. **Standardise text** — strip whitespace from all string columns; normalise casing on `Category`, `Region`, `Customer_Type`, `Profit_Status`.
3. **Detect duplicates** — 23 exact duplicate rows flagged with `Is_Duplicate = True`. **Not removed** — see note below.
4. **Parse Order_Date** — `pd.to_datetime()` with `errors='coerce'`. Rows that fail (e.g. literal "Invalid Date") receive `NaT` and `Order_Date_Valid = False`. The raw string is preserved in `Order_Date_Raw`.
5. **Clean Revenue & Profit** — converted to numeric. Missing Revenue stays `NaN` (never filled). `Revenue_Missing` and `Revenue_Is_Zero` flag columns added.
6. **Flag anomalous strings** — `Customer_Unknown`, `Product_Unknown`, `Region_Unknown` boolean columns added for filtering.
7. **Validate Profit_Status** — cross-checked against the sign of `Profit`; mismatches flagged in `Profit_Status_Mismatch`.
8. **Add Month_Num** — numeric sort key for calendar-order charting.

### Why duplicates are not removed

The 23 duplicate rows were present when the churn model was trained and validated in Masterclass 3 (Google Colab). Removing them now would change the feature distributions and produce different model metrics, breaking methodological consistency. They are documented but retained.

### Why missing Revenue is not filled

Replacing 201 missing Revenue values with zero or a mean would fabricate financial data. These rows are excluded from revenue aggregations but kept in the dataset so that non-revenue metrics (order counts, profit, customer type) remain accurate.

---

## Exploratory Data Analysis (EDA) Methodology

Implemented in [`analytics.py`](analytics.py).

All analysis is performed on the cleaned DataFrame. Revenue calculations exclude `NaN` Revenue rows.

| Analysis | Function | Notes |
|---|---|---|
| KPI Summary | `get_kpis()` | Total Revenue, Profit, Orders, Quantity, Unique Customers |
| Monthly Revenue Trend | `revenue_by_month()` | Valid-date rows only; calendar-sorted |
| Revenue by Category | `revenue_by_category()` | All 5 categories |
| Revenue by Region | `revenue_by_region()` | 7 known cities; Unknown Region excluded from chart |
| Top Products by Revenue | `revenue_by_product(top_n)` | Unknown Product excluded |
| Top Products by Profit | `profit_by_product(top_n)` | Negative values preserved |
| Customer Type Distribution | `customer_type_distribution()` | All 3 types shown |
| Profit/Loss/Break-even | `profit_loss_distribution()` | Reflects actual Profit_Status labels |
| Sidebar filters | `apply_filters()` | Region, Category, Customer Type |

---

## Churn Modelling Methodology

Implemented in [`churn_model.py`](churn_model.py).

### Eligibility filter

Only transactions satisfying **all three** conditions are used:
- `Order_Date_Valid = True` (parseable date)
- `Revenue_Missing = False` (non-NaN Revenue)
- `Customer_Unknown = False` (known Customer_ID)

### Observation windows

```
latest_date       = max(valid Order_Date)
historical_cutoff = latest_date − 180 days

Historical window : Order_Date <= historical_cutoff  ← features built here
Future window     : Order_Date >  historical_cutoff  ← target defined here
```

This strict separation prevents **target leakage**: features are computed only from the past; the churn label is derived only from the future.

### Features (RFM)

| Feature | Definition |
|---|---|
| `Recency` | Days between the customer's last historical purchase and `historical_cutoff` |
| `Frequency` | Number of orders in the historical window |
| `Monetary_Value` | Total Revenue in the historical window |
| `Average_Order_Value` | `Monetary_Value / Frequency` |

All four features are computed **per customer** using only the historical window.

### Churn target

| Label | Condition |
|---|---|
| `Churn_Status = 0` | Customer made at least one purchase in the future window (active) |
| `Churn_Status = 1` | Customer made **no** purchase in the future window (churned) |

### Model pipeline

```python
train_test_split(test_size=0.25, random_state=42, stratify=y)
StandardScaler()
LogisticRegression(max_iter=1000)
```

### Leakage prevention

| Potential leakage source | How it is prevented |
|---|---|
| Using future purchase dates to calculate Recency | Features computed from historical window only |
| Including future transactions in Frequency/Monetary | Filtered to `Order_Date <= historical_cutoff` |
| Training on the same customers used to define the target | Target uses the future window; features use the historical window — no overlap |
| Scaling with test-set statistics | `StandardScaler` is `fit` on the training set only; `transform` applied to test set |

---

## Features Used by the Churn Model

- **Recency** — a proxy for engagement recency; higher values indicate longer inactivity.
- **Frequency** — number of transactions; higher frequency signals loyalty.
- **Monetary_Value** — total spend; high-value customers may be more or less likely to churn depending on satisfaction.
- **Average_Order_Value** — spend per visit; captures purchasing intensity.

---

## Model Evaluation Metrics

Validated in **Google Colab — Masterclass 3** using the same dataset and pipeline:

| Metric | Value |
|---|---|
| Accuracy | 61.70% |
| Precision | 75.00% |
| Recall | 15.00% |
| Algorithm | Logistic Regression |
| Split | 75% train / 25% test (stratified) |

**Interpretation:**
- **High Precision (75%)** — when the model predicts churn, it is correct 75% of the time. Retention campaigns are targeted efficiently.
- **Low Recall (15%)** — the model misses a large proportion of actual churners. This is expected with an imbalanced dataset where churned customers are a minority class.
- **Accuracy (61.70%)** — moderate; reflective of the class imbalance and the limited date coverage (only 806 rows have valid dates for RFM calculation).

---

## Customer Risk Levels

| Risk Level | Churn Probability | Recommended Action |
|---|---|---|
| Low Risk | < 40% | Monitor regularly |
| Medium Risk | 40% – 69% | Send re-engagement campaign |
| High Risk | ≥ 70% | Immediate retention offer required |

---

## Project Structure

```
Customer_Sales_Churn_Analytics/
├── Clean_Data.csv           ← source dataset (never modified)
├── churn_risk_table.csv     ← pre-computed customer risk scores (from Masterclass 3 Colab)
├── Customer_Sales_Churn_Analytics_Submission.ipynb ← complete submission notebook
├── app.py                   ← Streamlit dashboard entry point
├── data_cleaning.py         ← data loading, validation, quality reporting
├── analytics.py             ← descriptive sales aggregation functions
├── churn_model.py           ← RFM feature engineering + Logistic Regression
├── requirements.txt         ← Python dependencies (tested environment)
└── README.md                ← this file
```

---

## Internship Submission Package

For the AICTE | IBM SkillsBuild Data Analytics with AI Internship submission, the project package should contain:

1. `Customer_Sales_Churn_Analytics_Submission.ipynb` — complete project code in Jupyter Notebook format
2. `requirements.txt` — Python dependencies
3. Project report in `.docx` format
4. `README.md` — project documentation
5. GitHub repository containing the final project files

The virtual environment folder (for example `.venv_final`) is a local development environment and should **not** be uploaded as part of the source repository/submission package.

---

## Tested Environment

| Component | Value |
|---|---|
| Python | 3.11.9 (64-bit) |
| Virtual environment | `.venv_final` |
| numpy | 2.4.6 |
| pandas | 2.3.3 |
| plotly | 5.24.1 |
| pyarrow | 19.0.1 |
| streamlit | 1.64.0 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.1 |

> **Important:** Do **not** use numpy 1.26.4 with Python 3.11 on Windows — it causes hangs on import.
> The project requires numpy ≥ 2.0.

---

## How to Install Dependencies

```bash
# Create a fresh virtual environment (Python 3.11 recommended)
py -3.11 -m venv .venv_final
.venv_final\Scripts\activate
pip install -r requirements.txt
```

> **Note:** `scikit-learn` is required only for `churn_model.py` (reproducibility). The Streamlit dashboard does **not** import scikit-learn at startup.

---

## How to Run the Streamlit Application

```bash
# From the project root, with the virtual environment activated:
.venv_final\Scripts\activate
streamlit run app.py
```

Or in a single command from PowerShell (no activation needed):

```powershell
.\.venv_final\Scripts\streamlit.exe run app.py
```

The dashboard will open at `http://localhost:8501` in your browser.

### churn_risk_table.csv

`churn_risk_table.csv` is already present in the project folder and was exported from the validated Masterclass 3 Google Colab run. The Customer Risk Table tab loads it automatically. No local model training is required.

---

## Project Limitations

1. **Date quality:** 59.7% of Order_Date values are "Invalid Date", limiting monthly trend analysis to 8 months and reducing the number of customers eligible for RFM-based churn modelling.
2. **Missing Revenue:** 201 rows (10.1%) have no Revenue value. These are excluded from revenue calculations; their absence may slightly understate total revenue.
3. **Churn model sample size:** Only the subset of rows with valid dates AND non-missing Revenue is used for churn modelling. The small eligible set limits model reliability.
4. **Low Recall:** The Logistic Regression model captures only 15% of true churners. A more sophisticated model (e.g. Random Forest with class weighting, or SMOTE oversampling) could improve recall but was outside the scope of this project.
5. **Single-period snapshot:** The dataset covers a single time period. Without multi-year data, trend analysis and seasonality cannot be reliably assessed.
6. **Unknown records:** 26 Unknown Customer, 16 Unknown Region, and 7 Unknown Product records represent data capture gaps that prevent full analysis of those rows.
7. **Windows numpy compatibility:** numpy 1.26.4 hangs on import under Python 3.11 on Windows. The project requires numpy ≥ 2.0 (tested with 2.4.6). Any fresh installation must use the `requirements.txt` provided — **do not downgrade numpy**.
8. **churn_model.py local runtime:** `churn_model.py` is kept for reproducibility. On Windows, scikit-learn initialisation can be slow. The dashboard is architecturally independent of scikit-learn — it reads `churn_risk_table.csv` directly.

---

## Future Improvements

1. **Data quality at source** — enforce mandatory, validated `Order_Date` and `Customer_ID` fields at the point of sale to eliminate the 59.7% invalid-date problem.
2. **Class imbalance handling** — apply SMOTE or class-weight balancing to the Logistic Regression to improve churn Recall without sacrificing Precision.
3. **Advanced models** — evaluate Random Forest, XGBoost, or a neural network classifier against the Logistic Regression baseline.
4. **Cohort analysis** — with clean date data, monthly customer cohorts could reveal retention curves and identify when customers are most likely to lapse.
5. **Product profitability analysis** — investigate the 15 loss-making orders by product and region to identify systematic margin erosion.
6. **CLV modelling** — extend the RFM model to a Customer Lifetime Value (CLV) prediction to prioritise retention spend on the highest-value at-risk customers.
7. **Automated retraining** — build a pipeline that re-trains the churn model whenever new data is uploaded, keeping risk scores current.
8. **Live database connection** — replace CSV loading with a database connector (e.g. PostgreSQL, BigQuery) for real-time analytics.
