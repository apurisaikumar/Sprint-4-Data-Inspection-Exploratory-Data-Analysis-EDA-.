# Sprint 4 — Data Inspection & Exploratory Data Analysis (EDA)

A complete, hands-on EDA series applying the statistics (Sprint 2) and NumPy/Pandas
(Sprint 3) foundations to a real-world dataset — systematically inspecting, understanding,
analyzing, and visualizing data *before* any preprocessing or model building begins.

## Objective

An AI/ML Engineer should never start preprocessing or model building without first
understanding the data. This sprint follows one guiding principle across all 15
notebooks:

**Load → Inspect → Understand → Analyze → Visualize → Identify Problems → Document Findings**

## Primary Dataset

| Field | Detail |
|---|---|
| **Name** | Telco Customer Churn |
| **Source** | IBM Sample Data Sets (`IBM/telco-customer-churn-on-icp4d` GitHub mirror) |
| **Domain** | Telecommunications — Customer Churn |
| **Size** | 7,043 rows (customers) × 21 columns |
| **Target** | `Churn` (Yes/No) — binary classification |
| **Why selected** | Genuine real-world data, a healthy mix of numerical/categorical columns, a clear business target, and a real (if subtle) data-quality issue: `TotalCharges` is stored as text with 11 blank entries invisible to `.isnull()` |

A second dataset — the **Titanic passenger manifest** (891 rows × 12 columns, via the
`datasciencedojo/datasets` GitHub mirror) — is used only in Notebook 14 (EDA Mini
Challenge), deliberately chosen to be unfamiliar so that notebook is a genuine test of
independent EDA ability rather than a repeat of already-known findings.

Both dataset CSVs (`telco_churn.csv`, `titanic.csv`) are included alongside the notebooks
so everything runs standalone.

## Learning Methodology

Every concept in every notebook follows the same structure:

1. **Concept** — explained in plain language, in my own words.
2. **Example** — a simple, real-world, business, or AI/ML example, explaining why the
   analysis matters.
3. **Implementation** — a working Pandas/NumPy/Matplotlib/Seaborn implementation.
4. **Output** — the actual result or visualization.
5. **Interpretation** — what the result means.
6. **Insight** — what was learned from the data.
7. **AI/ML Relevance** — why this analysis matters for machine learning.

## Notebooks

| # | Notebook | Covers |
|---|---|---|
| 01 | `01_Dataset_Understanding.ipynb` | Dataset/Feature/Target definitions, independent/dependent variables, numerical/categorical/discrete/continuous features, identifier columns, metadata, dataset dimensions — plus full dataset documentation |
| 02 | `02_Data_Inspection.ipynb` | Rows, columns, dtypes, index, `head()`/`tail()`/`sample()`, `shape`, `info()`, `describe()`, `nunique()`, `value_counts()`, column classification summary |
| 03 | `03_Data_Quality.ipynb` | Missing values (incl. disguised `TotalCharges` blanks), duplicates, invalid values |
| 04 | `04_Univariate_Analysis.ipynb` | Full numeric stats + histogram/KDE/box plot for `tenure`/`MonthlyCharges`/`TotalCharges`; categorical frequency analysis |
| 05 | `05_Bivariate_Analysis.ipynb` | Numerical-numerical, numerical-categorical, categorical-categorical relationships; confirms 3 churn hypotheses |
| 06 | `06_Multivariate_Analysis.ipynb` | Correlation matrix, heatmap, pair plot, multicollinearity (VIF), feature interactions |
| 07 | `07_Outlier_Analysis.ipynb` | IQR/Z-score univariate detection (none found) + genuine multivariate outlier detection (112 found); reasoned keep/remove/transform decisions |
| 08 | `08_Distribution_Analysis.ipynb` | Normality testing, skewness, kurtosis, shape classification for every numeric feature |
| 09 | `09_Categorical_Analysis.ipynb` | Cardinality, rare categories, category imbalance, full categorical-vs-`Churn` scan |
| 10 | `10_Target_Analysis.ipynb` | Class distribution/counts/percentages/imbalance for `Churn`; why target analysis matters before modeling |
| 11 | `11_EDA_Visualization.ipynb` | 10 purpose-built visualizations (histogram, box, bar, count, scatter, line, violin, pair, heatmap, distribution plot), each answering a new question |
| 12 | `12_Business_Insights.ipynb` | EDA findings converted into observation → insight business statements, patterns, segments, and opportunities |
| 13 | `13_EDA_Report.ipynb` | Complete consolidated EDA summary: overview, data quality, statistical/relationship/visualization findings, problems, and recommendations |
| 14 | `14_EDA_Mini_Challenge.ipynb` | Independent, unguided EDA performed on the Titanic dataset |
| 15 | `15_Mini_Assessment_Sprint4.ipynb` | Self-assessment: conceptual questions, coding exercises, short-answer prompts — **no explanations included by design** |

## Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab
- Sprints 2 (Statistics) and 3 (NumPy & Pandas) — this sprint builds directly on both

## Libraries Used

```
pandas
numpy
matplotlib
seaborn
scipy
statsmodels
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels
```

## How to Use This Repository

1. Work through the notebooks in order — later notebooks build directly on earlier
   findings (e.g., Notebook 6's multicollinearity flag informs Notebook 13's
   recommendations; Notebook 9's categorical scan feeds Notebook 12's business insights).
2. Run every code cell yourself — every number quoted in this series' markdown was
   independently verified against actual executed output before being finalized, and the
   objective is to be able to explain every finding the same way during a review.
3. Attempt `15_Mini_Assessment_Sprint4.ipynb` last, without referring back to earlier
   notebooks.

## Key Findings Carried Into Sprint 5

- `TotalCharges` needs a type correction (`object` → `float64`) and its 11 missing
  values should be filled with 0 (all belong to brand-new, `tenure == 0` customers).
- `TotalCharges` shows meaningful multicollinearity with `tenure` × `MonthlyCharges`
  (VIF ≈ 8.1) — a candidate to drop or transform before linear modeling.
- 112 genuine multivariate billing outliers exist and should be **retained**, with the
  residual considered as a candidate engineered feature.
- `Churn` is moderately imbalanced (73.46% / 26.54%) — models must be evaluated with
  precision/recall/F1, not accuracy alone, using a stratified train/test split.
- The highest-risk customer segment is **Month-to-month contract + Fiber optic internet**
  (54.6% churn, 30.2% of the customer base).

## AI Usage Policy

Learning resources such as ChatGPT, official documentation, books, YouTube, and technical
blogs were used to *understand* each concept — not to copy analyses or implementations
directly. Every explanation, implementation, and business insight in these notebooks
reflects my own understanding and independently-verified findings.


