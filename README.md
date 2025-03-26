# 📊 GSS Data Processing Pipeline

This project prepares General Social Survey (GSS) data for modeling political party identification (partyID) using LASSO and other machine learning models.

Below is an overview of each Python module and its role in the data pipeline.

## 🧠 Key Features
- Handles variable selection using LASSO with VIF filtering
- Supports missing data imputation and variable encoding
- Compares logistic regression with tree-based classifiers (Random Forest, XGBoost)
- Outputs cleaned datasets and publication-ready summary tables for appendices

## 🧹 Files Overview

### `data_analysis.py`

**Purpose:**

- Load the full GSS dataset (`.dta` format)
- Inspect all available variables
- Analyze missing values, data types, and year availability
- Categorize variables into themes (e.g., demographic, economic, religious)

**Use case:**  
Used for early-stage exploratory data analysis and to guide variable selection decisions.

---

### `data_filtering.py`

**Purpose:**

- Load and filter GSS data from 2006–2021
- Select a consistent set of key variables
- Add an `election_cycle` variable based on year
- Output cleaned dataset to `data/gss_2008_2020.csv`

**Use case:**  
Prepares the full working dataset for all downstream modeling.

---

### `generate_gss_2008_2012_partyid3.py`

**Purpose:**

- Subset the dataset to include only years 2008–2012
- Recode GSS `partyid` into a new 3-category outcome variable:  
  - 0 = Democrat (original 0–2)  
  - 1 = Independent (3)  
  - 2 = Republican (4–6)
- Convert column names to lowercase
- Save as `data/gss_2008_2012_partyid3.csv`

**Use case:**  
Creates the modeling-ready dataset for predicting partisan identification.

---

### `check_missing_model_vars.py`

**Purpose:**

- Report missingness for selected modeling variables
- Evaluate sample size loss if dropping all rows with missing values
- Export missing value report to CSV

**Use case:**  
To guide imputation or variable exclusion decisions.

---

### `run_lasso_vif_pipeline.py`

**Purpose:**

- Perform multinomial logistic regression with LASSO
- Select variables via non-zero coefficients
- Remove multicollinearity using Variance Inflation Factor (VIF)
- Train final logistic regression model

**Use case:**  
Core modeling script for 3-category party ID outcome.

---

### `appendix_a.py`

- Generate summary tables used in Appendices A.1–A.3:
  - `appendix_a_variable_summary.csv` – final model variables
  - `appendix_a2_categorical_summary.csv` – frequency tables
  - `appendix_a3_continuous_summary.csv` – descriptive stats



## 📂 Folder Structure

```
project_root/
├── data/                          # Intermediate datasets
│   ├── gss_2008_2020.csv
│   └── gss_2008_2012_partyid3.csv
├── output/                        # Results, plots, appendix exports
│   ├── gss_variable_analysis.csv
│   ├── model_var_missing_report.csv
│   ├── top20_multinomial_coef_plot.png
│   ├── vif_table.csv
│   └── appendix_a_variable_summary.csv
├── python scripts/                # All analysis code
│   ├── data_analysis.py
│   ├── data_filtering.py
│   ├── generate_gss_2008_2012_partyid3.py
│   ├── check_missing_model_vars.py
│   ├── run_lasso_vif_pipeline.py
│   └── appendix_a.py
├── requirements.txt              # Python dependencies
├── .gitignore                    # Exclude local files from GitHub
└── README.md                     # Project overview (this file)
```

---

## 📎 Appendix Reference

| Appendix | File | Description |
|----------|------|-------------|
| A | `appendix_a_variable_summary.csv` | Variable metadata and categories |
| C | `vif_table.csv` | VIF results after LASSO |
| (Chapter 4) | `top20_multinomial_coef_plot.png` | Top 20 predictors from multinomial model |

> Note: Code snippets will be included in **Appendix E** (not committed yet)

---

## 🧠 Script Execution Guide

| Script | When to Run | Output |
|--------|-------------|--------|
| `data_analysis.py` | Run if you want to review or update variable summaries | `output/gss_variable_analysis.csv` |
| `data_filtering.py` | Run if `.dta` file or key variable list changes | `data/gss_2008_2020.csv` |
| `generate_gss_2008_2012_partyid3.py` | Run to refresh 2008–2012 modeling dataset | `data/gss_2008_2012_partyid3.csv` |
| `check_missing_model_vars.py` | Run if modeling variable list or missing logic changes | `output/model_var_missing_report.csv` |
| `run_lasso_vif_pipeline.py` | Run to retrain models and output final results | `output/` folder files |
| `appendix_a.py` | Run to export clean appendix-ready variable table | `output/appendix_a_variable_summary.csv` |

---

## 🛠️ Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

---

## ✅ Notes

- All outputs (.csv, .png) are saved to /output/ and excluded from GitHub tracking via .gitignore
- Modeling emphasizes interpretability (via LASSO + VIF) and class balance
- Appendices are generated using scripts to ensure reproducibility