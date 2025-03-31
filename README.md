# 📊 Predicting Party Identification with GSS using ML

This project prepares General Social Survey (GSS) data for modeling political party identification (partyID) using LASSO and other machine learning models.

Below is an overview of each Python module and its role in the data pipeline.

## 🧠 Key Features
- Performs variable selection using LASSO with cross-validation and VIF filtering.
- Constructs a final multinomial logistic regression model based on LASSO-VIF selected predictors.
- Compares multinomial logistic regression model with tree-based classifiers (Random Forest, XGBoost)
- Uses 5-fold stratified cross-validation to tune hyperparameters and evaluate model performance(accuracy, macro F1, AUC, Brier).
- Outputs summary tables and figures for all model results and appendices.

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

### `run_ml_models.py`

**Purpose:**
- Train and tune Random Forest and XGBoost classifiers using LASSO-VIF selected variables
- Evaluate model performance (accuracy, macro F1)
- Generate performance comparison figure (Figure 4.2)

---

### `evaluate_model_scores.py`

**Purpose:**
- Compute additional evaluation metrics: AUC, Brier score
- Generate ROC and confusion matrix plots for both classifiers

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
│   ├── vif_table_full.csv
│   ├──appendix_a_variable_summary.csv
│   ├──appendix_a2_categorical_summary.csv
│   ├──appendix_a3_continuous_summary.csv   
│   ├── ml_model_comparison.csv
│   ├── figure_4_2_model_performance.png
│   ├── confusion_matrix_randomforest.png
│   ├── confusion_matrix_xgboost.png
│   ├── roc_curve_randomforest.png
│   └──  roc_curve_xgboost.png
├── python scripts/                # All analysis code
│   ├── data_analysis.py
│   ├── data_filtering.py
│   ├── generate_gss_2008_2012_partyid3.py
│   ├── check_missing_model_vars.py
│   ├── run_lasso_vif_pipeline.py
│   ├── run_ml_models.py              
│   ├── evaluate_model_scores.py      
│   ├── appendix_a.py
│   ├── appendix_e1_lasso_vif_final_model.py  
│   └── appendix_e2_ml_model_training.py       
├── requirements.txt              # Python dependencies
├── .gitignore                    # Exclude local files from GitHub
└── README.md                     # Project overview (this file)
```

---

## 📎 Appendix Reference

| Appendix | File | Description |
|----------|------|-------------|
| A.1 | `appendix_a_variable_summary.csv` | Metadata for final modeling variables (non-missing counts, categories) |
| A.2 | `appendix_a2_categorical_summary.csv` | Distribution of categorical predictors used in the model |
| A.3 | `appendix_a3_continuous_summary.csv` | Summary statistics (mean, std, min, max) for continuous variables |
| B | `final_model_coefficients.csv` | Coefficient estimates from the final multinomial logistic regression model |
| C | `vif_table_full.csv` | Variance Inflation Factor values after LASSO selection |
| D | `ml_model_comparison.csv` + plots | Classifier outputs, accuracy comparison (Figure 4.2), ROC, confusion matrix |
| E.1 | *(code snippet)* | LASSO + VIF + final model (multinomial logistic regression) |
| E.2 | *(code snippet)* | ML model training and performance visualization |

> 💡 *All appendix files are stored in the `/output/` folder.*


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
| `run_ml_models.py` | Run after LASSO-VIF selection to train classifiers | `ml_model_comparison.csv`, `figure_4_2_model_performance.png` |
| `evaluate_model_scores.py` | Run after model tuning to compute ROC / AUC / Brier | `confusion_matrix_*.png`, `roc_curve_*.png` |
---

## 🛠️ Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

---

## ✅ Notes

-Most outputs are saved to `/output/` and excluded from GitHub tracking via `.gitignore`
- However, selected appendix-related results (e.g., model coefficients, ROC plots, Figure 4.1/4.2) are tracked and committed to ensure thesis reproducibility
- Modeling emphasizes interpretability (via LASSO + VIF) and class balance
- Appendices are generated using scripts to ensure reproducibility