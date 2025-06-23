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

- Load the full GSS dataset (`.dta` format)
- Inspect all available variables
- Analyze missing values, data types, and year availability
- Categorize variables into themes (e.g., demographic, economic, religious)

---

### `data_filtering.py`

- Load and filter GSS data from 2006–2021
- Select a consistent set of key variables
- Add an `election_cycle` variable based on year
- Output cleaned dataset to `data/gss_2008_2020.csv`

---

### `generate_gss_2008_2012_partyid3.py`

- Subset the dataset to include only years 2008–2012
- Recode GSS `partyid` into a new 3-category outcome variable:  
  - 0 = Democrat (original 0–2)  
  - 1 = Independent (3)  
  - 2 = Republican (4–6)
- Convert column names to lowercase
- Save as `data/gss_2008_2012_partyid3.csv`

---

### `check_missing_model_vars.py`

- Report missingness for selected modeling variables
- Evaluate sample size loss if dropping all rows with missing values
- Export missing value report to CSV

---

### `run_lasso_vif_pipeline.py`

- Perform multinomial logistic regression with LASSO
- Select variables via non-zero coefficients
- Remove multicollinearity using Variance Inflation Factor (VIF)
- Train final logistic regression model

---

### `run_ml_models.py`

- Train and tune Random Forest and XGBoost classifiers using LASSO-VIF selected variables
- Evaluate model performance (accuracy, macro F1)
- Generate performance comparison figure (Figure 4.2)

---

### `evaluate_model_scores.py`

- Compute additional evaluation metrics: AUC, Brier score
- Generate ROC and confusion matrix plots for both classifiers

---

### `plot_model_test_performance.py`

- Plot bar chart comparing test set accuracy and macro F1 (Figure 4.2)

---
### `appendix_a.py`

- Generate summary tables used in Appendices A.1–A.3:
  - `appendix_a_variable_summary.csv` – final model variables
  - `appendix_a2_categorical_summary.csv` – frequency tables
  - `appendix_a3_continuous_summary.csv` – descriptive stats

---

### `appendix_b.py`

- Load coefficients from the final multinomial model (`final_model_coefficients.csv`)
- Reshape into a wide-format table by party ID class (Democrat, Independent, Republican)
- Save as `lasso_coefficients_by_class.csv` for interpretability analysis

---

### `appendix_c.py`

- Load final predictors after LASSO-VIF
- Reintroduce dropped high-VIF variables with dummy data
- Compute Variance Inflation Factor (VIF) for all predictors
- Flag which variables were removed due to multicollinearity
- Output sorted VIF table

---

### `appendix_e1_lasso_vif_final_model.py`

- Preprocess data (impute, encode, scale)
- Run multinomial LASSO regression with L1 penalty
- Select variables and remove multicollinearity with VIF
- Fit final logistic regression model
- Plot top 20 predictors (Figure 4.1)

---
### `aappendix_e2_ml_model_training_testsplit.py`

 - Train and tune Random Forest and XGBoost classifiers using LASSO-VIF selected variables on training data
 - Evaluate test set performance using accuracy, macro F1, AUC, and Brier Score
 - Save results to `ml_model_test_results.csv` and `figure_4_2_model_test_performance.png`


## 📂 Folder Structure

```
project_root/
├── data/                          
│   ├── gss_2008_2020.csv                 # Filtered GSS dataset (2006–2021)
│   ├── gss_2008_2012_partyid3.csv        # Subset with 3-class party ID (2008–2012)
│   ├── train_balanced.csv                # Balanced training set
│   └── test_unseen.csv                   # Held-out test set (20%)
│
├── output/                               # All model outputs, plots, and appendix files
│   ├── gss_variable_analysis.csv
│   ├── model_var_missing_report.csv
│   ├── final_X_after_vif.csv             # Final predictors after LASSO + VIF
│   ├── y_train.csv                       # Training labels
│   ├── final_model_coefficients.csv
│   ├── lasso_coefficients_by_class.csv   # Appendix B: Coefficient table
│   ├── vif_table_full.csv                # Appendix C: Full VIF table
│
│   ├── ml_model_comparison.csv           # (deprecated: in-sample scores)
│   ├── ml_model_test_results.csv         # Updated: test set scores (accuracy, F1, AUC, Brier)
│
│   ├── figure_4_1_top20_coef.png         # Figure 4.1: Top predictors from LASSO
│   ├── figure_4_2_model_test_performance.png   # Figure 4.2: Accuracy & F1 on test set
│   ├── roc_curve_randomforest.png        # Figure 4.3 RF: ROC curve (OvR)
│   ├── roc_curve_xgboost.png             # Figure 4.3 XGB: ROC curve (OvR)
│   ├── confusion_matrix_randomforest.png # Appendix D.1
│   ├── confusion_matrix_xgboost.png      # Appendix D.2
│
│   ├── appendix_a_variable_summary.csv   # Appendix A.1
│   ├── appendix_a2_categorical_summary.csv  # Appendix A.2
│   ├── appendix_a3_continuous_summary.csv  # Appendix A.3
│
├── python scripts/                       # All modeling and processing scripts
│   ├── data_analysis.py
│   ├── data_filtering.py
│   ├── generate_gss_2008_2012_partyid3.py
│   ├── check_missing_model_vars.py
│   ├── run_lasso_vif_pipeline.py
│   ├── run_ml_models.py                  # (legacy, pre-test split version)
│   ├── appendix_e2_ml_model_training.py  # (legacy)
│   ├── appendix_e2_ml_model_training_testsplit.py  # Updated: RF/XGB on train-test split
│   ├── evaluate_model_scores.py
│   ├── plot_model_test_performance.py    # Plots Figure 4.2 bar chart
│   ├── appendix_a.py
│   ├── appendix_b.py
│   ├── appendix_c.py
│   └── appendix_e1_lasso_vif_final_model.py
│
├── requirements.txt
├── .gitignore
└── README.md

```

---

## 📎 Appendix Reference

| Appendix | File | Description |
|----------|------|-------------|
| A.1 | `appendix_a_variable_summary.csv` | Metadata for final modeling variables (non-missing counts, categories) |
| A.2 | `appendix_a2_categorical_summary.csv` | Distribution of categorical predictors used in the model |
| A.3 | `appendix_a3_continuous_summary.csv` | Summary statistics (mean, std, min, max) for continuous variables |
| B | `lasso_coefficients_by_class.csv` | Pivoted coefficient table showing impact of variables per party category |
| C | `vif_table_full.csv` | Full Variance Inflation Factor table, including dropped predictors |
| D | `ml_model_comparison.csv` + plots | Classifier outputs, accuracy comparison (Figure 4.2), ROC, confusion matrix |
| D.1 | `confusion_matrix_randomforest.png` | Random Forest test set confusion matrix |
| D.2 | `confusion_matrix_xgboost.png` | XGBoost test set confusion matrix |
| D.3 | `roc_curve_randomforest.png` | One-vs-rest ROC for Random Forest |
| D.4 | `roc_curve_xgboost.png` | One-vs-rest ROC for XGBoost |
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
| `run_ml_models.py` | Run after LASSO-VIF selection to train classifiers | `ml_model_comparison.csv`, `figure_4_2_model_performance.png` |
| `evaluate_model_scores.py` | Run after model tuning to compute ROC / AUC / Brier | `confusion_matrix_*.png`, `roc_curve_*.png` |
| `appendix_a.py` | Run to export clean appendix-ready variable table | `output/appendix_a_variable_summary.csv` |
| `appendix_b.py` | Run after final model is trained | `output/lasso_coefficients_by_class.csv` |
| `appendix_c.py` | Run to re-generate full VIF table for documentation | `output/vif_table_full.csv` |
| `appendix_e1_lasso_vif_final_model.py` | Run to reproduce LASSO + VIF + Final Model pipeline | `output/final_X_after_vif.csv, top20_multinomial_coef_plot.png` |
| `appendix_e2_ml_model_training.py` | Run to train RF/XGB and compare models | `ml_model_comparison.csv, figure_4_2_model_performance.png` |
---

## 🛠️ Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

---

## Notes

- Most outputs are saved to `/output/` and excluded from GitHub tracking via `.gitignore`
- However, selected appendix-related results (e.g., model coefficients, ROC plots, Figure 4.1/4.2) are tracked and committed to ensure thesis reproducibility
- Modeling emphasizes interpretability (via LASSO + VIF) and class balance
- Appendices are generated using scripts to ensure reproducibility
