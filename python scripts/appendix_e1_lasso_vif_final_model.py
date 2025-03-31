# =============================================================================
# File: appendix_e1_lasso_vif_final_model.py
# 
# Purpose:
# - Appendix E.1 code for LASSO variable selection, VIF filtering, and final model
# - Minimal reproducible version (no plotting/debugging)
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load dataset
file_path = "./data/gss_2008_2012_partyid3.csv"
df = pd.read_csv(file_path)

# Define target and features
features = [
    "age", "sex", "race", "educ", "degree", "income", "wrkstat",
    "abany", "gunlaw", "natfare", "natenvir", "eqwlth", "sei", "hrs1",
    "relig", "reliten", "attend", "polviews"
]
target = "partyid_3cat"

# Handle missing values
continuous_vars = ["age", "educ", "income", "sei", "hrs1"]
categorical_vars = list(set(features) - set(continuous_vars))
df[continuous_vars] = SimpleImputer(strategy="median").fit_transform(df[continuous_vars])
df[categorical_vars] = SimpleImputer(strategy="most_frequent").fit_transform(df[categorical_vars])

# Encode categorical variables
ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
X_cat = ohe.fit_transform(df[categorical_vars])
X_cat_cols = ohe.get_feature_names_out(categorical_vars)

# Combine with continuous features
X_all = pd.concat([
    pd.DataFrame(X_cat, columns=X_cat_cols, index=df.index),
    df[continuous_vars]
], axis=1)
y = df[target]

# Normalize
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_all), columns=X_all.columns)

# LASSO Multinomial Logistic Regression
lasso_model = LogisticRegressionCV(
    Cs=10,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    penalty="l1",
    solver="saga",
    multi_class="multinomial",
    max_iter=10000,
    random_state=42
)
lasso_model.fit(X_scaled, y)

# Extract non-zero coefficient features
coef_matrix = pd.DataFrame(lasso_model.coef_, columns=X_scaled.columns)
selected_vars = coef_matrix.columns[(coef_matrix != 0).any(axis=0)].tolist()

# VIF Filtering
X_vif = X_scaled[selected_vars].copy()
while True:
    vif_series = pd.Series(
        [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])],
        index=X_vif.columns
    )
    max_vif = vif_series.max()
    if max_vif > 5:
        drop_var = vif_series.idxmax()
        X_vif = X_vif.drop(columns=[drop_var])
    else:
        break

# Final Multinomial Logistic Regression
final_model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000)
final_model.fit(X_vif, y)

# Save results
X_vif.to_csv("output/final_X_after_vif.csv", index=False)
coef_df = pd.DataFrame(final_model.coef_, columns=X_vif.columns)
coef_df.to_csv("output/final_model_coefficients.csv", index=False)
