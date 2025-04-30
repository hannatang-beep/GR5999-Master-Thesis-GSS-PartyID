# =============================================================================
# File: run_lasso_vif_pipeline.py (Updated)
#
# Purpose:
# - Run LASSO feature selection and VIF filtering on training set
# - Save cleaned, encoded, scaled, and filtered training data
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load balanced training data
df = pd.read_csv("data/train_balanced.csv")
print(f"\n✅ Loaded balanced training set — shape: {df.shape}")

# Define target and feature list
target = "partyid_3cat"
features = [
    "age", "sex", "race", "educ", "degree", "income", "wrkstat",
    "abany", "gunlaw", "natfare", "natenvir", "eqwlth", "sei", "hrs1",
    "relig", "reliten", "attend", "polviews"
]

# Split feature types
continuous_vars = ["age", "educ", "income", "sei", "hrs1"]
categorical_vars = list(set(features) - set(continuous_vars))

# Impute missing values
df[continuous_vars] = SimpleImputer(strategy="median").fit_transform(df[continuous_vars])
df[categorical_vars] = SimpleImputer(strategy="most_frequent").fit_transform(df[categorical_vars])

# One-hot encode categorical variables
ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
X_cat = ohe.fit_transform(df[categorical_vars])
X_cat_cols = ohe.get_feature_names_out(categorical_vars)

# Combine with continuous
X_all = pd.concat([
    pd.DataFrame(X_cat, columns=X_cat_cols, index=df.index),
    df[continuous_vars]
], axis=1)
y = df[target]

# Standardize
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_all), columns=X_all.columns)

# Multinomial LASSO
lasso = LogisticRegressionCV(
    Cs=10,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    penalty="l1",
    solver="saga",
    multi_class="multinomial",
    max_iter=10000,
    random_state=42
)
lasso.fit(X_scaled, y)
print("✅ LASSO model trained")

# Select non-zero variables
coef_matrix = pd.DataFrame(lasso.coef_, columns=X_scaled.columns)
selected_vars = coef_matrix.columns[(coef_matrix != 0).any(axis=0)].tolist()
print(f"📌 Variables selected by LASSO: {len(selected_vars)}")

# VIF filtering
X_vif = X_scaled[selected_vars].copy()
while True:
    vif_series = pd.Series(
        [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])],
        index=X_vif.columns
    )
    if vif_series.max() > 5:
        drop_var = vif_series.idxmax()
        print(f"⚠️ Dropping '{drop_var}' due to VIF = {vif_series.max():.2f}")
        X_vif = X_vif.drop(columns=drop_var)
    else:
        break

print(f"✅ Final predictors after VIF: {X_vif.shape[1]}")

# Train final multinomial model (no penalty)
final_model = LogisticRegression(
    penalty=None,
    solver="lbfgs",
    multi_class="multinomial",
    max_iter=10000
)
final_model.fit(X_vif, y)
print("✅ Final multinomial model trained")

# Save outputs
X_vif.to_csv("output/final_X_train_after_vif.csv", index=False)
pd.Series(y).to_csv("output/y_train.csv", index=False)
coef_df = pd.DataFrame(final_model.coef_, columns=X_vif.columns)
coef_df.to_csv("output/final_model_coefficients.csv", index=False)

print("📁 Outputs saved to /output/:")
print("- final_X_train_after_vif.csv")
print("- y_train.csv")
print("- final_model_coefficients.csv")
