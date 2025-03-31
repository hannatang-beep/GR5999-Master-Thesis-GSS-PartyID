# =============================================================================
# File: appendix_e2_ml_model_training.py
# 
# Purpose:
# - Appendix E.2 code for training Random Forest and XGBoost models
# - Uses LASSO-VIF-selected variables from 'final_X_after_vif.csv'
# - Saves accuracy and F1 scores to 'ml_model_comparison.csv'
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Load predictors and target
df_X = pd.read_csv("output/final_X_after_vif.csv")
df_y = pd.read_csv("data/gss_2008_2012_partyid3.csv")
y = df_y["partyid_3cat"]

# Define stratified 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Model configs
models = {
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1]
        }
    }
}

results = []

# Train and evaluate each model
for name, cfg in models.items():
    grid = GridSearchCV(
        estimator=cfg["model"],
        param_grid=cfg["params"],
        cv=skf,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(df_X, y)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(df_X)

    joblib.dump(best_model, f"output/{name.lower()}_model.pkl")

    acc = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro")

    results.append({
        "Model": name,
        "Best Params": grid.best_params_,
        "Accuracy": acc,
        "Macro_F1": macro_f1
    })

# Save performance results
pd.DataFrame(results).to_csv("output/ml_model_comparison.csv", index=False)