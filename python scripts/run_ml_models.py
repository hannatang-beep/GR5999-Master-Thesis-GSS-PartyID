# =============================================================================
# File: run_ml_models.py (Updated)
#
# Purpose:
# - Train Random Forest and XGBoost on LASSO-VIF-selected features
# - Evaluate performance on held-out test set
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# === Load training data ===
X_train = pd.read_csv("output/final_X_train_after_vif.csv")
y_train = pd.read_csv("output/y_train.csv").squeeze()  # series

# === Load raw test set and process ===
df_test = pd.read_csv("data/test_unseen.csv")
features = ["age", "sex", "race", "educ", "degree", "income", "wrkstat",
            "abany", "gunlaw", "natfare", "natenvir", "eqwlth", "sei", "hrs1",
            "relig", "reliten", "attend", "polviews"]
target = "partyid_3cat"

# Split variable types
continuous_vars = ["age", "educ", "income", "sei", "hrs1"]
categorical_vars = list(set(features) - set(continuous_vars))

# Impute
df_test[continuous_vars] = SimpleImputer(strategy="median").fit_transform(df_test[continuous_vars])
df_test[categorical_vars] = SimpleImputer(strategy="most_frequent").fit_transform(df_test[categorical_vars])

# Encode
ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
X_test_cat = ohe.fit(df_test[categorical_vars])  # fit on test to avoid crash
X_test_cat = ohe.transform(df_test[categorical_vars])
X_test_cat_cols = ohe.get_feature_names_out(categorical_vars)

# Combine with continuous
X_test = pd.concat([
    pd.DataFrame(X_test_cat, columns=X_test_cat_cols, index=df_test.index),
    df_test[continuous_vars]
], axis=1)
y_test = df_test[target]

# Match columns to training data
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# === Modeling ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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

for name, cfg in models.items():
    print(f"\n🔍 Training {name}...")
    grid = GridSearchCV(
        estimator=cfg["model"],
        param_grid=cfg["params"],
        cv=skf,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    joblib.dump(best_model, f"output/{name.lower()}_model.pkl")

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    auc_macro = roc_auc_score(pd.get_dummies(y_test), y_proba, average="macro")
    brier = np.mean([
        brier_score_loss((y_test == i).astype(int), y_proba[:, i])
        for i in range(len(np.unique(y_test)))
    ])

    print(f"✅ {name} Accuracy (test): {acc:.4f} | Macro F1: {macro_f1:.4f} | AUC: {auc_macro:.4f} | Brier: {brier:.4f}")

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Macro_F1": macro_f1,
        "AUC_Macro": auc_macro,
        "Brier_Score": brier,
        "Best_Params": grid.best_params_
    })

# Save results
pd.DataFrame(results).to_csv("output/ml_model_test_results.csv", index=False)
print("\n📁 Saved performance results to output/ml_model_test_results.csv")
