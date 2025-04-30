# =============================================================================
# File: appendix_e2_ml_model_training_testsplit.py
#
# Purpose:
# - Train Random Forest and XGBoost using training set only
# - Evaluate on held-out test set using LASSO-VIF-selected variables
# - Save performance metrics for reporting (accuracy, F1, AUC, Brier)
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, brier_score_loss
)
import joblib

# Load training and test sets
X_train = pd.read_csv("data/train_balanced.csv").drop(columns=["partyid_3cat"])
y_train = pd.read_csv("data/train_balanced.csv")["partyid_3cat"]
X_test = pd.read_csv("data/test_unseen.csv").drop(columns=["partyid_3cat"])
y_test = pd.read_csv("data/test_unseen.csv")["partyid_3cat"]

# CV strategy
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define models
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

    # Save trained model
    joblib.dump(best_model, f"output/{name.lower()}_model.pkl")

    # Evaluate on test set
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
        "Best Params": grid.best_params_,
        "Accuracy": round(acc, 4),
        "Macro_F1": round(macro_f1, 4),
        "AUC_macro": round(auc_macro, 4),
        "Brier": round(brier, 4)
    })

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv("output/ml_model_test_results.csv", index=False)
print("\n📁 Saved performance results to output/ml_model_test_results.csv")
