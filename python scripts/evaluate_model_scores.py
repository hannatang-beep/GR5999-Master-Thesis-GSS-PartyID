# =============================================================================
# File: evaluate_model_scores.py (Updated for test set)
#
# Purpose:
# - Evaluate RF and XGB on held-out test set
# - Output ROC, AUC, Brier score, and confusion matrix
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    roc_auc_score, roc_curve, brier_score_loss,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# === Load test data ===
df_test = pd.read_csv("data/test_unseen.csv")
target = "partyid_3cat"
features = ["age", "sex", "race", "educ", "degree", "income", "wrkstat",
            "abany", "gunlaw", "natfare", "natenvir", "eqwlth", "sei", "hrs1",
            "relig", "reliten", "attend", "polviews"]
continuous_vars = ["age", "educ", "income", "sei", "hrs1"]
categorical_vars = list(set(features) - set(continuous_vars))

# Preprocess test set
df_test[continuous_vars] = SimpleImputer(strategy="median").fit_transform(df_test[continuous_vars])
df_test[categorical_vars] = SimpleImputer(strategy="most_frequent").fit_transform(df_test[categorical_vars])

# One-hot encode
ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
X_test_cat = ohe.fit_transform(df_test[categorical_vars])
X_test_cat_cols = ohe.get_feature_names_out(categorical_vars)

# Combine with continuous
X_test = pd.concat([
    pd.DataFrame(X_test_cat, columns=X_test_cat_cols, index=df_test.index),
    df_test[continuous_vars]
], axis=1)
y_test = df_test[target]

# Align with train columns
X_train = pd.read_csv("output/final_X_train_after_vif.csv")
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Scale (use same scaler assumption)
scaler = StandardScaler()
X_test_scaled = pd.DataFrame(scaler.fit(X_train).transform(X_test), columns=X_test.columns)

# Model names and class labels
model_names = ["randomforest", "xgboost"]
class_names = ["Democrat", "Independent", "Republican"]

# === Evaluate each model ===
for name in model_names:
    print(f"\n📊 Evaluating {name}...")

    model = joblib.load(f"output/{name}_model.pkl")
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix – {name.title()}")
    plt.savefig(f"output/confusion_matrix_{name}.png")
    plt.close()

    # ROC Curve (One-vs-Rest)
    plt.figure(figsize=(8, 6))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_proba[:, i])
        auc = roc_auc_score((y_test == i).astype(int), y_proba[:, i])
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (OvR) – {name.title()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"output/roc_curve_{name}.png")
    plt.close()

    # AUC + Brier
    auc_macro = roc_auc_score(pd.get_dummies(y_test), y_proba, average="macro")
    brier = np.mean([
        brier_score_loss((y_test == i).astype(int), y_proba[:, i])
        for i in range(len(class_names))
    ])

    print(f" AUC (macro): {auc_macro:.4f}")
    print(f" Brier Score: {brier:.4f}")
