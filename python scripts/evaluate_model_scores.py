# =============================================================================
# File: evaluate_model_scores.py
# 
# Purpose:
# - Evaluate trained RandomForest and XGBoost models using:
#   - ROC curves (OvR)
#   - Confusion matrices
#   - AUC scores (macro + per class)
#   - Brier scores
# - Visualize results and save figures to output/
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, brier_score_loss
import joblib

# Load data
X = pd.read_csv("output/final_X_after_vif.csv")
y = pd.read_csv("data/gss_2008_2012_partyid3.csv")["partyid_3cat"]

# Model names
model_names = ["randomforest", "xgboost"]
class_names = ["Democrat", "Independent", "Republican"]

for name in model_names:
    print(f"\n📊 Evaluating {name}...")
    model = joblib.load(f"output/{name}_model.pkl")
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix: {name.title()}")
    plt.savefig(f"output/confusion_matrix_{name}.png")
    plt.close()

    # ROC Curve (One-vs-Rest)
    plt.figure(figsize=(8, 6))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve((y == i).astype(int), y_proba[:, i])
        auc = roc_auc_score((y == i).astype(int), y_proba[:, i])
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Figure 4.3: ROC Curve (OvR) — {name.title()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"output/roc_curve_{name}.png")
    plt.close()

    # AUC Macro
    auc_macro = roc_auc_score(pd.get_dummies(y), y_proba, average="macro")
    print(f"✅ AUC (macro): {auc_macro:.4f}")

    # Brier Score (avg of 3 classes)
    brier = np.mean([
        brier_score_loss((y == i).astype(int), y_proba[:, i])
        for i in range(len(class_names))
    ])
    print(f"✅ Brier Score (avg): {brier:.4f}")
