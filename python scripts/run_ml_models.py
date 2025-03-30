# =============================================================================
# File: run_ml_models.py
# 
# Purpose:
# - Train and evaluate Random Forest and XGBoost classifiers
# - Use the same predictor set from the LASSO-VIF pipeline (X_final)
# - Perform 5-fold stratified cross-validation with GridSearchCV for tuning
# - Output evaluation metrics for comparison with the multinomial model
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt


# Load data 
X = pd.read_csv("output/final_X_after_vif.csv")
y = pd.read_csv("data/gss_2008_2012_partyid3.csv")["partyid_3cat"]

print("\n✅ Loaded modeling dataset")

# Define CV strategy
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define models and parameter grids
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

# Train each model
for name, config in models.items():
    print(f"\n🔍 Tuning and training {name}...")
    grid = GridSearchCV(
        estimator=config["model"],
        param_grid=config["params"],
        cv=skf,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(X, y)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X)

    # Save model
    joblib.dump(best_model, f"output/{name.lower()}_model.pkl")

    # Evaluation
    acc = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro")

    print(f"✅ {name} Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
    print(classification_report(y, y_pred))

    results.append({
        "Model": name,
        "Best Params": grid.best_params_,
        "Accuracy": acc,
        "Macro_F1": macro_f1
    })

# Save results to CSV 
pd.DataFrame(results).to_csv("output/ml_model_comparison.csv", index=False)
print("\n📁 Model performance saved to 'output/ml_model_comparison.csv'")


# === Plot: Model Performance Comparison ===
# Reload performance result
df_perf = pd.read_csv("output/ml_model_comparison.csv")

# Convert model column to string + drop NA rows
df_perf["Model"] = df_perf["Model"].astype(str)
df_perf = df_perf.dropna(subset=["Model", "Accuracy", "Macro_F1"])

df_plot = df_perf[["Model", "Accuracy", "Macro_F1"]].copy()
df_plot.columns = df_plot.columns.astype(str)

# Melt DataFrame to long format
df_long = df_plot.melt(id_vars="Model", var_name="Metric", value_name="Score")

# to string or numeric
df_long["Model"] = df_long["Model"].astype(str)
df_long["Metric"] = df_long["Metric"].astype(str)
df_long["Score"] = pd.to_numeric(df_long["Score"], errors="coerce")

# debug: confirm structure
print("🧾 Final long-format DataFrame:\n", df_long)

# Plot
plt.figure(figsize=(8, 6))
sns.barplot(data=df_long, x="Model", y="Score", hue="Metric")
plt.title("Figure 4.2: Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.ylim(0, 1.0)
plt.legend(title="Metric")
plt.tight_layout()
plt.savefig("output/figure_4_2_model_performance.png")
plt.show()