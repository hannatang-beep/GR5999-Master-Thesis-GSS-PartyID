# run_lasso_vif_pipeline.py
# Purpose: Run LASSO variable selection, VIF filtering, and final model training

import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# === Step 1: Load data ===
file_path = "./data/gss_2008_2012_partyid3.csv"
df = pd.read_csv(file_path)
print(f"\n✅ Loaded dataset: {file_path} — shape: {df.shape}")

# === Step 2: Define target and features ===
target = "partyid_3cat"
features = [
    "age", "sex", "race", "educ", "degree", "income", "wrkstat",
    "abany", "gunlaw", "natfare", "natenvir", "eqwlth", "sei", "hrs1",
    "relig", "reliten", "attend", "polviews"
]

# === Step 3: Handle missing values ===
continuous_vars = ["age", "educ", "income", "sei", "hrs1"]
categorical_vars = list(set(features) - set(continuous_vars))

# Median imputation for continuous variables
df[continuous_vars] = SimpleImputer(strategy="median").fit_transform(df[continuous_vars])

# Mode imputation for categorical variables
df[categorical_vars] = SimpleImputer(strategy="most_frequent").fit_transform(df[categorical_vars])

# === Step 4: Encode categorical variables ===
ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
X_cat = ohe.fit_transform(df[categorical_vars])
X_cat_cols = ohe.get_feature_names_out(categorical_vars)

# Combine with continuous features
X_all = pd.concat([
    pd.DataFrame(X_cat, columns=X_cat_cols, index=df.index),
    df[continuous_vars]
], axis=1)
y = df[target]

# === Step 5: Normalize ===
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_all), columns=X_all.columns)

# === Step 6: LASSO Multinomial Logistic Regression ===
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
print("\n✅ LASSO model trained")

# Extract non-zero coefficient features
coef_matrix = pd.DataFrame(lasso_model.coef_, columns=X_scaled.columns)
selected_vars = coef_matrix.columns[(coef_matrix != 0).any(axis=0)].tolist()
print(f"\n📌 Variables selected by LASSO: {len(selected_vars)}")

# === Step 7: VIF Filtering ===
X_vif = X_scaled[selected_vars].copy()

while True:
    vif_series = pd.Series(
        [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])],
        index=X_vif.columns
    )
    max_vif = vif_series.max()
    if max_vif > 5:
        drop_var = vif_series.idxmax()
        print(f"⚠️ Dropping '{drop_var}' due to high VIF: {max_vif:.2f}")
        X_vif = X_vif.drop(columns=[drop_var])
    else:
        break

print(f"\n✅ Final predictors after VIF filtering: {X_vif.shape[1]} variables")

# === Step 8: Fit Final Model ===
final_model = LogisticRegression(
    penalty=None,
    solver="lbfgs",
    max_iter=10000
)
final_model.fit(X_vif, y)
print("\n✅ Final model trained")

# === Step 9: Export selected variables ===
X_vif.to_csv("output/final_X_after_vif.csv", index=False)
coef_df = pd.DataFrame(final_model.coef_, columns=X_vif.columns)
coef_df.to_csv("output/final_model_coefficients.csv", index=False)
print("\n📁 Results saved to 'output/' folder")


# plotting for top 20 most influential variables across classes
# figure 4.1
# Load coefficient data
df = pd.read_csv("output/final_model_coefficients.csv")
df["class"] = ["Democrat", "Independent", "Republican"]

df_long = df.melt(id_vars="class", var_name="variable", value_name="coefficient")
top_vars = df_long.groupby("variable")["coefficient"].apply(lambda x: x.abs().mean()).nlargest(20).index
top_df = df_long[df_long["variable"].isin(top_vars)]

plt.figure(figsize=(10, 8))
sns.barplot(data=top_df, x="coefficient", y="variable", hue="class", dodge=True, orient="h")
plt.axvline(0, color="gray", linestyle="--")
plt.title("Top 20 Influential Predictors (Multinomial LASSO)")
plt.xlabel("Coefficient", fontsize=12)
plt.ylabel("Variable", fontsize=12)
plt.legend(title="Party ID Category")
plt.tight_layout()
plt.savefig("output/top20_multinomial_coef_plot.png")
plt.show()






