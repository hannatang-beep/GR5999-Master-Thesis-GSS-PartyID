import pandas as pd

# Load coefficient data
coef_df = pd.read_csv("output/final_model_coefficients.csv")
coef_df["class"] = ["Democrat", "Independent", "Republican"]

# Reshape to tidy format
coef_long = coef_df.melt(id_vars="class", var_name="variable", value_name="coefficient")

# Pivot so each row = variable, each column = class
coef_pivot = coef_long.pivot(index="variable", columns="class", values="coefficient").reset_index()

# Save
coef_pivot.to_csv("output/lasso_coefficients_by_class.csv", index=False)
print("✅ Saved to output/lasso_coefficients_by_class.csv")
