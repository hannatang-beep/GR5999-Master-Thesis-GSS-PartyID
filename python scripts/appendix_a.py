import pandas as pd
# Appendix A: Full list of initial variables

# variable summary from data_analysis
var_df = pd.read_csv("output/gss_variable_analysis.csv")

# final modeling variables after VIF filtering
ohe_vars = list(pd.read_csv("output/final_X_after_vif.csv").columns)

# Extract raw variable names
raw_modeling_vars = list(set(
[col.split("_")[0] for col in ohe_vars if "" in col] +
[col for col in ohe_vars if "_" not in col]
))
# Filter metadata for those variables
summary_df = var_df[var_df["variable"].isin(raw_modeling_vars)].copy()

# compute missing %
summary_df["percent_missing"] = round(
summary_df["null_count"] / (summary_df["non_null_count"] + summary_df["null_count"]) * 100, 2
)

# Reorder columns
summary_df = summary_df[["variable", "category", "non_null_count", "null_count", "percent_missing"]]

# Save to output
summary_df.to_csv("output/appendix_a_variable_summary.csv", index=False)
print("✅ Saved cleaned Appendix A variable summary to output/appendix_a_variable_summary.csv")