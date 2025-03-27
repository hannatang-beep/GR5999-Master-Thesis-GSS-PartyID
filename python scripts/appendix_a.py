import pandas as pd
# Appendix A: Full list of initial variables

# Load full variable metadata from earlier analysis
var_df = pd.read_csv("output/gss_variable_analysis.csv")

# Load final modeling variables after VIF filtering 
ohe_vars = list(pd.read_csv("output/final_X_after_vif.csv").columns)

# Extract raw variable names 
raw_modeling_vars = list(set(
    [col.split("_")[0] for col in ohe_vars if "_" in col] +
    [col for col in ohe_vars if "_" not in col]
))

# Filter metadata for those variables
summary_df = var_df[var_df["variable"].isin(raw_modeling_vars)].copy()

# Compute % missing
summary_df["percent_missing"] = round(
    summary_df["null_count"] / (summary_df["non_null_count"] + summary_df["null_count"]) * 100, 2
)

# Reorder columns 
summary_df = summary_df[["variable", "category", "non_null_count", "null_count", "percent_missing"]]

# Save summary for Appendix A
summary_df.to_csv("output/appendix_a_variable_summary.csv", index=False)
print("✅ Saved cleaned Appendix A variable summary to output/appendix_a_variable_summary.csv")

# Load and save additional summary tables for Appendix
# Categorical summary
cat_table = pd.read_csv("output/table_4_1_categorical_summary.csv")
cat_table["Variable"] = cat_table["Variable"].fillna(method="ffill")
cat_table.to_csv("output/appendix_a2_categorical_summary.csv", index=False)
print("📊 Saved categorical variable summary to output/appendix_a2_categorical_summary.csv")

# Continuous summary
cont_table = pd.read_csv("output/table_4_2_continuous_summary.csv")

# Round numeric columns for clean formatting
for col in ["Mean", "Std. Dev", "Min", "25%", "50%", "75%", "Max"]:
    if col in cont_table.columns:
        cont_table[col] = cont_table[col].round(2)

cont_table.to_csv("output/appendix_a3_continuous_summary.csv", index=False)
print("📈 Saved continuous variable summary to output/appendix_a3_continuous_summary.csv")
