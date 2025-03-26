import pandas as pd
# Appendix A: Full list of initial variables
var_df = pd.read_csv("output/gss_variable_analysis.csv")

# Keep only selected columns
summary_df = var_df[["variable", "non_null_count", "null_count", "category"]].copy()

# Add missing percent
summary_df["percent_missing"] = round(summary_df["null_count"] / (summary_df["non_null_count"] + summary_df["null_count"]) * 100, 2)

# Reorder columns
summary_df = summary_df[["variable", "category", "non_null_count", "null_count", "percent_missing"]]


# Save to CSV
summary_df.to_csv("output/appendix_a_variable_summary.csv", index=False)
print("✅ Saved to output/appendix_a_variable_summary.csv")