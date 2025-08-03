# =============================================================================
# File: check_missing_model_vars.py
# 
# Purpose:
# - Check the availability and missingness of key modeling variables.
# - Provide a summary table of non-null counts and missing value rates.
# - Help identify variables with excessive missingness before modeling.
#
# Notes:
# - This script should be run after the full GSS dataset has been filtered.
# - The key variable list should match the one used in modeling scripts.
# - Output: 'output/model_var_missing_report.csv' for reference and documentation.
# =============================================================================


import pandas as pd

# Load modeling dataset
file_path = "data/gss_2008_2012_partyid3.csv"
df = pd.read_csv(file_path)

print("\n Loaded dataset:", file_path)
print("Shape:", df.shape)

# Calculate missing value percentage 
missing_report = (
    df.isnull().sum()
    .to_frame(name="missing_count")
    .assign(percent_missing=lambda x: round(100 * x["missing_count"] / len(df), 2))
    .sort_values("percent_missing", ascending=False)
)

# Only show variables with > 0% missing
missing_report = missing_report[missing_report["percent_missing"] > 0]

# Output result 
print("\n Missing Value Summary (sorted):\n")
print(missing_report)

# Save to output folder
missing_report.to_csv("output/model_var_missing_report.csv")
print("\n Report saved to: output/model_var_missing_report.csv")

# Check how many rows remain after dropping missing values ===

# Define final modeling variables (including target)
modeling_vars = [
    "partyid_3cat", "age", "sex", "race", "educ", "degree", "income", "wrkstat",
    "abany", "gunlaw", "natfare", "natenvir", "eqwlth", "sei", "hrs1",
    "relig", "reliten", "attend", "polviews"
]

# Drop rows with any missing in modeling variables
df_model = df[modeling_vars].dropna()
print(f"\n Remaining observations after dropping missing: {len(df_model)}")

# Plot missing value bar chart ===
import matplotlib.pyplot as plt

# Create a DataFrame with missing values for modeling variables
missing_df = df[modeling_vars].isnull().sum().reset_index()
missing_df.columns = ["variable", "missing_count"]
missing_df["percent_missing"] = (missing_df["missing_count"] / len(df) * 100).round(2)
missing_df = missing_df.sort_values(by="percent_missing", ascending=False)

# plot bar chart
plt.figure(figsize=(10, 6))
plt.barh(missing_df["variable"], missing_df["percent_missing"], color="skyblue")
plt.xlabel("Percent Missing")
plt.title("Missing Values by Variable (Modeling Variables)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("output/missing_model_vars.png")
print("\n Missing value plot saved to: output/missing_model_vars.png")
