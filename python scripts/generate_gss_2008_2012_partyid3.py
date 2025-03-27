# =============================================================================
# File: generate_gss_2008_2012_partyid3.py
# 
# Purpose:
# - Create a modeling-ready subset of GSS data covering years 2008–2012.
# - Filter from the full dataset (gss_2008_2020.csv) produced by data_filtering.py.
# - Generate a 3-category party ID variable (Democrat, Independent, Republican).
# - Save the cleaned and standardized subset as gss_2008_2012_partyid3.csv.
#
# Notes:
# - This file assumes the full cleaned data exists in the "data" directory.
# - Column names are converted to lowercase for consistency in modeling.
# =============================================================================

import pandas as pd

# Load filtered full dataset
input_path = "./data/gss_2008_2020.csv"
df = pd.read_csv(input_path)

# Filter for years 2008 to 2012
df = df[df["year"].between(2008, 2012)].copy()

# Create partyid_3cat variable
# Mapping: 0–2 -> Democrat (0), 3 -> Independent (1), 4–6 -> Republican (2)
partyid_map = {
    0: 0, 1: 0, 2: 0,
    3: 1,
    4: 2, 5: 2, 6: 2
}
df = df[df["partyid"].isin(partyid_map.keys())].copy()
df["partyid_3cat"] = df["partyid"].map(partyid_map)

# Standardize column names to lowercase
df.columns = [col.lower() for col in df.columns]

# Save to new file
output_path = "./data/gss_2008_2012_partyid3.csv"
df.to_csv(output_path, index=False)
print(f"✅ Cleaned dataset saved to: {output_path}")
print(f"Final shape: {df.shape}")
print("Sample of partyid_3cat counts:")
print(df["partyid_3cat"].value_counts().sort_index())
