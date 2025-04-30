# =============================================================================
# File: generate_gss_2008_2012_partyid3.py
# 
# Purpose:
# - Create a modeling-ready subset of GSS data covering years 2008–2012.
# - Filter from the full dataset (gss_2008_2020.csv) produced by data_filtering.py.
# - Recode PARTYID into a 3-category target variable: Democrat, Independent, Republican.
# - Perform an 80/20 stratified train-test split.
# - Balance the class distribution in the training set using undersampling.
# - Save two files: train_balanced.csv and test_unseen.csv for modeling.
#
# Notes:
# - This script assumes the cleaned full dataset exists in the "data" directory.
# - Only the training set is class-balanced; the test set retains original proportions.
# - Column names are converted to lowercase for consistency in modeling.
# =============================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Load dataset
df = pd.read_csv("./data/gss_2008_2020.csv")

# Filter 2008–2012
df = df[df["year"].between(2008, 2012)].copy()

# Recode partyid to 3-category outcome
partyid_map = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 2, 6: 2}
df = df[df["partyid"].isin(partyid_map)].copy()
df["partyid_3cat"] = df["partyid"].map(partyid_map)

# Standardize column names
df.columns = [col.lower() for col in df.columns]

# Train-test split (stratified, 80/20)
X = df.drop(columns=["partyid_3cat"])
y = df["partyid_3cat"]
X["partyid_3cat"] = y  # add back for merge

df_train, df_test = train_test_split(
    X, test_size=0.2, stratify=y, random_state=42
)

# Balance train set only
train_df = df_train.copy()
balanced_df = []

for cls in train_df["partyid_3cat"].unique():
    group = train_df[train_df["partyid_3cat"] == cls]
    n_minority = train_df["partyid_3cat"].value_counts().min()
    balanced = resample(group, replace=False, n_samples=n_minority, random_state=42)
    balanced_df.append(balanced)

df_train_balanced = pd.concat(balanced_df)

# Save results
df_train_balanced.to_csv("data/train_balanced.csv", index=False)
df_test.to_csv("data/test_unseen.csv", index=False)

print("✅ Saved training (balanced): data/train_balanced.csv")
print("✅ Saved test set (unseen): data/test_unseen.csv")
print(f"Train size: {df_train_balanced.shape[0]}, Test size: {df_test.shape[0]}")
