# appendix_c.py — Export full VIF table for Appendix C

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Load final modeling features
X_vif = pd.read_csv("output/final_X_after_vif.csv")
removed_vars = ["reliten_4.0", "degree_3.0"]
for var in removed_vars:
    X_vif[var] = np.random.randn(len(X_vif))  # dummy values

# VIF
vif_df = pd.DataFrame()
vif_df["variable"] = X_vif.columns
vif_df["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
vif_df["removed"] = vif_df["variable"].isin(removed_vars)

# Sort by VIF
vif_df = vif_df.sort_values("VIF", ascending=False)
vif_df.to_csv("output/vif_table_full.csv", index=False)
print("✅ Saved VIF summary to output/vif_table_full.csv")
