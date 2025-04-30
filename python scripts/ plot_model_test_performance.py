# =============================================================================
# File: plot_model_test_performance.py
#
# Purpose:
# - Visualize performance comparison across models on the test set
# - Metrics: Accuracy and Macro F1
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load test performance results
df = pd.read_csv("output/ml_model_test_results.csv")

# Keep only Accuracy and Macro_F1
df_plot = df[["Model", "Accuracy", "Macro_F1"]].copy()
df_plot.columns = ["Model", "Accuracy", "Macro F1"]

# Melt to long format
df_long = df_plot.melt(id_vars="Model", var_name="Metric", value_name="Score")

# Plot
plt.figure(figsize=(8, 6))
sns.barplot(data=df_long, x="Model", y="Score", hue="Metric")

plt.title("Figure 4.2: Model Performance Comparison on Test Set")
plt.ylim(0, 1.0)
plt.ylabel("Score")
plt.xlabel("Model")
plt.legend(title="Metric")
plt.tight_layout()

# Save to output
plt.savefig("output/figure_4_2_model_test_performance.png")
print("✅ Saved: output/figure_4_2_model_test_performance.png")
plt.show()
