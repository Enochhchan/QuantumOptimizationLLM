
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV files
df_35 = pd.read_csv("llm_qubo_results_detailed_with_fidelity_35.csv")
df_4 = pd.read_csv("llm_qubo_results_detailed_with_fidelity_gpt4_without_loop.csv")

# Label each with its model
df_35["Model"] = "GPT-3.5"
df_4["Model"] = "GPT-4"

# Combine into one DataFrame
df_all = pd.concat([df_35, df_4], ignore_index=True)

# Compute success rate by problem type and model
success_rate = df_all.groupby(["Case", "Model"])["Success"].mean().reset_index()

# Plot the success rate comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=success_rate, x="Case", y="Success", hue="Model")
plt.title("Success Rate Comparison: GPT-4 vs GPT-3.5")
plt.ylabel("Success Rate")
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("success_rate_comparison_gpt4_vs_gpt35.png", dpi=300)
plt.show()

success_rate

