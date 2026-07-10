import numpy as np
import pandas as pd

df = pd.read_csv("results/final_thesis_results.csv")
df["Preference"] = np.round(df["Preference"], 1)

summarized = df.drop("Seed", axis=1).groupby(["Framework", "Pooling", "Preference"], dropna=False).mean().reset_index()
summarized.to_csv("results/final_thesis_results_across_seeds.csv", index=False)