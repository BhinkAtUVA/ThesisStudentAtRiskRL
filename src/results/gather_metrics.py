import numpy as np
import pandas as pd

spar = pd.read_csv("results/master_sparsity_metrics.csv")
cons = pd.read_csv("results/master_consistency_metrics.csv")
demo = pd.read_csv("results/master_demographics_metrics.csv")

all = pd.merge(spar, cons, on=["Model", "pooling_method"]).merge(demo, on=["Model", "pooling_method"]).drop(["seed", "seed_x", "seed_y"], axis=1)
all["Preference"] = all["Model"].map(lambda x: float(x.split("pref_")[1][0]) if "pref_" in x else np.nan)
all["Model"] = all["Model"].map(lambda x: x.split("pref_")[0] if "pref_" in x else x)
all = all[['Model', 'Preference', 'pooling_method', 'Gini Coefficient', 'HHI Score', 'Spearman\'s Rank (Stability)', 'Jaccard (K=5)','Reliance on Demographics (%)']].rename(columns={"pooling_method": "Pooling"})
all.to_csv("results/master_all_metrics.csv", index=False)