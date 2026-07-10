import numpy as np
import pandas as pd

from src.results.metrics import DATASET_CONFIGS
from scipy.stats import ttest_ind, ttest_ind_from_stats

df = pd.read_csv("results/final_thesis_results.csv")
df = df[df["Preference"] != 1]

perf_results = []
bias_results = []

for pooling in ["MeanMLP", "MaxMLP", "AttentionMLP", "repset"]:
    pooling_df = df[df["Pooling"] == pooling]
    perf = []
    bias = []
    framework_names = []

    for framework, config in DATASET_CONFIGS["models"].items():
        framework_names.append(framework)
        perf.append(pooling_df.loc[pooling_df["Framework"] == framework, "Mean_F1"])
        bias.append(pooling_df.loc[pooling_df["Framework"] == framework, "Mean_EO"])

    for i in range(len(DATASET_CONFIGS.keys())):
        for j in range(i + 1, len(framework_names)):
            perf_i = perf[i]
            perf_j = perf[j]

            if np.mean(perf_i) > np.mean(perf_j):
                big_idx = i
                small_idx = j
            else:
                big_idx = j
                small_idx = i
            tvalue, pvalue = ttest_ind(perf_i, perf_j)
            perf_results.append({
                "Pooling": pooling,
                "Test": f"{framework_names[big_idx]} > {framework_names[small_idx]}",
                "T value": f"{np.round(np.abs(tvalue), 4):.04f}",
                "p value": f"{np.round(pvalue, 4):.04f}{" (*)" if pvalue < 0.05 else ""}",
            })

            bias_i = bias[i]
            bias_j = bias[j]
            if np.mean(bias_i) > np.mean(bias_j):
                big_idx = i
                small_idx = j
            else:
                big_idx = j
                small_idx = i
            tvalue, pvalue = ttest_ind(bias_i, bias_j)
            bias_results.append({
                "Pooling": pooling,
                "Test": f"{framework_names[big_idx]} > {framework_names[small_idx]}",
                "T value": f"{np.round(np.abs(tvalue), 4):.04f}",
                "p value": f"{np.round(pvalue, 4):.04f}{" (*)" if pvalue < 0.05 else ""}",
            })

pd.DataFrame(perf_results).to_csv("results/performance_ttests.csv", index=False)
pd.DataFrame(bias_results).to_csv("results/bias_ttests.csv", index=False)

df = pd.read_csv("results/final_thesis_results.csv")
df = df[(df["Preference"] == 1) | (df["Preference"] == 0)]

perf_results = []
bias_results = []

for pooling in ["MeanMLP", "MaxMLP", "AttentionMLP", "repset"]:
    pooling_df = df[df["Pooling"] == pooling]
    framework_names = []

    for framework, config in DATASET_CONFIGS["models"].items():
        if framework == "Baseline": continue
        framework_names.append(framework)
        perf_focused = pooling_df[(pooling_df["Framework"] == framework) & (pooling_df["Preference"] == 0)]
        bias_focused = pooling_df[(pooling_df["Framework"] == framework) & (pooling_df["Preference"] == 1)]

        if len(perf_focused) > 1: tvalue, pvalue = ttest_ind(perf_focused["Mean_F1"], bias_focused["Mean_F1"])
        else:
            tvalue, pvalue = ttest_ind_from_stats(perf_focused["Mean_F1"], perf_focused["STD_F1"], 2, bias_focused["Mean_F1"], bias_focused["STD_F1"], 2)
            tvalue = tvalue[0]
            pvalue = pvalue[0]
        perf_results.append({
            "Pooling": pooling,
            "Framework": framework,
            "Test": f"Preference 0 > Preference 1",
            "T value": f"{np.round(tvalue, 4):.04f}",
            "p value": f"{np.round(pvalue, 4):.04f}{" (*)" if pvalue < 0.05 else ""}",
        })

        if len(perf_focused) > 1: tvalue, pvalue = ttest_ind(perf_focused["Mean_EO"], bias_focused["Mean_EO"])
        else:
            tvalue, pvalue = ttest_ind_from_stats(perf_focused["Mean_EO"], perf_focused["STD_EO"], 2, bias_focused["Mean_EO"], bias_focused["STD_EO"], 2)
            tvalue = tvalue[0]
            pvalue = pvalue[0]
        bias_results.append({
            "Pooling": pooling,
            "Framework": framework,
            "Test": f"Preference 0 > Preference 1",
            "T value": f"{np.round(tvalue, 4):.04f}",
            "p value": f"{np.round(pvalue, 4):.04f}{" (*)" if pvalue < 0.05 else ""}",
        })

pd.DataFrame(perf_results).to_csv("results/performance_ttests_pareto.csv", index=False)
pd.DataFrame(bias_results).to_csv("results/bias_ttests_pareto.csv", index=False)