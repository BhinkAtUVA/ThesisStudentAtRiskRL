from src.results.format_latex import latex_table

pooling_lookup = {
    "MeanMLP": "Mean",
    "MaxMLP": "Max",
    "AttentionMLP": "Attention",
    "repset": "Rep-The-Set"
}

with open("results/performance_full.tex", "w") as f:
    f.write(latex_table(
        "results/final_thesis_results.csv",
        ["l", "l", "r", "r", "r", None, "r", None],
        [None, None, 0, 1, 4, 4, 4, 4],
        ["Framework", "Pooling", "Seed", "Preference", "F1 score (performance)", None, "Equalized Odds (bias)", None],
        [(4, 5), (6, 7)],
        [None, None, None, None, None, None, None, None],
        [None, lambda x: pooling_lookup[x], None, None, None, None, None, None],
        True
    ))

with open("results/performance.tex", "w") as f:
    f.write(latex_table(
        "results/final_thesis_results_across_seeds.csv",
        ["l", "l", "r", "r", None, "r", None],
        [None, None, 1, 4, 4, 4, 4],
        ["Framework", "Pooling", "Preference", "F1 score (performance)", None, "Equalized Odds (bias)", None],
        [(3, 4), (5, 6)],
        [None, None, None, None, None, None, None],
        [None, lambda x: pooling_lookup[x], None, None, None, None, None],
        True
    ))

with open("results/perf_ttests.tex", "w") as f:
    f.write(latex_table(
        "results/performance_ttests.csv",
        ["l", "l", "r", "l"],
        [None, None, 4, None],
        ["Pooling", "Tested Hypothesis", "T value", "p value"],
        [],
        [None, None, None, None],
        [lambda x: pooling_lookup[x], None, None, None]
    ))

with open("results/bias_ttests.tex", "w") as f:
    f.write(latex_table(
        "results/bias_ttests.csv",
        ["l", "l", "r", "l"],
        [None, None, 4, None],
        ["Pooling", "Tested Hypothesis", "T value", "p value"],
        [],
        [None, None, None, None],
        [lambda x: pooling_lookup[x], None, None, None]
    ))

with open("results/perf_ttests_pareto.tex", "w") as f:
    f.write(latex_table(
        "results/performance_ttests_pareto.csv",
        ["l", "l", "l", "r", "l"],
        [None, None, None, 4, None],
        ["Pooling", "Framework", "Tested Hypothesis", "T value", "p value"],
        [],
        [None, None, None, None, None],
        [lambda x: pooling_lookup[x], None, None, None, None]
    ))

with open("results/bias_ttests_pareto.tex", "w") as f:
    f.write(latex_table(
        "results/bias_ttests_pareto.csv",
        ["l", "l", "l", "r", "l"],
        [None, None, None, 4, None],
        ["Pooling", "Framework", "Tested Hypothesis", "T value", "p value"],
        [],
        [None, None, None, None, None],
        [lambda x: pooling_lookup[x], None, None, None, None]
    ))

with open("results/MeanMLP_topk_cmp.tex", "w") as f:
    f.write(latex_table(
        "results/MeanMLP_topk_cmp.csv",
        ["r", "r", "r"],
        [None, None, None],
        [None, None, None],
        [],
        [None, None, None],
        [lambda x: x.replace("%", "\\%"), lambda x: x.replace("%", "\\%"), lambda x: x.replace("%", "\\%")]
    ))

with open("results/all_metrics.tex", "w") as f:
    f.write(latex_table(
        "results/master_all_metrics.csv",
        ["l", "l", "l", "r", "r", "r", "r", "r"],
        [None, 1, None, 4, 4, 4, 4, 4],
        [None, None, None, None, None, None, None, None],
        [],
        [None, None, None, None, None, None, None, None],
        [None, None, lambda x: pooling_lookup[x], None, None, None, None, None]
    ))