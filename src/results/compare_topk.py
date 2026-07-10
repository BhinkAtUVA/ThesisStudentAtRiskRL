import pandas as pd

from src.results.metrics import DATASET_CONFIGS


POOLING_METHOD = "MeanMLP"

common_instances = None
for framework, config in DATASET_CONFIGS["models"].items():
    if config["variant"] == "baseline":
        df = pd.read_csv(f"results/{POOLING_METHOD}_{framework}_top20.csv")
        if common_instances is None: common_instances = set(df["Feature Type"])
        else: common_instances = common_instances.intersection(set(df["Feature Type"]))
    elif config["variant"] == "hypernet_rl":
        for pref in range(2):
            df = pd.read_csv(f"results/{POOLING_METHOD}_{framework}pref_{pref}-0_top20.csv")
            if common_instances is None: common_instances = set(df["Feature Type"])
            else: common_instances = common_instances.intersection(set(df["Feature Type"]))

cmp_topk = {}

for framework, config in DATASET_CONFIGS["models"].items():
    
    if config["variant"] == "baseline":
        df = pd.read_csv(f"results/{POOLING_METHOD}_{framework}_top20.csv")
        values = []
        for i, row in df.iterrows():
            if row["Feature Type"] not in common_instances: continue
            values.append(f"{row["Feature Type"]} ({round(row["Bag Presence %"], 4):.04f} %)")
        cmp_topk[f"{framework}"] = values
    elif config["variant"] == "hypernet_rl":
        for pref in range(2):
            df = pd.read_csv(f"results/{POOLING_METHOD}_{framework}pref_{pref}-0_top20.csv")
            values = []
            for i, row in df.iterrows():
                if row["Feature Type"] not in common_instances: continue
                values.append(f"{row["Feature Type"]} ({round(row["Bag Presence %"], 4):.04f} %)")
            cmp_topk[f"{framework}, preference {pref}"] = values
pd.DataFrame(cmp_topk).to_csv(f"results/{POOLING_METHOD}_topk_cmp.csv", index=False)