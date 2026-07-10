from argparse import Namespace
from logging import Logger

import numpy as np
import pandas as pd
import os
import json
from pathlib import Path

from sklearn.metrics import f1_score
import torch

from src.models.full import NetworkContainer
from src.models.rl import sample_action, select_from_action
from src.results.metrics import DATASET_CONFIGS, load_rl_model
from src.trainers.util import get_dataloaders, is_hypernet_variant, prepare_data

def gather_all_results_data():
    """
    Traverses the local runs/ directory to find all results.json files,
    selectively loading F1 score data into separate columns.
    Uses the original, robust path construction logic.
    """
    base_path = Path('./runs/classification/')
    
    if not base_path.exists():
        print(f"Error: Base path not found at '{base_path}'")
        return [], {}

    # Configuration for final thesis experiments
    seeds = range(0, 5)
    pooling_methods = ['MeanMLP', 'MaxMLP', 'AttentionMLP', 'repset']
    DEVICE = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    logger = Logger("Discard")

    # Data Gathering
    results_list = []    
    found_docs_count = {
        'Baseline': 0,
        'Task-agnostic Hypernetwork': 0,
        'Task-aware Hypernetwork': 0,
    }
    for seed in seeds:
        args = Namespace(
            data_embedded_column_name="instances",
            random_seed=seed,
            embedding_model="tabular",
            label="label",
            instance_labels_column=None,
            further_extra_columns=None,
            task_type="classification",
            balance_dataset=True,                               # Boolean flag for dataset balancing
            batch_size=20
        )
        train_dataset, eval_dataset, test_dataset, _ = prepare_data(args, logger)
        _, _, dataloader = \
            get_dataloaders(args, train_dataset, eval_dataset, test_dataset, logger)

        for pooling_base in pooling_methods:
            pooling_folder_name = f"{pooling_base}"
            
            base_model_path = (base_path / f'seed_{seed}' / 'instances/tabular/label/bag_size_20' / pooling_folder_name)
            base_mil_model_path = (base_path / f'seed_0' / 'instances/tabular/label/bag_size_20' / pooling_folder_name)

            if not base_model_path.is_dir():
                continue

            def gather_performance_metrics(net_container: NetworkContainer, variant: str, model_name: str):
                def process(pref=None):
                    print(f"Calculating performance metrics for method {pooling_base}, model {model_name}, seed {seed}, pref {pref}")
                    
                    if pref is not None:
                        net_container.set_preference(torch.fill(torch.zeros((1)), pref).to(DEVICE))

                    mean_eo_per_feature_pool = []
                    max_eo_pool = []
                    f1_pool = []

                    # --- 1. Processing Loop ---
                    for _ in range(2):
                        pred_ys, data_ys, protected_ys = [], [], []
                        
                        with torch.no_grad():
                            for batch_x, batch_y, indices, instance_labels in dataloader:
                                batch_x = batch_x.to(DEVICE)
                                # select batch_x
                                action_probs, _, _ = net_container.action(batch_x)
                                action, _ = sample_action(action_probs, 20, DEVICE, random=False, algorithm="without_replacement")
                                true_indices = (torch.max(batch_x[:, (2, 4, 5, 7), :], dim=-1).values - 1).to(dtype=torch.int64)
                                batch_x = select_from_action(action, batch_x)

                                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                                if is_hypernet_variant(variant):
                                    pred_out, _, _ = net_container.predict(torch.nn.CrossEntropyLoss(), batch_x, batch_y, true_indices)
                                else:
                                    pred_out, _, _ = net_container.predict(torch.nn.CrossEntropyLoss(), batch_x, batch_y)
                                
                                pred_y = torch.argmax(pred_out, dim=1)
                                pred_ys.append(pred_y.cpu())
                                data_ys.append(batch_y.cpu())
                                protected_ys.append(true_indices.cpu())

                        # Convert collected pool data to numpy arrays
                        all_preds = torch.cat(pred_ys, dim=0).numpy()
                        all_labels = torch.cat(data_ys, dim=0).numpy()
                        all_protected = torch.cat(protected_ys, dim=0).numpy() # Shape: [N, 4]
                        
                        # --- 2. Simplified Binary Equalized Odds Logic ---
                        # Helper function to get TPR and FPR for binary target (assuming 1 is positive, 0 is negative)
                        def get_binary_rates(labels, preds):
                            tp = np.sum((labels == 1) & (preds == 1))
                            fn = np.sum((labels == 1) & (preds != 1))
                            fp = np.sum((labels == 0) & (preds == 1))
                            tn = np.sum((labels == 0) & (preds != 0))
                            
                            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                            return tpr, fpr

                        # Global population rates as baseline
                        global_tpr, global_fpr = get_binary_rates(all_labels, all_preds)
                        
                        mean_eo_per_feature = []
                        num_protected_features = all_protected.shape[1] # Expected to be 4
                        
                        # Iterate through each of the 4 protected features
                        for f_idx in range(num_protected_features):
                            feature_column = all_protected[:, f_idx]
                            unique_subgroups = np.unique(feature_column)
                            
                            subgroup_eo_violations = []
                            
                            for subgroup in unique_subgroups:
                                mask = (feature_column == subgroup)
                                if np.sum(mask) == 0:
                                    continue
                                    
                                # Calculate rates for the specific protected subgroup
                                subgroup_tpr, subgroup_fpr = get_binary_rates(all_labels[mask], all_preds[mask])
                                
                                # Absolute difference compared to global population performance
                                tpr_diff = abs(subgroup_tpr - global_tpr)
                                fpr_diff = abs(subgroup_fpr - global_fpr)
                                
                                # Equalized odds violation is the worst-case disparity (maximum of TPR/FPR gap)
                                subgroup_eo_violations.append(max(tpr_diff, fpr_diff))
                            
                            # Mean over all unique classes/subgroups within this feature
                            feature_mean_eo = np.mean(subgroup_eo_violations) if subgroup_eo_violations else 0.0
                            mean_eo_per_feature.append(feature_mean_eo)
                        
                        # Track scores for this data block
                        mean_eo_per_feature_pool.append(mean_eo_per_feature)
                        max_eo_pool.append(np.max(mean_eo_per_feature))
                        f1_pool.append(f1_score(all_labels, all_preds, average='macro'))
                    
                    results_list.append({
                        'Framework': model_name,
                        'Pooling': pooling_base,
                        'Seed': seed,
                        "Preference": pref,
                        'Mean_F1': np.mean(f1_pool),
                        'STD_F1': np.std(f1_pool),
                        'Mean_EO': np.mean(max_eo_pool),
                        'STD_EO': np.std(max_eo_pool),
                    })
                    found_docs_count[model_name] += 1

                if is_hypernet_variant(variant):
                    for pref in np.linspace(0, 1, 11):
                        process(pref)
                else:
                    process()

            # Gather Simple MIL baseline results
            # process_json_file(base_model_path / 'results.json', 'Simple MIL')

            # Gather all RL model results
            for model_name, rl_config in DATASET_CONFIGS["models"].items():
                rl_path = base_model_path / rl_config["model_to_explain_suffix"]
                mil_path = base_mil_model_path / rl_config["model_to_explain_suffix"]
                try:
                    net_container = load_rl_model(rl_path, rl_config["variant"], mil_path=mil_path)
                except:
                    continue
                gather_performance_metrics(net_container, rl_config["variant"], model_name)
                            
    return results_list, found_docs_count

if __name__ == "__main__":
    all_results_data, doc_counts = gather_all_results_data()

    if all_results_data:
        final_df_clean = pd.DataFrame(all_results_data)
        output_filename = 'results/final_thesis_results.csv'
        os.makedirs('results', exist_ok=True)
        final_df_clean.to_csv(output_filename, index=False)
        print(final_df_clean.head())
        print("\nNumber of documents found per framework:")
        for framework, count in doc_counts.items():
            print(f"- {framework}: {count} documents")
    else:
        print("No result files were found.")
        