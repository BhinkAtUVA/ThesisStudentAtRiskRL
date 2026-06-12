# Imports
from argparse import Namespace
from logging import Logger

import pandas as pd
import numpy as np
import os
import pickle
import torch
from collections import defaultdict
from scipy.stats import spearmanr
import shap
import json

from src.models.full import HypernetRLMIL, NetworkContainer
from src.models.mil import create_mil_model_with_dict
from src.trainers.util import create_debiasing_model, get_dataloaders, prepare_data

# Main Configuration
SEED_TO_ANALYZE = 0
POOLING_METHOD_TO_ANALYZE = "MeanMLP"
TOP_K_FOR_COUNTS = 20
REPRODUCIBILITY_SEED = 42
DEMOGRAPHIC_LABELS = ['Imd Band', 'Region', 'Gender', 'Age Band']

OUTPUT_DIR = 'results/'


# These depend on which specific experiments you did
DATASET_CONFIGS = {
    "base_path": f'runs/classification/seed_{SEED_TO_ANALYZE}/instances/tabular/label/bag_size_20/{POOLING_METHOD_TO_ANALYZE}/',
    "raw_data_path": 'data/oulad/oulad_aggregated_raw.pkl',
    "output_dir": OUTPUT_DIR,
    "models": {
        """ "Baseline": {
            "score_column": "shap_value",
            "model_to_explain_suffix": 'neg_policy_only_loss_epsilon_greedy_reg_sum_sample_without_replacement/'
        }, """
        "Hypernetwork Architecture": {
            "score_column": "shap_value",
            "is_hypernet": True,
            "model_to_explain_suffix": 'neg_policy_only_loss_pareto_hypernet_epsilon_greedy_reg_sum_sample_without_replacement/',
        }
    }
}
os.makedirs(DATASET_CONFIGS['output_dir'], exist_ok=True)

# Helper functions
def generate_general_labels(raw_bag_data):
    labels = []
    for instance_list in raw_bag_data:
        main_tuple = instance_list[0]
        if main_tuple[0] == 'assessment_type':
            labels.append(f"Assessment: {main_tuple[1]}")
        elif main_tuple[0] == 'activity_type':
            labels.append(f"VLE Clicks: {main_tuple[1]}")
        else:
            labels.append(main_tuple[0].replace('_', ' ').title())
    return labels

def build_bag_to_labels_map(raw_data_path):
    print("\nBuilding global map from Bag ID to Instance Labels")
    try:
        with open(raw_data_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"FATAL: Raw data file not found at {raw_data_path}")
        return {}

    # Debug check
    if not all(k in data for k in ("bag_ids", "raw_bags")):
        print(f"Unexpected pickle format. Keys found: {list(data.keys())}")
        return {}

    bag_to_labels = {}
    for bag_id, raw_bag in zip(data['bag_ids'], data['raw_bags']):
        bag_to_labels[str(bag_id)] = generate_general_labels(raw_bag)

    print(f"Built map for {len(bag_to_labels)} bags.")
    return bag_to_labels


def print_counts_table(model_name, counts_dict, bag_presence_dict, total_bags):
    print(f"\n--- Top-{TOP_K_FOR_COUNTS} Feature Counts for: {model_name.upper()} ---")
    if not counts_dict:
        print("No instances were found in the top 20.")
        return

    df = pd.DataFrame(counts_dict.items(), columns=['Feature Type', 'Total Count in Top-20'])
    df['Present in X Bags'] = df['Feature Type'].map(bag_presence_dict)
    df['Bag Presence %'] = df['Present in X Bags'] / total_bags * 100
    df['Avg Count When Present'] = df['Total Count in Top-20'] / df['Present in X Bags']
    df = df.sort_values(by='Total Count in Top-20', ascending=False).reset_index(drop=True)
    print(f"Based on {total_bags} common bags.")
    print(df)


def calculate_hhi(counts_dict, total_bags):
    total_top_k_instances = total_bags * TOP_K_FOR_COUNTS
    if total_top_k_instances == 0: return 0
    frequencies = np.array(list(counts_dict.values())) / total_top_k_instances
    return np.sum(frequencies**2)


def calculate_gini(counts_dict):
    counts = np.array(list(counts_dict.values()), dtype=np.float64)
    if np.sum(counts) == 0: return 0
    sorted_counts = np.sort(counts)
    n = len(counts)
    cum_counts = np.cumsum(sorted_counts)
    return (n + 1 - 2 * np.sum(cum_counts) / cum_counts[-1]) / n


def get_counts_for_split(bags_list, df_source, score_column, bag_to_labels_map):
    counts = defaultdict(int)
    for bag_id in bags_list:
        if bag_id not in bag_to_labels_map: continue
        bag_labels = bag_to_labels_map[bag_id]
        bag_df = df_source[df_source['bag_id'] == bag_id].reset_index(drop=True)
        if len(bag_labels) != len(bag_df): continue
        top_indices = bag_df.nlargest(TOP_K_FOR_COUNTS, score_column).index
        for idx in top_indices:
            counts[bag_labels[idx]] += 1
    return counts

def load_rl_model(run_dir_path, load_best=True) -> NetworkContainer:
    print("Attempting to load RL model...")
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    model_weights_path = os.path.join(run_dir_path, 'sweep_best_model.pt' if load_best else 'model.pt')
    rl_config_path = os.path.join(run_dir_path, 'sweep_best_model_config.json')
    mil_config_path = os.path.join(run_dir_path, '..', 'best_model_config.json')
    mil_weights_path = os.path.join(run_dir_path, '..', 'best_model.pt')
    
    try:
        with open(mil_config_path) as f: mil_config = json.load(f)
        with open(rl_config_path) as f: rl_config = json.load(f)
    except FileNotFoundError as e:
        print(f"FATAL: A config file was not found: {e}"); return None

    task_model = create_mil_model_with_dict(mil_config)
    task_model.load_state_dict(torch.load(mil_weights_path, map_location=device))
    debiasing_model = create_debiasing_model(Namespace(**mil_config), run_dir_path, Logger("Discard"))
    
    net_container = HypernetRLMIL(
        task_model=task_model, debiasing_model=debiasing_model, state_dim=rl_config['state_dim'], hdim=rl_config['hdim'], hidden_dim=mil_config["hidden_dim"],
        learning_rate=rl_config['learning_rate'], device=device, task_type=rl_config['task_type'],
        min_clip=rl_config.get('min_clip'), max_clip=rl_config.get('max_clip'),
        sample_algorithm=rl_config.get('sample_algorithm'), no_autoencoder=rl_config.get('no_autoencoder_for_rl', False)
    )
    net_container.load_state_dict(torch.load(model_weights_path, map_location=device))
    net_container.policy.eval()
    print("RL model loaded successfully.")
    return net_container

def parse_feature_string(s):
    if isinstance(s, str): return np.fromstring(s.replace('[','').replace(']','').replace('\n',' '), sep=' ')
    if isinstance(s, np.ndarray): return s
    return np.array([])

def append_to_csv(df, filepath):
    file_exists = os.path.exists(filepath)
    df.to_csv(filepath, mode='a', header=(not file_exists), index=False)
    
    if not file_exists:
        print(f"Created new file and saved results to: {filepath}")
    else:
        print(f"Appended new results to: {filepath}")

if __name__ == "__main__":

    # Step 1: Load Bag-to-Label Maps

    bag_to_labels_maps = {}


    print(f"--- Loading label map for: OULAD_AGGREGATED ---")
    try:
        bag_to_labels_maps = build_bag_to_labels_map(DATASET_CONFIGS['raw_data_path'])
        print(f"Successfully loaded map with {len(bag_to_labels_maps)} bags.")
    except Exception as e:
        print(f"    ERROR: Could not load map for OULAD_AGGREGATED. Error: {e}")

    # Step 2: Calculate or Load SHAP Values

    all_dfs = {} 

    print("Starting SHAP Analysis")

    for model_name, model_config in DATASET_CONFIGS['models'].items():
        # Process only if the model is explicitly marked for SHAP analysis
        full_model_name = f"{model_name}"
        print(f"\n- Processing SHAP for: {full_model_name}")
        
        shap_filename = f"shap_scores_{POOLING_METHOD_TO_ANALYZE}_seed_{SEED_TO_ANALYZE}.csv"
        shap_output_file = os.path.join(DATASET_CONFIGS['output_dir'], shap_filename)
        
        try:
            if os.path.exists(shap_output_file):
                print(f"  Found pre-computed file. Loading from: {shap_output_file}")
                shap_df_loaded = pd.read_csv(shap_output_file, index_col=0)
                df = shap_df
            else:
                # SHAP CALCULATION LOGIC
                print(f"  No SHAP file found at {shap_output_file}.")
                print("  Starting new calculation (this may take a while)...")
                
                model_dir = os.path.join(DATASET_CONFIGS['base_path'], model_config['model_to_explain_suffix'])
                net_container: NetworkContainer = load_rl_model(model_dir) # Assumes load_rl_model is defined
                
                if not net_container:
                    print(f"  ERROR: Could not load model from {model_dir}. Skipping.")
                    continue

                net_container.policy.eval()
                all_instance_details = []

                with open(os.path.join(model_dir, "sweep_best_model_config.json"), "r") as f: rl_config = json.load(f)
                data_args = Namespace(
                    data_embedded_column_name="instances",
                    random_seed=SEED_TO_ANALYZE,
                    embedding_model="tabular",
                    label="label",
                    instance_labels_column=None,
                    further_extra_columns="bag_id",
                    task_type="classification",
                    balance_dataset=True,
                    batch_size=rl_config["batch_size"]
                )

                train_data, eval_data, test_data, _ = prepare_data(data_args, Logger("Discard"))
                _, _, test_loader = get_dataloaders(data_args, train_data, eval_data, test_data, Logger("Discard"))
                dataset_df = test_loader.dataset.original_dataframe

                with torch.no_grad():
                    for batch_idx, (batch_x_embeddings, batch_y_bag_labels, batch_original_indices, _) in enumerate(test_loader):
                        batch_original_indices_cpu_np = batch_original_indices.cpu().numpy() # Assuming batch_original_indices is CPU tensor or list
                        batch_y_bag_labels_cpu_np = batch_y_bag_labels.detach().cpu().numpy()

                        for i in range(batch_x_embeddings.shape[0]): # Iterate through bags in the batch
                            original_df_idx = batch_original_indices_cpu_np[i]

                            try:
                                bag_series = dataset_df.iloc[original_df_idx]
                            except IndexError:
                                print(f"Error: original_df_idx {original_df_idx} is out of bounds for dataset_df with shape {dataset_df.shape}. Skipping this bag.")
                                continue 

                            try:
                                bag_id_val = bag_series["bag_id"]
                            except KeyError:
                                print(f"KeyError: 'bag_id' not found in bag_series index for original_df_idx {original_df_idx}. "
                                            f"Available keys: {list(bag_series.index) if hasattr(bag_series, 'index') else 'N/A'}. Using placeholder.")
                                bag_id_val = f"error_idx_{original_df_idx}" 
                            
                            true_bag_label_val = batch_y_bag_labels_cpu_np[i]
                            
                            # Safely access 'bag' and 'bag_mask', providing defaults if bag_series is problematic
                            original_instances_in_bag = bag_series.get("bag", []) if isinstance(bag_series, pd.Series) else []
                            true_mask_for_bag = bag_series.get("bag_mask", []) if isinstance(bag_series, pd.Series) else []

                            for j in range(len(original_instances_in_bag)): 
                                is_padding = not true_mask_for_bag[j] if j < len(true_mask_for_bag) else True
                                
                                if is_padding or j >= len(original_instances_in_bag): continue
                                instance_content = original_instances_in_bag[j]

                                instance_data = {
                                    "bag_id": bag_id_val,
                                    "instance_index_in_bag": j,
                                    "true_bag_label": true_bag_label_val,
                                    "original_instance_content": instance_content,
                                }
                                all_instance_details.append(instance_data)
                
                df_details = pd.DataFrame(all_instance_details)
                instance_features = np.vstack(df_details['original_instance_content'].apply(parse_feature_string))
                
                def shap_wrapper_f(x):
                    with torch.no_grad():
                        return net_container.action(torch.from_numpy(x).float().to(torch.device("cuda:0")))[0].cpu().numpy()
                
                explainer = shap.KernelExplainer(shap_wrapper_f, shap.sample(instance_features, 50))

                if model_config["is_hypernet"]:
                    for preference in np.linspace(0, 1, 11):
                        shap_filename = f"shap_scores_{POOLING_METHOD_TO_ANALYZE}_seed_{SEED_TO_ANALYZE}_pref_{str(round(preference, 1)).replace(".", "-")}.csv"
                        shap_output_file = os.path.join(DATASET_CONFIGS['output_dir'], shap_filename)
                        if os.path.exists(shap_output_file): continue

                        net_container: HypernetRLMIL = net_container
                        net_container.set_preference(torch.fill(torch.zeros((1)), preference).to(torch.device("cuda:0")))

                        shap_values = explainer.shap_values(instance_features)
                        df_details["shap_value"] = np.abs(shap_values).mean(axis=1)
                        shap_df = df_details

                        shap_df.to_csv(shap_output_file)
                    shap_df = pd.read_csv(shap_output_file)
                else:
                    shap_values = explainer.shap_values(instance_features)
                    
                    shap_df = pd.DataFrame({'shap_value': np.abs(shap_values).mean(axis=1)}, index=df_details.index)
                    
                    print(f"  SHAP values calculated. Saving to: {shap_output_file}")
                    shap_df.to_csv(shap_output_file)
                
                df = shap_df

            all_dfs[full_model_name] = df

        except (FileNotFoundError, KeyError) as e:
            print(f"  WARNING: Could not process {full_model_name}. Error: {e}")

    print("\n SHAP analysis complete.")

    # Step 3: Find Common Bags

    if not all_dfs:
        print(f"No models loaded. Skipping.")
    else:
        common_bags = sorted(list(set.intersection(*(set(df['bag_id'].unique()) for df in all_dfs.values()))))
        print(f"Found {len(common_bags)} common bags for the {len(all_dfs)} models.")

    # Step 4: Generate Top-20 Counts

    all_counts = {name: defaultdict(int) for name in all_dfs.keys()}
    all_bag_presence = {name: defaultdict(int) for name in all_dfs.keys()}

    for model_name, df_source in all_dfs.items():
        print(f"Processing: {model_name}")
        
        dataset_name = "oulad_full" if "oulad_full" in model_name else "oulad_aggregated"
        correct_map = bag_to_labels_maps
        common_bags_for_model = common_bags
        
        base_model_name = model_name.replace(f" on {dataset_name}", "")
        score_col = DATASET_CONFIGS['models'][base_model_name]['score_column']

        grouped_df = df_source.groupby('bag_id')

        for bag_id in common_bags_for_model:
            if bag_id not in correct_map:
                continue
            
            try:
                bag_df = grouped_df.get_group(bag_id)
                bag_labels = correct_map[bag_id]

                if len(bag_labels) == len(bag_df):
                    unique_labels_in_top20 = set()
                    top_20 = bag_df.nlargest(TOP_K_FOR_COUNTS, score_col)
                    
                    for idx in top_20.index:
                        label = bag_labels[bag_df.index.get_loc(idx)]
                        all_counts[model_name][label] += 1
                        unique_labels_in_top20.add(label)
                        
                    for label in unique_labels_in_top20:
                        all_bag_presence[model_name][label] += 1
            except KeyError:
                continue

    # Step 5: Print Descriptive Statistics

    for model_name in all_dfs.keys():
        total_bags_for_dataset = len(common_bags)
        
        print_counts_table(
            model_name, 
            all_counts[model_name], 
            all_bag_presence[model_name], 
            total_bags_for_dataset
        )

    # Step 6: Reliance on Demographics

    reliance_results = []
    for model_name in all_dfs.keys():
        total_possible_instances = len(common_bags) * TOP_K_FOR_COUNTS
        demographic_instance_count = sum(
            all_counts[model_name].get(label, 0) for label in DEMOGRAPHIC_LABELS
        )
        if total_possible_instances > 0:
            reliance_percentage = (demographic_instance_count / total_possible_instances) * 100
        else:
            reliance_percentage = 0
            
        reliance_results.append({
            "Model": model_name,
            "Reliance on Demographics (%)": f"{reliance_percentage:.1f}%"
        })

    reliance_df = pd.DataFrame(reliance_results)

    print(reliance_df.sort_values(by="Model").reset_index(drop=True))

    # Step 7: Sparsity Metrics
    sparsity_results = []
    for model_name in all_dfs.keys():
        total_bags_for_dataset = len(common_bags)
        sparsity_results.append({
            "Model": model_name,
            "Gini Coefficient": calculate_gini(all_counts[model_name]),
            "HHI Score": calculate_hhi(all_counts[model_name], total_bags_for_dataset),
        })

    sparsity_df = pd.DataFrame(sparsity_results)

    print(sparsity_df.sort_values(by="Model").reset_index(drop=True))

    # Step 8: Consistency Metrics (Split-Half Method) 

    consistency_results = []
    np.random.seed(REPRODUCIBILITY_SEED)

    print(f"\n- Splitting bags")

    shuffled_bags = np.array(common_bags)
    np.random.shuffle(shuffled_bags)
    split_point = len(shuffled_bags) // 2
    bags_a, bags_b = shuffled_bags[:split_point], shuffled_bags[split_point:]

    correct_map_for_model = bag_to_labels_maps

    for model_name, df in all_dfs.items():
        print(f"  -> Calculating for model: {model_name}...")
        score_col = DATASET_CONFIGS['models'][model_name]['score_column']
        
        counts_a = get_counts_for_split(bags_a, df, score_col, correct_map_for_model)
        counts_b = get_counts_for_split(bags_b, df, score_col, correct_map_for_model)
        
        policy_df = pd.DataFrame({'split_A': counts_a, 'split_B': counts_b}).fillna(0)
        spearman_corr, _ = spearmanr(policy_df['split_A'], policy_df['split_B'])
        
        top5_a = set(policy_df.nlargest(5, 'split_A').index)
        top5_b = set(policy_df.nlargest(5, 'split_B').index)

        print(f"    - Top 5 (Split A): {sorted(list(top5_a))}")
        print(f"    - Top 5 (Split B): {sorted(list(top5_b))}")
        
        jaccard_k5_score = len(top5_a.intersection(top5_b)) / len(top5_a.union(top5_b)) if top5_a and top5_b else 0.0

        result_row = {"Model": model_name, "Spearman's Rank (Stability)": spearman_corr, "Jaccard (K=5)": jaccard_k5_score}
        consistency_results.append(result_row)

    consistency_df = pd.DataFrame(consistency_results)
    print(consistency_df.sort_values(by="Model").reset_index(drop=True))

    # Step 9: save results to csv's
    SPARSITY_FILE = os.path.join(OUTPUT_DIR, 'master_sparsity_metrics.csv')
    CONSISTENCY_FILE = os.path.join(OUTPUT_DIR, 'master_consistency_metrics.csv')
    DEMOGRAPHICS_FILE = os.path.join(OUTPUT_DIR, 'master_demographics_metrics.csv')

    sparsity_df['seed'] = SEED_TO_ANALYZE
    sparsity_df['pooling_method'] = POOLING_METHOD_TO_ANALYZE

    consistency_df['seed'] = SEED_TO_ANALYZE
    consistency_df['pooling_method'] = POOLING_METHOD_TO_ANALYZE

    reliance_df['seed'] = SEED_TO_ANALYZE
    reliance_df['pooling_method'] = POOLING_METHOD_TO_ANALYZE

    try:
        append_to_csv(sparsity_df, SPARSITY_FILE)
        append_to_csv(consistency_df, CONSISTENCY_FILE)
        append_to_csv(reliance_df, DEMOGRAPHICS_FILE)
    except NameError as e:
        print(f"ERROR: A results DataFrame is not defined. Please ensure the notebook has run successfully. Details: {e}")