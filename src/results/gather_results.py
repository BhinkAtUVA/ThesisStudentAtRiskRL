import pandas as pd
import os
import json
from pathlib import Path

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
    seeds = range(1, 11)
    datasets_config = {
        'oulad_aggregated': '22_16_22',
        'oulad_full': '20_16_20',
    }
    pooling_methods = ['MeanMLP', 'MaxMLP', 'AttentionMLP', 'repset']

    RL_MODEL_CONFIGS = [
        {
            "model_type": "RL-MIL (Epsilon-Greedy)",
            "folder_name": "neg_policy_only_loss_epsilon_greedy_reg_sum_sample_without_replacement"
        },
        {
            "model_type": "RL-MIL (Gated Attention)",
            "folder_name": "neg_policy_only_loss_attention_gated_reg_sum_sample_without_replacement"
        },
        {
            "model_type": "RL-MIL (Multi-Head Attention)",
            "folder_name": "neg_policy_only_loss_attention_multi_head_reg_sum_sample_without_replacement"
        }
    ]

    # Data Gathering
    results_list = []    
    found_docs_count = {
        'Simple MIL': 0,
        'RL-MIL (Epsilon-Greedy)': 0,
        'RL-MIL (Gated Attention)': 0,
        'RL-MIL (Multi-Head Attention)': 0
    }
    for seed in seeds:
        for dataset_name, layers in datasets_config.items():
            for pooling_base in pooling_methods:
                pooling_folder_name = f"{pooling_base}_{layers}"
                
                base_model_path = (base_path / f'seed_{seed}' / dataset_name / 
                                   'instances/tabular/label/bag_size_20' / pooling_folder_name)

                if not base_model_path.is_dir():
                    continue

                def process_json_file(path, model_name):
                    if path.is_file():
                        with open(path, 'r') as f:
                            json_content = json.load(f)
                        
                        f1_score = json_content.get('test/f1')
                        avg_f1_score = json_content.get('test/avg-f1')
                        
                        results_list.append({
                            'Framework': model_name,
                            'Pooling': pooling_base,
                            'Dataset': dataset_name,
                            'Seed': seed,
                            'F1_Score': f1_score,
                            'Avg_F1_Score': avg_f1_score 
                        })
                        found_docs_count[model_name] += 1
                # Gather Simple MIL baseline results
                process_json_file(base_model_path / 'results.json', 'Simple MIL')

                # Gather all RL model results
                for rl_config in RL_MODEL_CONFIGS:
                    rl_json_path = base_model_path / rl_config['folder_name'] / 'results.json'
                    process_json_file(rl_json_path, rl_config['model_type'])
                            
    return results_list, found_docs_count

if __name__ == "__main__":
    all_results_data, doc_counts = gather_all_results_data()

    if all_results_data:
        final_df = pd.DataFrame(all_results_data)
        final_df['Unified_F1_Score'] = final_df['Avg_F1_Score'].fillna(final_df['F1_Score'])
        final_df_clean = final_df[['Framework', 'Pooling', 'Dataset', 'Seed', 'Unified_F1_Score']].copy()
        final_df_clean = final_df_clean.rename(columns={'Unified_F1_Score': 'F1_Score'})
        final_df_clean.dropna(subset=['F1_Score'], inplace=True)
        output_filename = 'results/final_thesis_results.csv'
        os.makedirs('results', exist_ok=True)
        final_df_clean.to_csv(output_filename, index=False)
        print(final_df_clean.head())
        print("\nNumber of documents found per framework:")
        for framework, count in doc_counts.items():
            print(f"- {framework}: {count} documents")
    else:
        print("No result files were found.")
        