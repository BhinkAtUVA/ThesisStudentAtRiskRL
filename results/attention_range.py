# This file gives insight about the attentionscore ranges and compares them over the different models 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

LAYERS = {
    'oulad_aggregated': '22_16_22',
    'oulad_full': '20_16_20',
}
# Put your own experiments here
RL_MODEL_CONFIGS = [
    {
        "model_type": "RL-MIL (Linear Attention)",
        "folder_name": "neg_policy_only_loss_attention_gated_reg_sum_sample_without_replacement",
        "output_file": "attention_gated_outputs.csv",
        "score_column": "attention_score"
    },
    {
        "model_type": "RL-MIL (Gated Attention)",
        "folder_name": "neg_policy_only_loss_attention_gated_reg_sum_sample_without_replacement",
        "output_file": "attention_gated_outputs.csv",
        "score_column": "attention_score"
    },
    {
        "model_type": "RL-MIL (Multi-Head Attention)",
        "folder_name": "neg_policy_only_loss_attention_multi_head_reg_sum_sample_without_replacement",
        "output_file": "attention_multi_head_outputs.csv",
        "score_column": "attention_score"
    }
]

OUTPUT_DIR = '.' 
SUMMARY_CSV_FILE = 'attention_score_summary.csv'
CHART_PNG_FILE = 'attention_score_comparison.png'


# Part 1: Analysis Function 
def analyze_attention_scores():
    """
    Analyzes the range of attention scores for each model across all seeds, datasets, and pooling methods.
    Returns a DataFrame with a single row of average statistics for each model/dataset combo.
    """
    all_individual_results = []
    datasets_to_analyze = ['oulad_aggregated', 'oulad_full']
    pooling_methods = ['MeanMLP', 'MaxMLP', 'AttentionMLP', 'repset']
    
    for dataset in datasets_to_analyze:
        for pooling_method in pooling_methods:
            for model_config in RL_MODEL_CONFIGS:
                model_type = model_config["model_type"]
                folder_name = model_config["folder_name"]
                output_file = model_config["output_file"]
                score_column = model_config["score_column"]
                
                print(f"Processing {model_type} on {dataset} with {pooling_method}...")
                
                path_pattern = f'/projects/prjs1491/Attention-based-RL-MIL/runs/classification/seed_*/{dataset}/instances/tabular/label/bag_size_20/{pooling_method}_{LAYERS[dataset]}/{folder_name}/'
                seed_dirs = glob.glob(path_pattern.replace('seed_*', 'seed_*/'))
                
                for seed_dir in seed_dirs:
                    file_path = os.path.join(seed_dir, output_file)
                    
                    if not os.path.exists(file_path):
                        continue
                    
                    match = re.search(r'seed_(\d+)', file_path)
                    seed = int(match.group(1)) if match else 'N/A'
                    
                    try:
                        df = pd.read_csv(file_path)
                        df_real = df[df['is_padding_instance'] == False].copy()
                        
                        if df_real.empty:
                            continue
                        
                        scores = df_real[score_column]
                        
                        stats = {
                            "Model": model_type,
                            "Dataset": dataset,
                            "Pooling Method": pooling_method,
                            "Seed": seed,
                            "Min_Score": scores.min(),
                            "Max_Score": scores.max(),
                            "Mean_Score": scores.mean(),
                            "Median_Score": scores.median(),
                        }
                        all_individual_results.append(stats)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        continue
    
    if not all_individual_results:
        return pd.DataFrame()

    individual_df = pd.DataFrame(all_individual_results)
    
    summary_df = individual_df.groupby(['Model', 'Dataset']).agg({
        'Min_Score': 'mean',
        'Max_Score': 'mean',
        'Mean_Score': 'mean',
        'Median_Score': 'mean'
    }).reset_index().round(4)
    
    return summary_df


# Part 2: Charting Function
def create_attention_range_chart(data, output_path):
    """
    Creates a grouped bar chart visualizing the min, max, and mean attention scores
    for each dataset and saves the plot to a file.
    """
    datasets = data['Dataset'].unique()
    models = data['Model'].unique()
    
    x = np.arange(len(models))
    width = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        df_dataset = data[data['Dataset'] == dataset].set_index('Model')
        df_dataset = df_dataset.reindex(models)
        
        ax.bar(x - width, df_dataset['Min_Score'], width, label='Min Score', color=colors[0])
        ax.bar(x, df_dataset['Mean_Score'], width, label='Mean Score', color=colors[1])
        ax.bar(x + width, df_dataset['Max_Score'], width, label='Max Score', color=colors[2])
        
        ax.set_ylabel('Attention Score', fontsize=12)
        ax.set_title(f'Attention Score Range on {dataset}', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([m.split('(')[1].replace(')','').strip() for m in models], rotation=0, ha='center', fontsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.legend(fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.savefig(output_path)
    print(f"\nChart saved to '{output_path}'")
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Run the analysis
    summary_df = analyze_attention_scores()
    
    # Step 2: Check for results and proceed
    if not summary_df.empty:
        csv_output_path = os.path.join(OUTPUT_DIR, SUMMARY_CSV_FILE)
        summary_df.to_csv(csv_output_path, index=False)
        print(f"\nAnalysis complete. Results saved to '{csv_output_path}'")
        print("\nFinal Aggregated Summary:")
        print(summary_df.to_string())

        # Step 3: Create the chart from the analysis results
        chart_output_path = os.path.join(OUTPUT_DIR, CHART_PNG_FILE)
        create_attention_range_chart(summary_df, chart_output_path)
    else:
        print("\nNo data was processed. Please check the file paths and your directories.")