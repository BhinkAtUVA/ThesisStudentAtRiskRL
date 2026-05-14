# This file uses the csv created by gather_results.py to create a summary of all results. 
# It avarages the F1-score of all model architectures over 10 seeds and calculates the standard deviations. 
# It also performs statistical tests to prove significance based on the sub-questions of the thesis.
import pandas as pd
from scipy.stats import ttest_ind
from itertools import combinations
import os
    
def run_statistical_tests(df):
    print("\n\n Running Statistical Tests")
    print("Comparing every framework to every other framework.")
    
    all_frameworks = df['Framework'].unique()
    datasets = df['Dataset'].unique()
    pooling_methods = df['Pooling'].unique()

    model_pairs = list(combinations(all_frameworks, 2))
    
    if not model_pairs:
        print("Not enough different frameworks to compare.")
        return

    results = []

    # Loop through each dataset, pooling method, and model pair
    for dataset in datasets:
        for pooling in pooling_methods:
            row = {'Dataset': dataset, 'Pooling': pooling}
            for model1, model2 in model_pairs:
                # Get the F1 scores for each model under the current conditions
                scores1 = df[(df['Framework'] == model1) & (df['Dataset'] == dataset) & (df['Pooling'] == pooling)]['F1_Score']
                scores2 = df[(df['Framework'] == model2) & (df['Dataset'] == dataset) & (df['Pooling'] == pooling)]['F1_Score']
                
                col_name = f"{model1} vs. {model2}"
                
                # Perform Welch's t-test if there's enough data for both models
                if len(scores1) > 1 and len(scores2) > 1:
                    _, p_value = ttest_ind(scores1, scores2, equal_var=False)
                    # Add significance star for quick reference
                    significance = ' *' if p_value < 0.05 else ''
                    row[col_name] = f"{p_value:.4f}{significance}"
                else:
                    row[col_name] = "N/A"
            
            results.append(row)
    
    # Create, print, and save the final summary DataFrame
    if results:
        summary_df = pd.DataFrame(results).set_index(['Dataset', 'Pooling'])
        print("\nP-Values for All Framework Comparisons (p < 0.05 marked with *):")
        print(summary_df)
        
        output_path = 'results/p_values_comparison.csv'
        summary_df.to_csv(output_path)
        print(f"\nComparison saved to '{output_path}'")
    else:
        print("No results were generated.")


def create_summary_table(df):
    print("\nGenerating Summary Table (Mean & Std Dev)")
    grouping_cols = ['Framework', 'Pooling', 'Dataset']
    summary = df.groupby(grouping_cols)['F1_Score'].agg(['mean', 'std']).round(4)
    return summary

if __name__ == "__main__":
    INPUT_FILE = 'results/final_thesis_results.csv'
    SUMMARY_TABLE_OUTPUT_FILE = 'results/performance_summary_table.csv'

    print(f"Loading data from '{INPUT_FILE}'...")
    try:
        main_df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"\nERROR: Input file not found at '{INPUT_FILE}'.")
        print("Please run the gather_results.py script first to generate this file.")
        exit()
    
    # Run Analysis 
    if not main_df.empty:
        summary_table = create_summary_table(main_df)
        os.makedirs('results', exist_ok=True)
        summary_table.to_csv(SUMMARY_TABLE_OUTPUT_FILE)
        print("\n Performance Summary Table:")
        print(summary_table)
        print(f"\n Summary table saved to '{SUMMARY_TABLE_OUTPUT_FILE}'")

        # Run and print the statistical tests
        run_statistical_tests(main_df)
        print("\nAnalysis complete.")
    else:
        print("Analysis finished with no data to process.")
