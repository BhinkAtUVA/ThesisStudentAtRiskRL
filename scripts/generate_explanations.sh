#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=20:00:00
#SBATCH --output=../logs/DATASET_NAME/rlmil/seed_0/%j.out
#SBATCH --error=../logs/DATASET_NAME/rlmil/seed_0/%j.err

source venv/bin/activate
baseline_types=("MeanMLP" "MaxMLP" "AttentionMLP" "repset") # "MeanMLP" "MaxMLP" "AttentionMLP" "repset"

total_runs=$((${#baseline_types[@]} * ${#rl_variants[@]} * ${#target_labels[@]} * ${#bag_sizes[@]} * ${#embedding_models[@]} * 5))
for baseline_type in "${baseline_types[@]}"; do
    python -m src.results.metrics $baseline_type
done