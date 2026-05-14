#!/bin/bash
#SBATCH --partition= #YOUR PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=../logs/DATASET_NAME/mil/seed_0/%j.out
#SBATCH --error=../logs/DATASET_NAME/mil/seed_0/%j.err

# module purge
# module load 2023
cd ~/StudiumDS/Sem2/Thesis/ThesisStudentAtRiskRL # ROOT OF YOUR PROJECT
source venv/bin/activate

baseline_types=("AttentionMLP" "repset") # "MeanMLP" "MaxMLP" "AttentionMLP" "repset"
target_labels=("label")
gpus=(0)
wandb_entity="BhinkAtUVA"
wandb_project="Thesis"

data_embedded_column_name="instances"
task_type="classification"
# "22,16,22" for oulad_aggregated and "20,16,20" for oulad_full
bag_sizes=(20)                                    # for all experiments in this project bag_size 20 is used
embedding_models=("tabular")

total_runs=$((${#baseline_types[@]} * ${#target_labels[@]} * ${#bag_sizes[@]} * ${#embedding_models[@]}))
current_run=1

for target_label_index in "${!target_labels[@]}"; do
  for bag_size_index in "${!bag_sizes[@]}"; do
    for embedding_model_index in "${!embedding_models[@]}"; do
      for baseline_type_index in "${!baseline_types[@]}"; do
        target_label=${target_labels[$target_label_index]}
        bag_size=${bag_sizes[$bag_size_index]}
        embedding_model=${embedding_models[$embedding_model_index]}
        baseline_type=${baseline_types[$baseline_type_index]}
        gpu=${gpus[$target_label_index]}
        echo "$baseline_type, $dataset $target_label, bag_size_$bag_size, $embedding_model, gpu_$gpu ($current_run/$total_runs)"

        CUDA_VISIBLE_DEVICES=$gpu python run_mil.py \
                                      --baseline "$baseline_type" \
                                      --label "$target_label" \
                                      --bag_size "$bag_size" \
                                      --embedding_model $embedding_model \
                                      --wandb_entity $wandb_entity \
                                      --wandb_project $wandb_project \
                                      --data_embedded_column_name $data_embedded_column_name \
                                      --task_type $task_type \
                                      --random_seed 0 \
                                      --run_sweep ;
        
        ((current_run++))
      done
    done
  done
done
