#!/bin/bash
#SBATCH --partition= #YOUR PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=20:00:00
#SBATCH --output=../logs/DATASET_NAME/rlmil/seed_0/%j.out
#SBATCH --error=../logs/DATASET_NAME/rlmil/seed_0/%j.err

#module purge
#module load 2023
cd ~/StudiumDS/Sem2/Thesis/ThesisStudentAtRiskRL # ROOT OF YOUR PROJECT
source venv/bin/activate

baseline_types=("MeanMLP" "MaxMLP" "AttentionMLP" "repset")
target_labels=("label")
gpus=(0)
wandb_entity="BhinkAtUVA"
wandb_project="Thesis"

data_embedded_column_name="instances"
task_type="classification"
autoencoder_layer_sizes="22,16,22"  	        # "22,16,22" for oulad_aggregated and "20,16,20" for oulad_full
bag_sizes=(20)                                # for all experiments in this project bag_size 20 is used
embedding_models=("tabular")
random_seed=0

rl_task_model="vanilla"
sample_algorithm="without_replacement"
prefix="loss"
rl_model="policy_only"
search_algorithm="epsilon_greedy"
reg_alg="sum"

total_runs=$((${#baseline_types[@]} * ${#target_labels[@]} * ${#bag_sizes[@]} * ${#embedding_models[@]}))
current_run=1

for target_label in "${target_labels[@]}"; do
  for bag_size in "${bag_sizes[@]}"; do
    for embedding_model in "${embedding_models[@]}"; do
      for baseline_type in "${baseline_types[@]}"; do
        gpu=${gpus[$target_label_index]}
        echo "$baseline_type, $dataset $target_label, bag_size_$bag_size, $embedding_model, gpu_$gpu ($current_run/$total_runs)"

        CUDA_VISIBLE_DEVICES=$gpu python run_trainer.py --rl --baseline $baseline_type \
                                            --autoencoder_layer_sizes $autoencoder_layer_sizes \
                                            --label $target_label \
                                            --data_embedded_column_name $data_embedded_column_name \
                                            --prefix $prefix \
                                            --dataset $dataset \
                                            --bag_size $bag_size \
                                            --embedding_model $embedding_model \
                                            --train_pool_size 1 --eval_pool_size 10 --test_pool_size 10 \
                                            --balance_dataset \
                                            --wandb_entity $wandb_entity \
                                            --wandb_project $wandb_project \
                                            --random_seed $random_seed \
                                            --task_type $task_type \
                                            --rl_model $rl_model \
                                            --search_algorithm $search_algorithm \
                                            --rl_task_model $rl_task_model \
                                            --sample_algorithm $sample_algorithm \
                                            --reg_alg $reg_alg \
                                            --run_sweep ;
        ((current_run++))
      done
    done
  done
done
