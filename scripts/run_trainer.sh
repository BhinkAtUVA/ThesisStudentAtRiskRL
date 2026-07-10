#!/bin/bash
#SBATCH --partition=gpu_a100
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

baseline_types=("repset") # "MeanMLP" "MaxMLP" "AttentionMLP" "repset"
rl_variants=("hypernet_rlmil") # "baseline" "hypernet_rl" "hypernet_rlmil"
target_labels=("label")
gpus=(0)
wandb_entity="BhinkAtUVA"
wandb_project="Thesis"

data_embedded_column_name="instances"
task_type="classification"
bag_sizes=(20)                                # for all experiments in this project bag_size 20 is used
embedding_models=("tabular")

rl_task_model="vanilla"
sample_algorithm="without_replacement"
rl_model="policy_only"
search_algorithm="epsilon_greedy"
reg_alg="sum"

total_runs=$((${#baseline_types[@]} * ${#rl_variants[@]} * ${#target_labels[@]} * ${#bag_sizes[@]} * ${#embedding_models[@]} * 5))
current_run=1

for target_label in "${target_labels[@]}"; do
  for bag_size in "${bag_sizes[@]}"; do
    for embedding_model in "${embedding_models[@]}"; do
      for baseline_type in "${baseline_types[@]}"; do
        for rl_variant in "${rl_variants[@]}"; do
          prefix="loss"
          if [ $rl_variant = "hypernet_rl" ]; then
            prefix="loss_pareto_hypernet"
          fi
          if [ $rl_variant = "hypernet_rlmil" ]; then
            prefix="loss_pareto_full_hypernet"
          fi
          gpu=${gpus[$target_label_index]}

          for seed in $(seq 0 4); do
            echo "$baseline_type $rl_variant $target_label, bag_size_$bag_size, $embedding_model, seed_$seed, gpu_$gpu ($current_run/$total_runs)"
            CUDA_VISIBLE_DEVICES=$gpu python -m src.run_trainer --rl --baseline $baseline_type \
                                                --rl_variant $rl_variant \
                                                --label $target_label \
                                                --data_embedded_column_name $data_embedded_column_name \
                                                --prefix $prefix \
                                                --bag_size $bag_size \
                                                --embedding_model $embedding_model \
                                                --train_pool_size 1 --eval_pool_size 2 --test_pool_size 2 \
                                                --balance_dataset \
                                                --wandb_entity $wandb_entity \
                                                --wandb_project $wandb_project \
                                                --random_seed $seed \
                                                --task_type $task_type \
                                                --rl_model $rl_model \
                                                --search_algorithm $search_algorithm \
                                                --rl_task_model $rl_task_model \
                                                --sample_algorithm $sample_algorithm \
                                                --reg_alg $reg_alg;
                                                #--run_sweep ;
            ((current_run++))
          done
        done
      done
    done
  done
done
