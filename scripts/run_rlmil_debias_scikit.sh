#!/bin/bash
#SBATCH --partition= #YOUR PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=20:00:00
#SBATCH --output=../logs/oulad_aggregated/rlmil/seed_0/%j.out
#SBATCH --error=../logs/oulad_aggregated/rlmil/seed_0/%j.err

# module purge
# module load 2023
cd ~/StudiumDS/Sem2/Thesis/ThesisStudentAtRiskRL
source venv/bin/activate

python tune_rlmil_debias.py \
    --random_seed 0 \
    --gpu 0 \
    --num_samples 1