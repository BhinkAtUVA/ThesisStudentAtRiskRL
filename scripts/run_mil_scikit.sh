#!/bin/bash
#SBATCH --partition= #YOUR PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=../logs/oulad_aggregated/mil/seed_0/%j.out
#SBATCH --error=../logs/oulad_aggregated/mil/seed_0/%j.err

cd ~/StudiumDS/Sem2/Thesis/ThesisStudentAtRiskRL
source venv/bin/activate

python tune_mil.py \
    --random_seed 0 \
    --gpu 0 \
    --num_samples 5