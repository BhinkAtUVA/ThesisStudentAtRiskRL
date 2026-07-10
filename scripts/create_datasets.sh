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

seeds=4

for i in $(seq 0 $seeds); do
    python src/prepare_oulad_data.py $i
done