import subprocess
import argparse
import os
import numpy as np
from sklearn.model_selection import ParameterGrid

def run_tuning():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='oulad_aggregated')
    parser.add_argument('--autoencoder_layer_sizes', type=str, default='20,16,20')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=5, help="Matches run_cap")
    args = parser.parse_args()

    np.random.seed(args.random_seed)

    # 1. Define the Static Grid (Categorical/Fixed from your Shell Script)
    static_grid = {
        'baseline': ["MeanMLP", "MaxMLP", "AttentionMLP", "repset"],
        'label': ["label"],
        'bag_size': [20],
        'embedding_model': ["tabular"],
        'task_type': ["classification"],
        'data_embedded_column_name': ["instances"]
    }

    # 2. Define the Hyperparameter Distributions (from your YAML)
    # Constants
    dropout_p = 0.5
    scheduler_patience = 5
    early_stopping = 10
    
    # Categorical/Continuous choices
    batch_sizes = [8, 16, 32, 64]
    hidden_dims = [32, 64, 128, 256, 512]

    combinations = list(ParameterGrid(static_grid))
    
    for combo in combinations:
        print(f"\n=== Starting Tuning for Baseline: {combo['baseline']} ===")
        
        for i in range(1, args.num_samples + 1):
            # Sample from YAML distributions
            current_batch_size = int(np.random.choice(batch_sizes))
            current_hidden_dim = int(np.random.choice(hidden_dims))
            current_epochs = int(np.random.randint(50, 401))
            # Log uniform for Learning Rate: 10^x between 1e-4 and 1e-2
            current_lr = 10**np.random.uniform(-4, -2)

            # Construct the command
            cmd = [
                "python", "run_mil.py",
                "--baseline", combo['baseline'],
                "--label", combo['label'],
                "--bag_size", str(combo['bag_size']),
                "--embedding_model", combo['embedding_model'],
                "--autoencoder_layer_sizes", args.autoencoder_layer_sizes,
                "--data_embedded_column_name", combo['data_embedded_column_name'],
                "--task_type", combo['task_type'],
                "--random_seed", str(args.random_seed),
                "--no_wandb",
                # Hyperparameters sampled from YAML
                "--batch_size", str(current_batch_size),
                "--hidden_dim", str(current_hidden_dim),
                "--epochs", str(current_epochs),
                "--learning_rate", f"{current_lr:.6f}",
                "--dropout_p", str(dropout_p),
                "--scheduler_patience", str(scheduler_patience),
                "--early_stopping_patience", str(early_stopping),
            ]

            print(f"Run {i}/{args.num_samples} | LR: {current_lr:.5f} | HDim: {current_hidden_dim} | BS: {current_batch_size}")

            # Set environment variables
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

            # Execute
            subprocess.run(cmd, env=env)

if __name__ == "__main__":
    run_tuning()