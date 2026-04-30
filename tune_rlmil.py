import subprocess
import argparse
import os
import numpy as np

def run_tuning():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='oulad_aggregated')
    parser.add_argument('--autoencoder_layer_sizes', type=str, default='20,16,20')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=5, help="Number of random search iterations")
    args = parser.parse_args()

    np.random.seed(args.random_seed)

    # Values derived from your hp_rl_policy_only_loss_attention_gated_reg_sum.yaml
    for i in range(1, args.num_samples + 1):
        # Sample hyperparameters based on the YAML distributions
        # actor_lr: log_uniform 1e-6 to 1e-2
        actor_lr = 10**np.random.uniform(-6, -2)
        # reg_coef: uniform 0.0 to 1.0
        reg_coef = np.random.uniform(0.0, 1.0)
        # gated_temperature: uniform 0.0 to 10.0
        gated_temp = np.random.uniform(0.0, 10.0)
        # gated_attention_size: categorical [16, 32, 64, 128]
        attn_size = int(np.random.choice([16, 32, 64, 128]))
        # gated_attention_dropout_p: uniform 0.0 to 0.5
        attn_dropout = np.random.uniform(0.0, 0.5)

        # Fixed parameters from YAML
        hdim = 8
        epochs = 200
        critic_lr = 0
        learning_rate = 1e-6
        early_stopping = 25
        batch_size = 128

        # Prefix logic to stay consistent with configs.py
        # Based on: hp_rl_policy_only_loss_attention_gated_reg_sum
        # We'll use "loss_attention_gated" as the middle part
        raw_prefix = "loss_attention_gated_reg_sum"
        final_prefix = f"neg_policy_only_{raw_prefix}_sample_without_replacement"

        print(f"--- Run {i}/{args.num_samples} | actor_lr: {actor_lr:.6f} | attn_size: {attn_size} ---")

        cmd = [
            "python", "run_rlmil.py",
            "--rl",
            "--baseline", "AttentionMLP", # Based on the config name
            "--autoencoder_layer_sizes", args.autoencoder_layer_sizes,
            "--label", "label",
            "--data_embedded_column_name", "instances",
            "--prefix", final_prefix,
            "--bag_size", "20",
            "--embedding_model", "tabular",
            "--random_seed", str(args.random_seed),
            "--rl_model", "policy_only",
            "--rl_task_model", "vanilla",
            "--sample_algorithm", "without_replacement",
            "--reg_alg", "sum",
            "--search_algorithm", "epsilon_greedy",
            "--no_wandb",
            "--task_type", "classification",
            # Hyperparameters sampled from your YAML:
            "--actor_learning_rate", str(actor_lr),
            "--critic_learning_rate", str(critic_lr),
            "--learning_rate", str(learning_rate),
            "--reg_coef", str(reg_coef),
            "--hdim", str(hdim),
            "--epochs", str(epochs),
            "--early_stopping_patience", str(early_stopping),
            "--batch_size", str(batch_size),
            "--gated_temperature", str(gated_temp),
            "--gated_attention_size", str(attn_size),
            "--gated_attention_dropout_p", str(attn_dropout)
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        
        subprocess.run(cmd, env=env)

if __name__ == "__main__":
    run_tuning()