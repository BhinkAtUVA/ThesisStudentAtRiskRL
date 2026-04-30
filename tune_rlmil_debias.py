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

    for i in range(1, args.num_samples + 1):
        # Sampled hyperparameters based on YAML config
        actor_lr = 10**np.random.uniform(-6, -2)
        reg_coef = np.random.uniform(0.0, 1.0)
        gated_temp = np.random.uniform(0.0, 10.0)
        attn_size = int(np.random.choice([16, 32, 64, 128]))
        attn_dropout = np.random.uniform(0.0, 0.5)

        # YAML config translation
        hdim = 8
        epochs = 200
        critic_lr = 0
        learning_rate = 1e-6
        early_stopping = 25
        batch_size = 128

        print(f"--- Run {i}/{args.num_samples} | actor_lr: {actor_lr:.6f} | attn_size: {attn_size} ---")

        cmd = [
            "python", "run_rlmil_debias.py",
            "--rl",
            "--baseline", "MeanMLP", # Based on the config name
            "--label", "label",
            "--data_embedded_column_name", "instances",
            "--prefix", "DEBIASTEST",
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
            "--train_pool_size", "1",
            "--eval_pool_size", "10",
            "--test_pool_size", "10",
            "--epsilon", "0",
            # YAML config translation:
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