from argparse import Namespace
import json
from logging import Logger
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.models.full import HypernetRLMIL
from src.models.rl import sample_action_without_replacement, select_from_action
from src.results.metrics import DATASET_CONFIGS, POOLING_METHOD_TO_ANALYZE, SEED_TO_ANALYZE, load_rl_model
from src.trainers.util import get_dataloaders, prepare_data

loss_fn = nn.CrossEntropyLoss()
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

for model_name, model_config in DATASET_CONFIGS['models'].items():
    if not model_config["is_hypernet"]: continue

    model_dir = os.path.join(DATASET_CONFIGS['base_path'], model_config['model_to_explain_suffix'])
    net_container: HypernetRLMIL = load_rl_model(model_dir, load_best=True) # Assumes load_rl_model is defined
    net_container.to(device)
    
    with open(os.path.join(model_dir, "sweep_best_model_config.json"), "r") as f: rl_config = json.load(f)
    data_args = Namespace(
        data_embedded_column_name="instances",
        random_seed=SEED_TO_ANALYZE,
        embedding_model="tabular",
        label="label",
        instance_labels_column=None,
        further_extra_columns="bag_id",
        task_type="classification",
        balance_dataset=True,
        batch_size=rl_config["batch_size"]
    )

    train_data, eval_data, test_data, _ = prepare_data(data_args, Logger("Discard"))
    _, _, test_loader = get_dataloaders(data_args, train_data, eval_data, test_data, Logger("Discard"))

    with torch.no_grad():
        losses_by_preference = []
        for preference in np.linspace(0, 1, 11):
            net_container.set_preference(torch.fill(torch.zeros((1)), preference).to(device))
            sel_losses, bias_losses = [], []
            for batch_idx, (batch_x_embeddings, batch_y_bag_labels, batch_original_indices, _) in enumerate(test_loader):
                action_probs, batch_rep, exp_reward = net_container.action(batch_x_embeddings.to(device))
                action, action_log_prob = sample_action_without_replacement(action_probs, 20, device)
                selected_bag = select_from_action(action, batch_x_embeddings.to(device))
                true_indices = (torch.max(batch_x_embeddings[:, (2, 4, 5, 7), :], dim=-1).values - 1).to(dtype=torch.int64, device=device)
                _, sel_loss, bias_loss = net_container.predict(loss_fn, selected_bag.to(device), batch_y_bag_labels.to(device), true_indices)
                sel_losses.append(sel_loss)
                bias_losses.append(bias_loss)
            losses_by_preference.append({
                "preference": round(preference, 1),
                "task_loss": np.mean(sel_losses),
                "bias_loss": np.mean(bias_losses)
            })
        df = pd.DataFrame(losses_by_preference)
        df.to_csv(f"results/pareto_front_{POOLING_METHOD_TO_ANALYZE}_seed_{SEED_TO_ANALYZE}.csv")