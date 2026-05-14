from argparse import Namespace
import json
import os, sys
from logging import Logger
from pathlib import Path
import torch

from src.models.full import HypernetRLMIL
from src.trainers.util import create_net_container

logger = Logger("Investigation")
script_folder = Path(os.path.realpath(__file__)).parent
root_path = script_folder.parent.parent

MODEL_DIRECTORY = root_path / "runs" / "classification" / "seed_0" / "instances" / "tabular" / "label" / "bag_size_20" / "MeanMLP" / "neg_policy_only_loss_epsilon_greedy_reg_sum_sample_without_replacement"

with open(MODEL_DIRECTORY / "sweep_best_model_config.json") as f: hyper_config = json.load(f)
model: HypernetRLMIL = create_net_container(Namespace(
    rl_task_model=hyper_config["rl_task_model"],
    state_dim=hyper_config["state_dim"],
    hdim=hyper_config["hdim"],
    learning_rate=hyper_config["learning_rate"],
    device=torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu"),
    task_type=hyper_config["task_type"],
    min_clip=hyper_config["min_clip"],
    max_clip=hyper_config["max_clip"],
    sample_algorithm=hyper_config["sample_algorithm"],
    no_autoencoder_for_rl=hyper_config["no_autoencoder_for_rl"]
), MODEL_DIRECTORY, HypernetRLMIL, logger)
print(model.hyper.state_dict())
print(model.policy_weights)
print(model.task_model.state_dict())
print(model.debiasing_model.state_dict())