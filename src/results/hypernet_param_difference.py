import os

import numpy as np
import torch

from src.models.full import HypernetRL
from src.models.hypernet import pack_weights_policy
from src.results.metrics import DATASET_CONFIGS, load_rl_model


device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

for model_name, model_config in DATASET_CONFIGS['models'].items():
    if not model_config["is_hypernet"]: continue

    model_dir = os.path.join(DATASET_CONFIGS['base_path'], model_config['model_to_explain_suffix'])
    net_container: HypernetRL = load_rl_model(model_dir) # Assumes load_rl_model is defined
    net_container.to(device)

    hyper_weights = net_container.hyper(net_container.preference)
    zero_weights: dict[torch.Tensor] = pack_weights_policy(hyper_weights, net_container.policy_weights, 0.05, net_container.state_dim, net_container.hdim)
    
    net_container.set_preference(torch.fill(torch.zeros((1)), 1).to(device))
    hyper_weights = net_container.hyper(net_container.preference)
    one_weights: dict[torch.Tensor] = pack_weights_policy(hyper_weights, net_container.policy_weights, 0.05, net_container.state_dim, net_container.hdim)

    differences = []
    for name, values in zero_weights.items():
        diff: torch.Tensor = (values / one_weights[name])
        diff_series = diff.reshape(np.prod(diff.shape)).cpu().detach().numpy()
        diff_series[diff_series < 1] = 1 / diff_series[diff_series < 1]
        differences.extend(diff_series.tolist())

    print(np.median(differences))

    