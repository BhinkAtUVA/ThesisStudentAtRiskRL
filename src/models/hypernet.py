from collections import OrderedDict
from typing import List, Tuple

import numpy as np
from torch import nn
import torch

from models.rl import PolicyNetwork

# Except for the applied Sigmoid at the end, this network is very similar to the SimpleMLP in atoms.py
class MILHypernetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_weights: int,
        dropout_p: float = 0.5,
    ):
        super(MILHypernetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_weights = num_weights
        self.dropout_p = dropout_p  # register the droupout probability as a buffer

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_dim, self.num_weights),
            nn.Sigmoid()
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)  # Apply the MLP
        return x

# Utility for typing tensors holding the actual parameters belonging to a network controlled by a hypernetwork
class MILParamStorage(torch.Tensor):
    pass
    
# Calculates the amount of weights for a given PolicyNetwork configuration
def get_num_weights(state_dim: int, hdim: int):
    return 8514 + 256 * state_dim + hdim * state_dim + hdim + hdim

# Utility for correctly initializing a 1D-Tensor representing parameters of a PolicyNetwork with the given Configuration
def init_policy_storage(state_dim: int, hdim: int) -> MILParamStorage:
    tensors = []
    for shape in [
        (256, state_dim),
        (256),
        (32, 256),
        (32),
        (1, 32),
        (1),
        (hdim, state_dim),
        (hdim),
        (1, hdim),
        (1)
    ]:
        if type(shape) != tuple or len(shape) == 1: tensors.append(torch.fill(torch.zeros(shape), 0.01))
        else: tensors.append(torch.nn.init.xavier_uniform_(torch.zeros(shape)).reshape((np.prod(shape))))
    return torch.cat(tensors)

# Utility for applying parameters to the respective layers of a RL policy network
def pack_weights(weights_hypernet: torch.Tensor, weights_storage: MILParamStorage, alpha: float, state_dim: int, hdim: int):
    weights = alpha * weights_hypernet + (1 - alpha) * weights_storage

    shapes = OrderedDict()
    shapes["actor.actor.0.weight"] = (256, state_dim)
    shapes["actor.actor.0.bias"] = (256)
    shapes["actor.actor.2.weight"] = (32, 256)
    shapes["actor.actor.2.bias"] = (32)
    shapes["actor.actor.4.weight"] = (1, 32)
    shapes["actor.actor.4.bias"] = (1)
    shapes["critic.critic.0.weight"] = (hdim, state_dim)
    shapes["critic.critic.0.bias"] = (hdim)
    shapes["critic.critic.2.weight"] = (1, hdim)
    shapes["critic.critic.2.bias"] = (1)
    params = {}

    idx = 0
    for key, shape in shapes.items():
        offset = np.prod(shape)
        params[key] = weights[idx:(idx + offset)].reshape(shape)
        idx += offset
    
    return params