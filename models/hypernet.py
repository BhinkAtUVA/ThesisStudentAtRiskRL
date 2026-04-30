from typing import List, Tuple

import numpy as np
from torch import nn
import torch

from models import PolicyNetwork

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
    

# Utility for applying parameters to the respective layers of a RL policy network
def propagate_weights(weights_hypernet: torch.Tensor, weights_storage: MILParamStorage, alpha: float, net: PolicyNetwork, structure: List[List[List[Tuple[int]]]]):
    weights = alpha * weights_hypernet + (1 - alpha) * weights_storage

    actor_structure = structure[0]
    critic_structure = structure[1]

    idx, actor_params, critic_params = 0, [], []
    for struct, params in zip((actor_structure, critic_structure), (actor_params, critic_params)):
        for layer in struct:
            layer_params = []
            for shape in layer:
                offset = np.prod(shape)
                layer_params.append(weights[:, idx:(idx + offset)].reshape(shape))
                idx += offset
            params.append(layer_params)
    
    idx = 0
    for submod in net.actor.actor:
        if submod is nn.Linear:
            wb = actor_params[idx]
            w, b = wb[0], wb[1]
            submod.weight = w
            submod.bias = b
            idx += 1
    idx = 0
    for submod in net.critic.critic:
        if submod is nn.Linear:
            wb = actor_params[idx]
            w, b = wb[0], wb[1]
            submod.weight = w
            submod.bias = b
            idx += 1