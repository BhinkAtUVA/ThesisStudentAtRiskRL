from collections import OrderedDict
import math

import numpy as np
from torch import nn
import torch

# Except for the applied Sigmoid at the end, this network is very similar to the SimpleMLP in atoms.py
class Hypernetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_weights: int,
        dropout_p: float = 0.5,
    ):
        super(Hypernetwork, self).__init__()
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

# Transforms low dimensional vectors into high dimensional frequency-based embeddings
class FourierEmbedding(nn.Module):
    def __init__(self, in_features=1, embedding_dim=64, scale=5.0):
        super().__init__()
        assert embedding_dim % 2 == 0, "Embedding dimension must be even"
        self.embedding_dim = embedding_dim
        
        # Sample random frequencies
        self.register_buffer("B", torch.randn(in_features, embedding_dim // 2) * scale)

    def forward(self, x: torch.Tensor):
        projection = 2 * math.pi * x @ self.B
        sin_comp, cos_comp = torch.sin(projection), torch.cos(projection)
        return torch.cat([sin_comp, cos_comp], dim=-1)
    
class FourierHypernetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_weights: int,
        embedding_dim: int = 64,
        fourier_scale: float = 5.0,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_weights = num_weights
        self.fourier_scale = fourier_scale
        self.dropout_p = dropout_p  # register the droupout probability as a buffer

        self.mlp = nn.Sequential(
            FourierEmbedding(self.input_dim, self.embedding_dim, self.fourier_scale),
            nn.Linear(self.embedding_dim, self.hidden_dim),
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
class ParamStorage(torch.Tensor):
    pass

# Calculates the amount of weights for a given MIL task model configuration
def get_num_weights_task(input_dim: int, hidden_dim: int, output_dim: int):
    return (input_dim + 1) * hidden_dim + (hidden_dim + 1) * output_dim
# Calculates the amount of weights for a given PolicyNetwork configuration
def get_num_weights_policy(state_dim: int, hdim: int):
    return 8514 + 256 * state_dim + hdim * state_dim + hdim + hdim

def init_storage(shapes: list[tuple[int]]):
    tensors = []
    for shape in shapes:
        if type(shape) != tuple or len(shape) == 1: tensors.append(torch.fill(torch.zeros(shape, requires_grad=True), 0.01))
        else: tensors.append(torch.nn.init.xavier_uniform_(torch.zeros(shape)).reshape((np.prod(shape))))
    return torch.cat(tensors).detach()
# Utility for correctly initializing a 1D-Tensor representing parameters of a PolicyNetwork with the given Configuration
def init_policy_storage(state_dim: int, hdim: int) -> ParamStorage:
    return init_storage([
        (256, state_dim),
        (256,),
        (32, 256),
        (32,),
        (1, 32),
        (1,),
        (hdim, state_dim),
        (hdim),
        (1, hdim),
        (1,)
    ])
def init_task_storage(pretrained_state_dict: dict[str, torch.Tensor]) -> ParamStorage:
    tensors = []
    for _, v in pretrained_state_dict.items():
        tensors.append(v.reshape(np.prod(v.shape)))
    return torch.cat(tensors).detach()

def pack_weights(hypernet_weights: torch.Tensor, stored_weights: ParamStorage, alpha: float, shapes: OrderedDict[str, tuple[int]]):
    weights = alpha * hypernet_weights + (1 - alpha) * stored_weights
    
    params = {}
    idx = 0
    for key, shape in shapes.items():
        offset = math.prod(shape)
        params[key] = weights[idx:(idx + offset)].view(shape)
        idx += offset
    
    return params
# Utility for applying parameters to the respective layers of a RL policy network
def pack_weights_policy(hypernet_weights: torch.Tensor, stored_weights: ParamStorage, alpha: float, state_dim: int, hdim: int):
    shapes = OrderedDict()
    shapes["actor.actor.0.weight"] = (256, state_dim)
    shapes["actor.actor.0.bias"] = (256,)
    shapes["actor.actor.2.weight"] = (32, 256)
    shapes["actor.actor.2.bias"] = (32,)
    shapes["actor.actor.4.weight"] = (1, 32)
    shapes["actor.actor.4.bias"] = (1,)
    shapes["critic.critic.0.weight"] = (hdim, state_dim)
    shapes["critic.critic.0.bias"] = (hdim,)
    shapes["critic.critic.2.weight"] = (1, hdim)
    shapes["critic.critic.2.bias"] = (1,)
    return pack_weights(hypernet_weights, stored_weights, alpha, shapes)
def pack_weights_task(hypernet_weights: torch.Tensor, stored_weights: ParamStorage, alpha: float, input_dim: int, hidden_dim: int, output_dim: int, is_repset: bool):
    shapes = OrderedDict()
    shapes["0.weight"] = (hidden_dim, input_dim)
    shapes["0.bias"] = (hidden_dim,)
    shapes["3.weight" if not is_repset else "2.weight"] = (output_dim, hidden_dim)
    shapes["3.bias" if not is_repset else "2.bias"] = (output_dim,)
    return pack_weights(hypernet_weights, stored_weights, alpha, shapes)