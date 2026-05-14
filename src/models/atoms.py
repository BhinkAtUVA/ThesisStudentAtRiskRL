from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, f1_score, r2_score, roc_auc_score

def build_layers(sizes):
    layers = []

    for in_size, out_size in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(in_size, out_size))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class BaseNetwork(nn.Module):
    def __init__(self, layer_sizes=None):
        super(BaseNetwork, self).__init__()
        self.layer_sizes = layer_sizes
        if self.layer_sizes:
            self.net = build_layers(layer_sizes)
            self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.layer_sizes:
            # batch_size, bag_size, d = x.size()
            # x = x.view(batch_size * bag_size, d)
            x = self.net(x)
            # x = x.view(batch_size, bag_size, -1)
        return x


class SimpleMLP(nn.Module, ABC):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            dropout_p: float = 0.5,
    ):
        super(SimpleMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_p = dropout_p  # register the droupout probability as a buffer

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_dim, self.output_dim),
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