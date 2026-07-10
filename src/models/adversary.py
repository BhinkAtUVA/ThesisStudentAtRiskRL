import torch
from torch import nn

from src.models.atoms import SimpleMLP

# With this, one call to .backward() on the loss is enough to train main model and adversary differently (currently unused due to separate backpropagation calls)
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient during backprop
        return grad_output.neg(), None

# Multi-head MLP with gradient reversal
class AdversarialMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dims: list[int],
            dropout_p: float = 0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dims = output_dims
        self.dropout_p = dropout_p  # register the droupout probability as a buffer

        self.shared = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p)
        )
        self.heads = nn.ModuleList([nn.Linear(self.hidden_dim, out_dim) for out_dim in self.output_dims])

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_result = self.shared(GradientReversal.apply(x))
        return [head(hidden_result) for head in self.heads] # GradientReversal.apply(head(hidden_result)...)