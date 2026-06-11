import torch

from src.models.atoms import SimpleMLP

# With this, one call to .backward() on the loss is enough to train main model and adversary differently
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient during backprop
        return grad_output.neg(), None

# Regular MLP with gradient reversal
class AdversarialMLP(SimpleMLP):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            dropout_p: float = 0.5,
    ):
        super(AdversarialMLP, self).__init__(
            input_dim,
            hidden_dim,
            output_dim,
            dropout_p
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super(AdversarialMLP, self).forward(x) # GradientReversal.apply(x)