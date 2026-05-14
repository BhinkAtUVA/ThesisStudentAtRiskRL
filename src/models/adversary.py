import torch

from models.atoms import SimpleMLP

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
        return super(AdversarialMLP, self).forward(GradientReversal.apply(x))
    
""" class DebiasingPolicyNetwork(PolicyNetwork):
    def __init__(self, **kwargs):
        super(DebiasingPolicyNetwork, self).__init__(
            task_model=kwargs['task_model'],
            state_dim=kwargs['state_dim'],
            hdim=kwargs['hdim'],
            learning_rate=kwargs['learning_rate'],
            device=kwargs['device'],
            task_type=kwargs['task_type'],
            min_clip=kwargs['min_clip'],
            max_clip=kwargs['max_clip'],
            sample_algorithm=kwargs['sample_algorithm'],
            no_autoencoder=kwargs['no_autoencoder'],
        )
        # self.args = args
        self.debiasing_model = AdversarialMLP(kwargs["hidden_dim"], kwargs["hidden_dim"] // 4, 4)
        self.task_model.mlp[-2].register_forward_hook(self._peek_task_last_hidden)
    
    def _peek_task_last_hidden(self, module, input, output):
        self.batch_hidden = output

    def train_minibatch(self, batch_x, batch_y):
        self.task_model.train()
        batch_out = self.task_model(batch_x)
        batch_loss = self.loss_fn(batch_out.squeeze(), batch_y.squeeze())
        batch_bias_pred = self.debiasing_model(self.batch_hidden)
        batch_bias_loss = self.loss_fn(batch_bias_pred.squeeze(), torch.max(batch_x[:, (2, 4, 5, 7), :], dim=-1).values) # Indices of protected features, maximum is valid because instances of protected features are sparse
        self.task_optim.zero_grad()
        batch_loss.backward(retain_graph=True)
        # self.task_optim.step() # Moved to training script for using biases in total_loss
        return batch_loss.item(), batch_bias_loss

    def eval_minibatch(self, batch_x, batch_y):
        self.task_model.eval()
        batch_out = self.task_model(batch_x)
        batch_loss = self.loss_fn(batch_out.squeeze(), batch_y.squeeze())
        return batch_out, batch_loss.item() """