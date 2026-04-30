import torch

from models.full import RLMILDebias
from trainers.base import RLMILTrainer

class RLMILDebiasTrainer(RLMILTrainer):
    def __init__(self, model: RLMILDebias, **kwargs):
        super(RLMILDebiasTrainer, self).__init__(
            model=model,
            learning_rate=kwargs['learning_rate'],
            device=kwargs['device'],
            task_type=kwargs['task_type'],
            min_clip=kwargs['min_clip'],
            max_clip=kwargs['max_clip'],
            sample_algorithm=kwargs['sample_algorithm'],
        )

    def get_model_constructor():
        return RLMILDebias
    
    def train_minibatch(self, batch_x, batch_y):
        self.model.task_model.train()
        batch_out = self.model.task_model(batch_x)
        batch_loss = self.loss_fn(batch_out.squeeze(), batch_y.squeeze())
        batch_bias_pred = self.model.debiasing_model(self.model.batch_hidden)
        batch_bias_loss = self.loss_fn(batch_bias_pred.squeeze(), torch.max(batch_x[:, (2, 4, 5, 7), :], dim=-1).values) # Indices of protected features, maximum is valid because instances of protected features are sparse
        self.task_optim.zero_grad()
        batch_loss.backward(retain_graph=True)
        # self.task_optim.step() # Moved to training script for using biases in total_loss
        return batch_loss.item(), batch_bias_loss

    def eval_minibatch(self, batch_x, batch_y):
        self.model.task_model.eval()
        batch_out = self.model.task_model(batch_x)
        batch_loss = self.loss_fn(batch_out.squeeze(), batch_y.squeeze())
        return batch_out, batch_loss.item()