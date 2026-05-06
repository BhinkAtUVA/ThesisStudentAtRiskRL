from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.func import functional_call

from models.adversary import AdversarialMLP
from models.hypernet import MILHypernetwork, get_num_weights, init_policy_storage, pack_weights
from models.rl import PolicyNetwork

# Utility for better typing
class NetworkContainer(ABC):
    @abstractmethod
    def action(self, batch_x) -> tuple[torch.Tensor, torch.Tensor, float]:
        pass

    @abstractmethod
    def predict(self, loss_fn, batch_x, batch_y) -> tuple[torch.Tensor, float]:
        pass
    
    @abstractmethod
    def predict_train(self, loss_fn, task_optim, batch_x, batch_y) -> float:
        pass

    @abstractmethod
    def store_in_buffer(self, transition: tuple[torch.Tensor | float]):
        pass

    @abstractmethod
    def reset_buffers(self):
        pass

    @abstractmethod
    def normalize_rewards(self):
        pass

# Status quo
class RLMILBase(NetworkContainer):
    def __init__(self, **kwargs):
        super(RLMILBase, self).__init__()
        # self.args = args
        self.policy = PolicyNetwork(state_dim=kwargs['state_dim'], hdim=kwargs['hdim'])
        self.task_model = kwargs['task_model']
        self.no_autoencoder = kwargs.get('no_autoencoder', False)

        self.saved_actions = []
        self.rewards = []

    def action(self, batch_x):
        if self.no_autoencoder:
            batch_rep = batch_x
        else:
            batch_rep = self.task_model.base_network(batch_x).detach()

        action_probs, exp_reward = self.policy(batch_rep)
        action_probs = action_probs.squeeze(-1)

        exp_reward = torch.mean(exp_reward, dim=1)
        return action_probs, batch_rep, exp_reward

    def predict(self, loss_fn, batch_x, batch_y):
        self.task_model.eval()
        batch_out = self.task_model(batch_x)
        batch_loss = loss_fn(batch_out.squeeze(), batch_y.squeeze())
        return batch_out, batch_loss.item()
    
    def predict_train(self, loss_fn, task_optim, batch_x, batch_y):
        self.task_model.train()
        batch_out = self.task_model(batch_x)
        batch_loss = loss_fn(batch_out.squeeze(), batch_y.squeeze())
        task_optim.zero_grad()
        batch_loss.backward()
        task_optim.step()
        return batch_loss.item()
    
    def store_in_buffer(self, transition):
        if(len(transition) != 2):
            ValueError

        self.saved_actions.append(transition[0])
        self.rewards.append(transition[1])

    def reset_buffers(self):
        self.saved_actions, self.rewards = [], []

    def normalize_rewards(self, eps=1e-5):
        R_mean = np.mean(self.rewards)
        R_std = np.std(self.rewards)
        for i, r in enumerate(self.rewards):
            self.rewards[i] = float((r - R_mean) / (R_std + eps))

# TODO: Remove when complete full model is established
class RLMILDebias(RLMILBase):
    def __init__(self, **kwargs):
        super(RLMILDebias, self).__init__(
            task_model=kwargs['task_model'],
            state_dim=kwargs['state_dim'],
            hdim=kwargs['hdim'],
            no_autoencoder=kwargs['no_autoencoder'],
        )
        # self.args = args
        self.debiasing_model = AdversarialMLP(kwargs["hidden_dim"], kwargs["hidden_dim"] // 4, 4)
        self.task_model.mlp[-2].register_forward_hook(self._peek_task_last_hidden)
    
    def _peek_task_last_hidden(self, module, input, output):
        self.batch_hidden = output

class HypernetRLMIL(NetworkContainer):
    def __init__(self, **kwargs):
        super(RLMILBase, self).__init__()
        self.state_dim = kwargs["state_dim"]
        self.hdim = kwargs["hdim"]

        # self.args = args
        self.num_weights = get_num_weights(self.state_dim, self.hdim)
        self.hyper = MILHypernetwork(1, 256, self.num_weights)
        self.policy_weights = init_policy_storage(self.state_dim, self.hdim)
        self.preference = torch.zeros((1))

        self.policy = PolicyNetwork(state_dim=self.state_dim, hdim=self.hdim)
        self.task_model = kwargs['task_model']
        self.no_autoencoder = kwargs.get('no_autoencoder', False)

        self.saved_actions = []
        self.rewards = []
        self.preferences = []
        
        self.debiasing_model = AdversarialMLP(kwargs["hidden_dim"], kwargs["hidden_dim"] // 4, 4)
        self.task_model.mlp[-2].register_forward_hook(self._peek_task_last_hidden)
    
    def _peek_task_last_hidden(self, module, input, output):
        self.batch_hidden = output

    def set_preference(self, value: torch.Tensor):
        self.preference = value

    def action(self, batch_x):
        if self.no_autoencoder:
            batch_rep = batch_x
        else:
            batch_rep = self.task_model.base_network(batch_x).detach()

        hyper_weights = self.hyper(self.preference)
        combined_weights = pack_weights(hyper_weights, self.policy_weights, 0.05, self.state_dim, self.hdim)

        action_probs, exp_reward = functional_call(self.policy, combined_weights, batch_rep)
        action_probs = action_probs.squeeze(-1)

        exp_reward = torch.mean(exp_reward, dim=1)
        return action_probs, batch_rep, exp_reward
    
    def predict(self, loss_fn, batch_x, batch_y):
        self.task_model.eval()
        batch_out = self.task_model(batch_x)
        batch_loss = loss_fn(batch_out.squeeze(), batch_y.squeeze())
        return batch_out, batch_loss.item()
    
    def predict_train(self, loss_fn, task_optim, batch_x, batch_y):
        self.task_model.train()
        batch_out = self.task_model(batch_x)
        batch_loss = loss_fn(batch_out.squeeze(), batch_y.squeeze())
        batch_bias_pred = self.debiasing_model(self.batch_hidden)
        batch_bias_loss = loss_fn(batch_bias_pred.squeeze(), torch.max(batch_x[:, (2, 4, 5, 7), :], dim=-1).values) # Indices of protected features, maximum is valid because instances of protected features are sparse
        task_optim.zero_grad()
        batch_loss.backward(retain_graph=True)
        # self.task_optim.step() # Moved to training script for using biases in total_loss
        return batch_loss.item(), batch_bias_loss
    
    def store_in_buffer(self, transition):
        if(len(transition) != 2):
            ValueError

        self.saved_actions.append(transition[0])
        self.rewards.append(transition[1])
        self.preferences.append(self.preference.item())

    def reset_buffers(self):
        self.saved_actions, self.rewards = [], []

    def normalize_rewards(self, eps=1e-5):
        R_mean = np.mean(self.rewards)
        R_std = np.std(self.rewards)
        for i, r in enumerate(self.rewards):
            self.rewards[i] = float((r - R_mean) / (R_std + eps))