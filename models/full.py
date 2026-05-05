from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.func import functional_call

from models.adversary import AdversarialMLP
from models.hypernet import MILHypernetwork, MILParamStorage, get_num_weights
from models.rl import PolicyNetwork

# Utility for better typing
class NetworkContainer(ABC):
    @abstractmethod
    def action(self, batch_x) -> tuple[torch.Tensor, torch.Tensor, float]:
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
        # self.args = args
        self.num_weights = get_num_weights(kwargs["state_dim"], kwargs["state_dim"])
        self.hyper = MILHypernetwork(1, 256, self.num_weights) # TODO: Calculate number of weights properly
        self.policy_weights: MILParamStorage = torch.Tensor()

        self.policy = PolicyNetwork(state_dim=kwargs['state_dim'], hdim=kwargs['hdim'])
        self.task_model = kwargs['task_model']
        self.no_autoencoder = kwargs.get('no_autoencoder', False)

        self.saved_actions = []
        self.rewards = []
        self.preferences = []

    def action(self, batch_x):
        if self.no_autoencoder:
            batch_rep = batch_x
        else:
            batch_rep = self.task_model.base_network(batch_x).detach()

        action_probs, exp_reward = self.policy(batch_rep)
        action_probs = action_probs.squeeze(-1)

        exp_reward = torch.mean(exp_reward, dim=1)
        return action_probs, batch_rep, exp_reward
    
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