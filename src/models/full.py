from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.func import functional_call

from src.models.adversary import AdversarialMLP
from src.models.hypernet import FourierHypernetwork, Hypernetwork, get_num_weights_policy, get_num_weights_task, init_policy_storage, init_task_storage, pack_weights_policy, pack_weights_task
from src.models.mil import ApproxRepSet, BaseMLP
from src.models.rl import PolicyNetwork

# Utility for better typing
class NetworkContainer(ABC):
    @abstractmethod
    def action(self, batch_x) -> tuple[torch.Tensor, torch.Tensor, float]:
        pass

    @abstractmethod
    def predict(self, loss_fn, batch_x, batch_y) -> tuple[torch.Tensor, float, float | None]:
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

    @abstractmethod
    def to(self, device: torch.device):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
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
        
        if kwargs["device"] is not None:
            self.to(kwargs["device"])

    def action(self, batch_x):
        if self.no_autoencoder:
            batch_rep = batch_x
        else:
            batch_rep = self.task_model.base_network(batch_x).detach()

        action_probs, exp_reward = self.policy(batch_rep)
        return action_probs, batch_rep, exp_reward

    def predict(self, loss_fn, batch_x, batch_y):
        self.task_model.eval()
        batch_out = self.task_model(batch_x)
        batch_loss = loss_fn(batch_out.squeeze(), batch_y.squeeze())
        return batch_out, batch_loss.item(), None
    
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

    def to(self, device):
        self.policy.to(device)
        self.task_model.to(device)

    def state_dict(self):
        return self.policy.state_dict()
    
    def load_state_dict(self, state_dict):
        self.policy.load_state_dict(state_dict)

class DebiasWarmup():
    def __init__(self, **kwargs):
        self.hidden_dim = kwargs["hidden_dim"]
        if self.hidden_dim is None:
            self.hidden_dim = 32

        self.task_model: BaseMLP = kwargs['task_model']
        self.no_autoencoder = kwargs.get('no_autoencoder', False)
        
        self.debiasing_model = AdversarialMLP(self.hidden_dim, self.hidden_dim // 4, [2, 11, 3, 5]) # Number of levels for gender, socioeconomic status, age and education
        self.task_model.mlp[-2].register_forward_hook(self._peek_task_last_hidden)

        self.bias_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        if kwargs["device"] is not None:
            self.to(kwargs["device"])
    
    def _peek_task_last_hidden(self, module, input, output):
        self.batch_hidden = output.detach()
    
    def predict(self, batch_x, batch_y, protected_labels):
        self.task_model.eval()
        batch_out = self.task_model(batch_x)
        batch_bias_pred = self.debiasing_model(self.batch_hidden)
        batch_bias_losses = torch.stack([self.bias_loss_fn(pred, target) for pred, target in zip(batch_bias_pred, protected_labels.unbind(dim=-1))])
        batch_bias_loss = batch_bias_losses.mean()
        return batch_out, batch_bias_loss.item()
    
    def predict_train(self, task_optim, batch_x, batch_y, protected_labels):
        self.task_model.train()
        self.task_model(batch_x)
        batch_bias_pred = self.debiasing_model(self.batch_hidden)
        batch_bias_losses = torch.stack([self.bias_loss_fn(pred, target) for pred, target in zip(batch_bias_pred, protected_labels.unbind(dim=-1))])
        batch_bias_loss = batch_bias_losses.mean()
        task_optim.zero_grad()
        batch_bias_loss.backward()
        task_optim.step()
        return batch_bias_loss.item()

    def to(self, device):
        self.task_model = self.task_model.to(device)
        self.debiasing_model = self.debiasing_model.to(device)

    def state_dict(self):
        return self.debiasing_model.state_dict()
    
    def load_state_dict(self, state_dict):
        self.debiasing_model.load_state_dict(state_dict)

class HypernetRL(NetworkContainer):
    def __init__(self, **kwargs):
        super(HypernetRL, self).__init__()
        self.state_dim = kwargs["state_dim"]
        self.hdim = kwargs["hdim"]
        self.hidden_dim = kwargs["hidden_dim"]
        if self.hidden_dim is None:
            self.hidden_dim = 32

        # self.args = args
        self.num_weights = get_num_weights_policy(self.state_dim, self.hdim)
        self.hyper = FourierHypernetwork(1, 512, self.num_weights, kwargs["embedding_dim"] or 64, kwargs["fourier_scale"] or 5.0)
        self.policy_weights = torch.nn.Parameter(init_policy_storage(self.state_dim, self.hdim))
        self.cached_policy = None
        self.hyper_ratio = kwargs["hyper_ratio"] or 0.05
        self.preference = torch.zeros((1), requires_grad=True)

        self.policy = PolicyNetwork(state_dim=self.state_dim, hdim=self.hdim)
        self.task_model: BaseMLP = kwargs['task_model']
        self.no_autoencoder = kwargs.get('no_autoencoder', False)

        self.saved_actions = []
        self.rewards = []
        self.preferences = []
        
        self.debiasing_model: AdversarialMLP = kwargs["debiasing_model"]
        self.task_model.mlp[-2].register_forward_hook(self._peek_task_last_hidden)

        self.bias_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        if kwargs["device"] is not None:
            self.to(kwargs["device"])
    
    def _peek_task_last_hidden(self, module, input, output):
        self.batch_hidden = output.detach()

    def set_preference(self, value: torch.Tensor):
        self.preference = value
        self.cached_policy = None

    def action(self, batch_x):
        if self.no_autoencoder:
            batch_rep = batch_x
        else:
            batch_rep = self.task_model.base_network(batch_x).detach()

        if self.cached_policy is None:
            hyper_weights = self.hyper(self.preference)
            self.cached_policy = pack_weights_policy(hyper_weights, self.policy_weights, self.hyper_ratio, self.state_dim, self.hdim)

        action_probs, exp_reward = functional_call(self.policy, self.cached_policy, batch_rep)

        return action_probs, batch_rep, exp_reward
    
    def predict(self, loss_fn, batch_x, batch_y, protected_labels):
        self.task_model.eval()
        batch_out = self.task_model(batch_x)
        batch_loss = loss_fn(batch_out.squeeze(), batch_y.squeeze())
        batch_bias_pred = self.debiasing_model(self.batch_hidden)
        batch_bias_losses = torch.stack([self.bias_loss_fn(pred, target) for pred, target in zip(batch_bias_pred, protected_labels.unbind(dim=-1))])
        batch_bias_loss = batch_bias_losses.mean()
        return batch_out, batch_loss.item(), batch_bias_loss.item()
    
    def predict_train(self, loss_fn, task_optim, batch_x, batch_y, protected_labels):
        self.task_model.train()
        batch_out = self.task_model(batch_x)
        batch_loss = loss_fn(batch_out.squeeze(), batch_y.squeeze())
        batch_bias_pred = self.debiasing_model(self.batch_hidden)
        batch_bias_losses = torch.stack([self.bias_loss_fn(pred, target) for pred, target in zip(batch_bias_pred, protected_labels.unbind(dim=-1))])
        batch_bias_loss = batch_bias_losses.mean()
        task_optim.zero_grad()
        batch_loss.backward()
        batch_bias_loss.backward()
        task_optim.step()
        return batch_loss.item(), batch_bias_loss.item()
    
    def store_in_buffer(self, transition):
        if(len(transition) != 2):
            ValueError

        self.saved_actions.append(transition[0])
        self.rewards.append(transition[1])
        self.preferences.append(self.preference.item())

    def reset_buffers(self):
        self.saved_actions, self.rewards, self.cached_policy = [], [], None

    def normalize_rewards(self, eps=1e-5):
        rewards_tensor = torch.cat(self.rewards)
        R_mean = rewards_tensor.mean()
        R_std = rewards_tensor.std()
        for i, r in enumerate(self.rewards):
            self.rewards[i] = (r - R_mean) / (R_std + eps)

    def to(self, device):
        self.hyper = self.hyper.to(device)
        self.task_model = self.task_model.to(device)
        self.debiasing_model = self.debiasing_model.to(device)
        self.policy_weights = torch.nn.Parameter(self.policy_weights.to(device).detach())
        self.preference = self.preference.to(device)
        self.cached_policy = None

    def state_dict(self):
        return {
            "hyper": self.hyper.state_dict(),
            "storage": self.policy_weights,
            "debias": self.debiasing_model.state_dict(),
            "task": self.task_model.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.hyper.load_state_dict(state_dict["hyper"])
        policy_weights = state_dict["storage"]
        if self.policy_weights.get_device() >= 0: policy_weights = policy_weights.to(self.policy_weights.get_device())
        self.policy_weights = policy_weights
        self.debiasing_model.load_state_dict(state_dict["debias"])
        if "task" in state_dict: self.task_model.load_state_dict(state_dict["task"])
        self.cached_policy = None

class HypernetRLMIL(NetworkContainer):
    def __init__(self, **kwargs):
        super(HypernetRLMIL, self).__init__()
        self.state_dim = kwargs["state_dim"]
        self.hdim = kwargs["hdim"]
        self.hidden_dim = kwargs["hidden_dim"]
        if self.hidden_dim is None:
            self.hidden_dim = 32

        self.policy = PolicyNetwork(state_dim=self.state_dim, hdim=self.hdim)
        self.task_model: BaseMLP = kwargs['task_model']
        self.no_autoencoder = kwargs.get('no_autoencoder', False)

        # self.args = args
        self.is_repset = type(self.task_model) == ApproxRepSet
        self.task_hidden_dim, self.task_input_dim = next(self.task_model.mlp[0].parameters()).size()
        self.task_output_dim = next(self.task_model.mlp[3].parameters()).size()[0] if not self.is_repset else next(self.task_model.mlp[2].parameters()).size()[0]
        self.num_weights_task = get_num_weights_task(self.task_input_dim, self.task_hidden_dim, self.task_output_dim)
        self.num_weights_policy = get_num_weights_policy(self.state_dim, self.hdim)
        self.hyper = FourierHypernetwork(1, 512, self.num_weights_task + self.num_weights_policy, kwargs["embedding_dim"] or 64, kwargs["fourier_scale"] or 5.0)
        self.task_weights = torch.nn.Parameter(init_task_storage(self.task_model.mlp.state_dict()))
        self.policy_weights = torch.nn.Parameter(init_policy_storage(self.state_dim, self.hdim))
        self.cached_task = None
        self.cached_policy = None
        self.hyper_ratio = kwargs["hyper_ratio"] or 0.05
        self.preference = torch.zeros((1), requires_grad=True)

        self.saved_actions = []
        self.rewards = []
        self.preferences = []
        
        self.debiasing_model: AdversarialMLP = kwargs["debiasing_model"]
        self.task_model.mlp[-2].register_forward_hook(self._peek_task_last_hidden)

        self.bias_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        if kwargs["device"] is not None:
            self.to(kwargs["device"])
    
    def _peek_task_last_hidden(self, module, input, output):
        self.batch_hidden = output

    def set_preference(self, value: torch.Tensor):
        self.preference = value
        self.cached_task = None
        self.cached_policy = None

    def action(self, batch_x):
        if self.no_autoencoder:
            batch_rep = batch_x
        else:
            batch_rep = self.task_model.base_network(batch_x).detach()

        if self.cached_task is None or self.cached_policy is None:
            hyper_weights: torch.Tensor = self.hyper(self.preference)
            self.cached_task = pack_weights_task(hyper_weights[0:self.num_weights_task], self.task_weights, self.hyper_ratio, self.task_input_dim, self.task_hidden_dim, self.task_output_dim, self.is_repset)
            self.task_model.mlp.load_state_dict(self.cached_task)
            self.cached_policy = pack_weights_policy(hyper_weights[self.num_weights_task:None], self.policy_weights, self.hyper_ratio, self.state_dim, self.hdim)

        action_probs, exp_reward = functional_call(self.policy, self.cached_policy, batch_rep)

        return action_probs, batch_rep, exp_reward
    
    def predict(self, loss_fn, batch_x, batch_y, protected_labels):
        batch_out = self.task_model(batch_x)
        batch_loss = loss_fn(batch_out.squeeze(), batch_y.squeeze())
        batch_bias_pred = self.debiasing_model(self.batch_hidden)
        batch_bias_losses = torch.stack([self.bias_loss_fn(pred, target) for pred, target in zip(batch_bias_pred, protected_labels.unbind(dim=-1))])
        batch_bias_loss = batch_bias_losses.mean()
        return batch_out, batch_loss.item(), batch_bias_loss.item()
    
    def predict_train(self, loss_fn, task_optim, batch_x, batch_y, protected_labels):
        self.task_model.train()
        batch_out = self.task_model(batch_x)
        batch_loss = loss_fn(batch_out.squeeze(), batch_y.squeeze())
        batch_bias_pred = self.debiasing_model(self.batch_hidden)
        batch_bias_losses = torch.stack([self.bias_loss_fn(pred, target) for pred, target in zip(batch_bias_pred, protected_labels.unbind(dim=-1))])
        batch_bias_loss = batch_bias_losses.mean()
        task_optim.zero_grad()
        ((1 - self.self.preference) * batch_loss + self.preference * batch_bias_loss).backward()
        task_optim.step()
        return batch_loss.item(), batch_bias_loss.item()
    
    def store_in_buffer(self, transition):
        if(len(transition) != 2):
            ValueError

        self.saved_actions.append(transition[0])
        self.rewards.append(transition[1])
        self.preferences.append(self.preference.item())

    def reset_buffers(self):
        self.saved_actions, self.rewards, self.cached_task, self.cached_policy = [], [], None, None

    def normalize_rewards(self, eps=1e-5):
        rewards_tensor = torch.cat(self.rewards)
        R_mean = rewards_tensor.mean()
        R_std = rewards_tensor.std()
        for i, r in enumerate(self.rewards):
            self.rewards[i] = (r - R_mean) / (R_std + eps)

    def to(self, device):
        self.hyper = self.hyper.to(device)
        self.task_model = self.task_model.to(device)
        self.debiasing_model = self.debiasing_model.to(device)
        self.task_weights = torch.nn.Parameter(self.task_weights.to(device).detach())
        self.policy_weights = torch.nn.Parameter(self.policy_weights.to(device).detach())
        self.preference = self.preference.to(device)
        self.cached_task = None
        self.cached_policy = None

    def state_dict(self):
        return {
            "hyper": self.hyper.state_dict(),
            "task": self.task_weights,
            "policy": self.policy_weights,
            "debias": self.debiasing_model.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.hyper.load_state_dict(state_dict["hyper"])
        task_weights = state_dict["task"]
        if self.task_weights.get_device() >= 0: task_weights = task_weights.to(self.task_weights.get_device())
        self.task_weights = task_weights
        policy_weights = state_dict["policy"]
        if self.policy_weights.get_device() >= 0: policy_weights = policy_weights.to(self.policy_weights.get_device())
        self.policy_weights = policy_weights
        self.debiasing_model.load_state_dict(state_dict["debias"])
        self.cached_task = None
        self.cached_policy = None