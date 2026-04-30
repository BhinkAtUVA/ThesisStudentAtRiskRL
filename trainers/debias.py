import numpy as np
import torch

from models.full import RLMILDebias
from models.rl import sample_action, select_from_action
from trainers.base import RLMILTrainer

# TODO: Remove when trianer for full model is established
class RLMILDebiasTrainer(RLMILTrainer):
    def __init__(self, net_container: RLMILDebias, **kwargs):
        super(RLMILDebiasTrainer, self).__init__(
            net_container=net_container,
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
        self.net_container.task_model.train()
        batch_out = self.net_container.task_model(batch_x)
        batch_loss = self.loss_fn(batch_out.squeeze(), batch_y.squeeze())
        batch_bias_pred = self.net_container.debiasing_model(self.net_container.batch_hidden)
        batch_bias_loss = self.loss_fn(batch_bias_pred.squeeze(), torch.max(batch_x[:, (2, 4, 5, 7), :], dim=-1).values) # Indices of protected features, maximum is valid because instances of protected features are sparse
        self.task_optim.zero_grad()
        batch_loss.backward(retain_graph=True)
        # self.task_optim.step() # Moved to training script for using biases in total_loss
        return batch_loss.item(), batch_bias_loss

    def eval_minibatch(self, batch_x, batch_y):
        self.net_container.task_model.eval()
        batch_out = self.net_container.task_model(batch_x)
        batch_loss = self.loss_fn(batch_out.squeeze(), batch_y.squeeze())
        return batch_out, batch_loss.item()
    
    def episode(
        self,
        train_dataloader,
        eval_dataloader,
        optimizer,
        device,
        bag_size,
        train_pool_size,
        scheduler,
        only_ensemble, 
        epsilon,
        reg_coef, 
        sample_algorithm
    ):
        # Get one selection of eval data for computing reward
        self.net_container.policy.eval()
        eval_pool = self.net_container.policy.create_pool_data(eval_dataloader, bag_size, train_pool_size, random=only_ensemble)
        sel_losses, regularization_losses, bias_losses = [], [], []
        for batch_x, batch_y, _, _  in train_dataloader:
            self.net_container.policy.train()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            action_probs, _, _ = self.net_container.policy(batch_x)
            # logger.info(f"action_probs.shape={action_probs.shape}")
            action, action_log_prob = sample_action(action_probs, 
                                                    bag_size, 
                                                    device=device, 
                                                    random=(epsilon > np.random.random()) or only_ensemble,
                                                    algorithm=sample_algorithm)
            sel_x = select_from_action(action, batch_x)
            sel_y = batch_y
            sel_loss, bias_loss = self.train_minibatch(sel_x, sel_y)
            sel_losses.append(sel_loss)
            bias_losses.append(bias_loss)
            self.net_container.policy.eval()
            # reward = policy_network.compute_reward(eval_data)
            if not only_ensemble:
                reward, _, _ = self.expected_reward_loss(eval_pool)
                reward += 10.0 * bias_loss.item()
                self.net_container.saved_actions.append(action_log_prob)
                self.net_container.rewards.append(reward)
                regularization_losses.append(action_probs.sum(dim=-1).mean(dim=-1))

        
        if only_ensemble:
            return 0, 0, 0, np.mean(sel_losses), 0

        self.net_container.normalize_rewards(eps=1e-5)

        policy_losses = []
        self.net_container.policy.train()
        for log_prob, reward in zip(self.net_container.saved_actions, self.net_container.rewards):
            policy_losses.append(-reward * log_prob.cuda())

        optimizer.zero_grad()
        policy_loss = torch.cat(policy_losses).mean()
        regularization_loss = torch.stack(regularization_losses).mean() / 100
        bias_loss = torch.stack(bias_losses).mean()
        total_loss = policy_loss + reg_coef * regularization_loss - 100.0 * bias_loss
        # perform backprop
        total_loss.backward()

        optimizer.step()
        self.task_optim.step()
        
        if scheduler is not None:
            scheduler.step()
        # reset rewards and action buffer
        self.net_container.reset_reward_action()

        return total_loss.item(), policy_loss.item(), 0, \
            np.mean(sel_losses), reg_coef * regularization_loss.item()