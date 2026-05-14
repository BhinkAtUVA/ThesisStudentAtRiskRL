import numpy as np
import torch

from models.full import HypernetRLMIL
from models.rl import sample_action, select_from_action
from trainers.base import RLMILTrainer

# TODO: Remove when trianer for full model is established
class HypernetRLMILTrainer(RLMILTrainer):
    def __init__(self, net_container: HypernetRLMIL, **kwargs):
        super(HypernetRLMILTrainer, self).__init__(
            net_container=net_container,
            learning_rate=kwargs['learning_rate'],
            device=kwargs['device'],
            task_type=kwargs['task_type'],
            min_clip=kwargs['min_clip'],
            max_clip=kwargs['max_clip'],
            sample_algorithm=kwargs['sample_algorithm'],
        )
        self.net_container = net_container

    def get_model_constructor():
        return HypernetRLMIL
    
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
        # Sample preference and apply it to the hypernet
        preference = torch.rand((1), device=device)
        self.net_container.set_preference(preference)

        # Get one selection of eval data for computing reward
        self.net_container.policy.eval()
        eval_pool = self.create_pool_data(eval_dataloader, bag_size, train_pool_size, random=only_ensemble)
        sel_losses, regularization_losses, bias_losses = [], [], []
        for batch_x, batch_y, _, _  in train_dataloader:
            self.net_container.policy.train()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            action_probs, _, _ = self.net_container.action(batch_x)
            # logger.info(f"action_probs.shape={action_probs.shape}")
            action, action_log_prob = sample_action(action_probs, 
                                                    bag_size, 
                                                    device=device, 
                                                    random=(epsilon > np.random.random()) or only_ensemble,
                                                    algorithm=sample_algorithm)
            sel_x = select_from_action(action, batch_x)
            sel_y = batch_y
            sel_loss, bias_loss = self.net_container.predict_train(self.loss_fn, self.task_optim, sel_x, sel_y)
            sel_losses.append(sel_loss)
            bias_losses.append(bias_loss)
            self.net_container.policy.eval()
            # reward = policy_network.compute_reward(eval_data)
            if not only_ensemble:
                reward, _, _ = self.expected_reward_loss(eval_pool)
                reward += 10.0 * bias_loss.item()
                self.net_container.store_in_buffer((action_log_prob, reward))
                regularization_losses.append(action_probs.sum(dim=-1).mean(dim=-1))

        
        if only_ensemble:
            return 0, 0, 0, np.mean(sel_losses), 0

        self.net_container.normalize_rewards(eps=1e-5)

        policy_losses = []
        self.net_container.policy.train()
        for log_prob, reward in zip(self.net_container.saved_actions, self.net_container.rewards):
            policy_losses.append(-reward * log_prob.cuda())

        # TODO: Decide on whether or not to sample randomly from batch results for training

        optimizer.zero_grad()
        policy_loss = torch.cat(policy_losses).mean()
        regularization_loss = torch.stack(regularization_losses).mean() / 100
        bias_loss = torch.stack(bias_losses).mean()
        total_loss = (1 - preference.item()) * policy_loss + reg_coef * regularization_loss - preference.item() * bias_loss
        # perform backprop
        total_loss.backward()

        optimizer.step()
        self.task_optim.step()
        
        if scheduler is not None:
            scheduler.step()
        # reset rewards and action buffer
        self.net_container.reset_buffers()

        return total_loss.item(), policy_loss.item(), 0, \
            np.mean(sel_losses), reg_coef * regularization_loss.item()