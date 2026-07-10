import numpy as np
from sklearn.metrics import f1_score, r2_score
import torch
from torch import optim

from src.models.full import HypernetRL, HypernetRLMIL
from src.models.rl import sample_action, select_from_action
from src.trainers.base import RLMILTrainer
from src.util.timing import TimingAnalyzer

class HypernetRLTrainer(RLMILTrainer):
    def __init__(self, net_container: HypernetRL, **kwargs):
        super(HypernetRLTrainer, self).__init__(
            net_container=net_container,
            learning_rate=kwargs['learning_rate'],
            device=kwargs['device'],
            task_type=kwargs['task_type'],
            min_clip=kwargs['min_clip'],
            max_clip=kwargs['max_clip'],
            sample_algorithm=kwargs['sample_algorithm'],
        )
        try:
            self.task_optim = optim.AdamW([
                { "params": self.net_container.task_model.parameters() },
                { "params": self.net_container.debiasing_model.parameters(), "lr": 1.0e-05 } # Train the debiasing model slowly to allow hypernet to make meaningful adjustments
            ], lr=self.learning_rate)
        except:
            self.task_optim = None

        self.net_container = net_container

    def get_model_constructor():
        return HypernetRL
    
    def make_optimizer(net_container: HypernetRL, learning_rate):
        return optim.AdamW(
            [{"params": net_container.hyper.parameters(),
            "lr": learning_rate,},
            {"params": [net_container.policy_weights],
            "lr": learning_rate,},
            {"params": net_container.debiasing_model.parameters(),
            "lr": learning_rate,}],
            lr=learning_rate,
        )
        
    def select_from_dataloader(self, dataloader, bag_size, random=False):
        with torch.no_grad():
            data = []
            for batch_x, batch_y, indices, instance_labels in dataloader:
                batch_x = batch_x.to(self.device)
                # select batch_x
                action_probs, _, _ = self.net_container.action(batch_x)
                action, _ = sample_action(action_probs, bag_size, self.device, random=random, algorithm=self.sample_algorithm)
                true_indices = (torch.max(batch_x[:, (2, 4, 5, 7), :], dim=-1).values - 1).to(dtype=torch.int64)
                batch_x = select_from_action(action, batch_x)
                batch_x = batch_x.cpu()
                data.append((batch_x, batch_y, indices, instance_labels, true_indices))
        return data
    
    def compute_reward(self, eval_data, preference):
        with torch.no_grad():
            data_ys, pred_ys, losses, prob_ys, hyper_rewards = [], [], [], [], []
            for batch_x, batch_y, _, _, true_indices in eval_data:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_out, loss, bias_loss = self.net_container.predict(self.loss_fn, batch_x, batch_y, true_indices)
                hyper_rewards.append(preference * bias_loss - (1 - preference) * loss)

                if self.task_type == 'regression':
                    prob_y = pred_out
                    pred_y = torch.clamp(pred_out, min=self.min_clip, max=self.max_clip)
                elif self.task_type == 'classification':
                    prob_y = torch.softmax(pred_out, dim=1)
                    pred_y = torch.argmax(pred_out, dim=1)
                    
                pred_ys.append(pred_y.detach().cpu())
                prob_ys.append(prob_y.detach().cpu())
                data_ys.append(batch_y.detach().cpu())
                losses.append(loss)
            pred_Y = torch.cat(pred_ys, dim=0)
            data_Y = torch.cat(data_ys, dim=0)
            prob_Y = torch.cat(prob_ys, dim=0)
            if self.task_type == 'classification':
                reward = f1_score(data_Y.data, pred_Y.data, average='macro')
            elif self.task_type == 'regression':   
                reward = r2_score(data_Y.data, pred_Y.data)
        return reward, np.mean(losses), prob_Y, data_Y, np.mean(hyper_rewards)

    def expected_reward_loss(self, pool_data, balance_preference=True, average='macro', verbos=False):
        reward_pool, loss_pool, preds_pool, hyper_reward_pool = [], [], [], []
        self.net_container.cached_policy = None

        prefs = np.linspace(0, 1, len(pool_data))
        for data, pref in zip(pool_data, prefs):
            if balance_preference: self.net_container.set_preference(torch.fill(torch.zeros((1)), pref).to(self.device))
            reward, loss, preds, labels, hyper_reward = self.compute_reward(data, pref)
            reward_pool.append(reward)
            loss_pool.append(loss)
            preds_pool.append(preds)
            hyper_reward_pool.append(hyper_reward)
        mean_reward = np.mean(reward_pool)
        mean_loss = np.mean(loss_pool)
        mean_hyper_reward = np.mean(hyper_reward_pool)
        self.net_container.cached_policy = None
        return mean_reward, mean_loss, mean_hyper_reward
    
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
        sample_algorithm,
        timer: TimingAnalyzer
    ):
        timer.sub_category("Batches")
        # Sample preference and apply it to the hypernet
        preference = torch.rand((1), device=device, requires_grad=True)
        self.net_container.set_preference(preference)

        # Get one selection of eval data for computing reward
        self.net_container.policy.eval()
        eval_pool = self.create_pool_data(eval_dataloader, bag_size, train_pool_size, random=only_ensemble)
        sel_losses, bias_losses, regularization_losses = [], [], []
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
            true_indices = (torch.max(batch_x[:, (2, 4, 5, 7), :], dim=-1).values - 1).to(dtype=torch.int64)
            sel_loss, bias_loss = self.net_container.predict_train(self.loss_fn, self.task_optim, sel_x, sel_y, true_indices)
            sel_losses.append(sel_loss)
            bias_losses.append(bias_loss)
            self.net_container.policy.eval()
            # reward = policy_network.compute_reward(eval_data)
            if not only_ensemble:
                _, loss, _ = self.expected_reward_loss(eval_pool, balance_preference=False)
                reward = (1 - preference) * (-loss) + preference * bias_loss
                self.net_container.store_in_buffer((action_log_prob, reward))
                regularization_losses.append(action_probs.sum(dim=-1).mean(dim=-1))

        timer.next_category("Loss backpropagation")

        if only_ensemble:
            return 0, 0, 0, np.mean(sel_losses), 0

        self.net_container.normalize_rewards(eps=1e-5) # TODO: Decide whether to normalize at all and if so, how

        policy_losses = []
        self.net_container.policy.train()
        for log_prob, reward in zip(self.net_container.saved_actions, self.net_container.rewards):
            policy_losses.append(-reward * log_prob)

        # TODO: Decide on whether or not to sample randomly from batch results for training

        optimizer.zero_grad()
        policy_loss = torch.cat(policy_losses).mean()
        bias_loss = np.mean(bias_losses)
        regularization_loss = torch.stack(regularization_losses).mean() / 100
        total_loss = 1000 * (policy_loss + reg_coef * regularization_loss) # The gradients are very small, so we blow up the loss artificially here

        # perform backprop
        total_loss.backward()

        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        # reset rewards and action buffer
        self.net_container.reset_buffers()

        return total_loss.item(), policy_loss.item(), 0, \
            np.mean(sel_losses), reg_coef * regularization_loss.item(), bias_loss, preference.item()
    
class HypernetRLMILTrainer(HypernetRLTrainer):
    def get_model_constructor():
        return HypernetRLMIL
    
    def make_optimizer(net_container: HypernetRLMIL, learning_rate):
        return optim.AdamW(
            [{"params": net_container.hyper.parameters(),
            "lr": learning_rate,},
            {"params": [net_container.task_weights],
            "lr": learning_rate,},
            {"params": [net_container.policy_weights],
            "lr": learning_rate,},
            {"params": net_container.debiasing_model.parameters(),
            "lr": learning_rate,}],
            lr=learning_rate,
        )