import numpy as np
from sklearn.metrics import f1_score, r2_score, roc_auc_score
from torch import optim
import torch

from models.full import NetworkContainer, RLMILBase
from models.rl import get_loss_fn, sample_action, select_from_action

class Trainer:
    def get_model_constructor() -> type[NetworkContainer]:
        return NetworkContainer

class RLMILTrainer(Trainer):
    def __init__(self, model: RLMILBase, **kwargs):
        # self.args = args
        self.model = model
        self.learning_rate = kwargs['learning_rate']
        self.device = kwargs['device']
        self.task_type = kwargs['task_type']
        self.min_clip = kwargs['min_clip']
        self.max_clip = kwargs['max_clip']
        self.sample_algorithm = kwargs.get('sample_algorithm', 'with_replacement')
        
        try:
            self.task_optim = optim.AdamW(self.model.task_model.parameters(), lr=self.learning_rate)
        except:
            self.task_optim = None
        self.loss_fn = get_loss_fn(self.task_type)

        self.saved_actions = []
        self.rewards = []

    def get_model_constructor():
        return RLMILBase
        
    def select_from_dataloader(self, dataloader, bag_size, random=False):
        with torch.no_grad():
            data = []
            for batch_x, batch_y, indices, instance_labels in dataloader:
                batch_x = batch_x.to(self.device)
                # select batch_x
                action_probs, _, _ = self.model.action(batch_x)
                action, _ = sample_action(action_probs, bag_size, self.device, random=random, algorithm=self.sample_algorithm)
                batch_x = select_from_action(action, batch_x)
                batch_x = batch_x.cpu()
                data.append((batch_x, batch_y, indices, instance_labels))
        return data

    def compute_reward(self, eval_data):
        with torch.no_grad():
            data_ys, pred_ys, losses, prob_ys = [], [], [], []
            for batch_x, batch_y, _, _ in eval_data:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_out, loss = self.eval_minibatch(batch_x, batch_y)
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
        return reward, np.mean(losses), prob_Y, data_Y

    def compute_metrics_and_details(self, eval_data):
        with torch.no_grad():
            data_ys, pred_ys, losses, prob_ys = [], [], [], []
            for batch_x, batch_y, _, _ in eval_data:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_out, loss = self.eval_minibatch(batch_x, batch_y)
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
            metrics = {'loss': np.mean(losses)}
            if self.task_type == 'classification':
                f1_macro = f1_score(data_Y.data, pred_Y.data, average='macro')
                f1_micro = f1_score(data_Y.data, pred_Y.data, average='micro')
                if prob_Y.shape[1] == 2:
                    auc = roc_auc_score(data_Y.data, prob_Y.data[:, 1], average='macro')
                else:
                    auc = roc_auc_score(data_Y.data, prob_Y.data, average='macro', multi_class='ovr')
                metrics.update({
                    'f1': f1_macro,
                    'f1_micro': f1_micro,
                    'auc': auc,
                })
            elif self.task_type == 'regression':   
                reward = r2_score(data_Y.data, pred_Y.data)
                metrics.update({
                    'r2': reward,
                })
        return metrics, prob_Y.tolist(), data_Y.tolist(), pred_Y.tolist()
    
    def train_minibatch(self, batch_x, batch_y):
        self.model.task_model.train()
        batch_out = self.model.task_model(batch_x)
        batch_loss = self.loss_fn(batch_out.squeeze(), batch_y.squeeze())
        self.task_optim.zero_grad()
        batch_loss.backward()
        self.task_optim.step()
        return batch_loss.item()

    def eval_minibatch(self, batch_x, batch_y):
        self.model.task_model.eval()
        batch_out = self.model.task_model(batch_x)
        batch_loss = self.loss_fn(batch_out.squeeze(), batch_y.squeeze())
        return batch_out, batch_loss.item()

    def create_pool_data(self, dataloader, bag_size, pool_size, random=False):
        pool = []
        for _ in range(pool_size):
            data = self.select_from_dataloader(dataloader, bag_size, random=random)
            pool.append(data)
        return pool

    def expected_reward_loss(self, pool_data, average='macro', verbos=False):
        reward_pool, loss_pool, preds_pool = [], [], []
        for data in pool_data:
            reward, loss, preds, labels = self.compute_reward(data)
            reward_pool.append(reward)
            loss_pool.append(loss)
            preds_pool.append(preds)
        if self.task_type == 'classification':
            preds_pool = torch.stack(preds_pool, dim=2).mean(dim=2).argmax(dim=1)
            ensemble_reward = f1_score(labels.data, preds_pool.data, average=average)
        elif self.task_type == 'regression':
            preds_pool = torch.stack(preds_pool, dim=2).mean(dim=2).squeeze()
            preds_pool = torch.clamp(preds_pool, min=self.min_clip, max=self.max_clip)
            ensemble_reward = r2_score(labels.data, preds_pool.data)
        mean_reward = np.mean(reward_pool)
        mean_loss = np.mean(loss_pool)
        return mean_reward, mean_loss, ensemble_reward
    
    def predict_pool(self, pool_data):
        probs_pool = []
        for data in pool_data:
            prob_Y = self.predict(data)
            probs_pool.append(prob_Y)
        preds_pool = torch.stack(probs_pool, dim=2).mean(dim=2).argmax(dim=1)
        return preds_pool