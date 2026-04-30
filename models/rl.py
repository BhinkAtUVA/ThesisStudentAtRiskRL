import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
            
class ActorNetwork(nn.Module):
    def __init__(self, **kwargs):
        super(ActorNetwork, self).__init__()
        # self.args = args
        self.state_dim = kwargs['state_dim']
        # self.actor = nn.Linear(self.state_dim, 2)
        self.actor  = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            # nn.Linear(32, 2),
            nn.Linear(32, 1),
        )
        self.actor.apply(init_weights)
        # nn.init.xavier_uniform_(self.actor.weight)
        
    def forward(self, x):
        # action_probs = F.softmax(self.actor(x), dim=-1)
        action_probs = F.sigmoid(self.actor(x))
        return action_probs


class CriticNetwork(nn.Module):
    def __init__(self, **kwargs):
        super(CriticNetwork, self).__init__()
        # self.args = args
        self.state_dim = kwargs['state_dim']
        self.hdim = kwargs['hdim']

        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, self.hdim),
            nn.Tanh(),
            nn.Linear(self.hdim, 1)
        )
        nn.init.xavier_uniform_(self.critic[0].weight)
        nn.init.xavier_uniform_(self.critic[2].weight)

    def forward(self, x):
        out = torch.sigmoid(self.critic(x))
        # out = torch.mean(out)
        return out
    
class PolicyNetwork(nn.Module):
    def __init__(self, **kwargs):
        super(PolicyNetwork, self).__init__()
        self.actor = ActorNetwork(state_dim=kwargs['state_dim'])
        self.critic = CriticNetwork(state_dim=kwargs['state_dim'], hdim=kwargs['hdim'])

    def forward(self, x):
        exp_reward = self.critic(x)
        action_probs = self.actor(x)
        
        action_probs = action_probs.squeeze(-1)
        exp_reward = torch.mean(exp_reward, dim=1)
        
        return action_probs, exp_reward


def get_loss_fn(task_type):
    if task_type == 'classification':
        return nn.CrossEntropyLoss()
    elif task_type == 'regression':
        return nn.MSELoss()
    else:
        NotImplementedError

def sample_action(action_probs, n, device, random=False, algorithm="with_replacement"):
    if algorithm == "static":
        # print("static")
        return sample_static_action(action_probs, n, device, random=random)
    elif algorithm == "with_replacement":
        # print("with_replacement")
        return sample_action_with_replacement(action_probs, n, device, random=random)
    elif algorithm == "without_replacement":
        # print("without_replacement")
        return sample_action_without_replacement(action_probs, n, device, random=random)
    else:
        NotImplementedError

def sample_action_with_replacement(action_probs, n, device, random=False):
    # with replacement  
    m = Categorical(action_probs)  
    if random:
        action = torch.randint(0, action_probs.shape[1], (n, action_probs.shape[0])).to(device)
    else:
        action = m.sample((n,))
    
    log_prob = m.log_prob(action).sum(dim=0)
    # from IPython import embed; embed(); exit()
    return action.T, log_prob

def sample_action_without_replacement(action_probs, n, device, random=False):
    # multinomial sampling without replacement
    # sample_weights = action_probs[:, :, 1]
    sample_weights = action_probs
    if random:
        action = torch.empty((action_probs.shape[0], n), dtype=torch.long)
        for i in range(action_probs.shape[0]):
            action[i] = torch.randperm(action_probs.shape[1])[:n]  
        action = action.to(device)
    else:
        action = torch.multinomial(sample_weights, n)
    log_prob = torch.log(sample_weights.gather(1, action))
    log_prob = log_prob.mean(dim=1)
    return action, log_prob


def sample_static_action(action_probs, n, device, random=False):
    # action_sort = action_probs[:, :, 1].sort(descending=True)
    if random:
        action = torch.empty((action_probs.shape[0], n), dtype=torch.long)
        for i in range(action_probs.shape[0]):
            action[i] = torch.randperm(action_probs.shape[1])[:n]  
        action = action.to(device)
        log_prob = torch.gather(action_probs, 1, action)
    else:
        action_sort = action_probs.sort(descending=True)
        action = action_sort.indices[:, :n]
        log_prob = torch.log(action_sort.values[:, :n])
    log_prob = torch.mean(log_prob, dim=1)
    return action, log_prob


def select_from_action(action, x):
    return x[torch.arange(action.shape[0]).unsqueeze(1), action]


""" class PolicyNetwork(nn.Module):
    def __init__(self, **kwargs):
        super(PolicyNetwork, self).__init__()
        # self.args = args
        self.actor = ActorNetwork(state_dim=kwargs['state_dim'])
        self.critic = CriticNetwork(state_dim=kwargs['state_dim'], hdim=kwargs['hdim'])
        self.task_model = kwargs['task_model']
        self.learning_rate = kwargs['learning_rate']
        self.device = kwargs['device']
        self.task_type = kwargs['task_type']
        self.min_clip = kwargs['min_clip']
        self.max_clip = kwargs['max_clip']
        self.sample_algorithm = kwargs.get('sample_algorithm', 'with_replacement')
        self.no_autoencoder = kwargs.get('no_autoencoder', False)
        
        try:
            self.task_optim = optim.AdamW(self.task_model.parameters(), lr=self.learning_rate)
        except:
            self.task_optim = None
        self.loss_fn = get_loss_fn(self.task_type)

        self.saved_actions = []
        self.rewards = []

    def forward(self, batch_x):
        if self.no_autoencoder:
            batch_rep = batch_x
        else:
            batch_rep = self.task_model.base_network(batch_x).detach()
        
        # batch_size, bag_size, embedding_size = batch_rep.shape
        # batch_rep = batch_rep.view(batch_size * bag_size, embedding_size)

        exp_reward = self.critic(batch_rep)
        action_probs = self.actor(batch_rep)
        action_probs = action_probs.squeeze(-1)
        # action_probs = action_probs.view(batch_size, bag_size)
        # exp_reward = exp_reward.view(batch_size, bag_size)
        # batch_rep = batch_rep.view(batch_size, bag_size, embedding_size)

        # action_probs = action_probs[:, :, 1]
        # action_probs = F.softmax(action_probs, dim=-1)
        # from IPython import embed; embed(); exit()

        exp_reward = torch.mean(exp_reward, dim=1)
        return action_probs, batch_rep, exp_reward

    def reset_reward_action(self):
        self.saved_actions, self.rewards = [], []

    def normalize_rewards(self, eps=1e-5):
        R_mean = np.mean(self.rewards)
        R_std = np.std(self.rewards)
        for i, r in enumerate(self.rewards):
            self.rewards[i] = float((r - R_mean) / (R_std + eps))

    def select_from_dataloader(self, dataloader, bag_size, random=False):
        with torch.no_grad():
            data = []
            for batch_x, batch_y, indices, instance_labels in dataloader:
                batch_x = batch_x.to(self.device)
                # select batch_x
                action_probs, _, _ = self.forward(batch_x)
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
        self.task_model.train()
        batch_out = self.task_model(batch_x)
        batch_loss = self.loss_fn(batch_out.squeeze(), batch_y.squeeze())
        self.task_optim.zero_grad()
        batch_loss.backward()
        self.task_optim.step()
        return batch_loss.item()

    def eval_minibatch(self, batch_x, batch_y):
        self.task_model.eval()
        batch_out = self.task_model(batch_x)
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

    def predict(self, data):
        with torch.no_grad():
            prob_ys = []
            for batch_x in data:
                batch_x = batch_x.to(self.device)
                pred_out = self.task_model(batch_x)
                prob_y = torch.softmax(pred_out, dim=1)
                prob_ys.append(prob_y.detach().cpu())
            prob_Y = torch.cat(prob_ys, dim=0)
        return prob_Y
    
    def ensemble_predict(self, pool_data):
        preds_pool = []
        for data in pool_data:
            _, _, preds, labels = self.compute_reward(data)
            preds_pool.append(preds)
        preds_pool = torch.stack(preds_pool, dim=2).mean(dim=2)
        return preds_pool, labels """