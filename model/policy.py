import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.input_dim = state_dim//28
        self.fc_layers=nn.ModuleList()
        self.fc_layers.append(nn.Linear(self.input_dim, hidden_dim))
        self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc_low = nn.Linear(hidden_dim, hidden_dim//4)
        self.fc_out = nn.Linear(hidden_dim//4*28, action_dim)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值
        self.activation = nn.PReLU()
    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # return torch.tanh(self.fc2(x)) * self.action_bound
        x_ = x.reshape(-1,28,126)
        for i, layer in enumerate(self.fc_layers):
            if i==0:
                x_ = self.activation(layer(x_))
            else:
                x_ = self.activation(x_+layer(x_))
        x_ = self.activation(self.fc_low(x_))
        res = x_.flatten(start_dim=1)
        return torch.tanh(self.fc_out(res)) * self.action_bound

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)

        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob.sum(-1)
        # normal_sample = dist.rsample()  # rsample()是重参数化采样
        # log_prob = dist.log_prob(normal_sample).sum(axis=-1)
        
        # action = torch.tanh(normal_sample)
        # # 计算tanh_normal分布的对数概率密度
        # log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        # action = action * self.action_bound
        return pi_action, logp_pi
