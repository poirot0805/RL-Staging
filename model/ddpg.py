import random
import gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import model.rl_utils

from model.policy import PolicyNet,PolicyNetAttention
from model.valuenet import QValueNet

class DDPG:
    ''' DDPG算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device,mean,std,
                 policytype = 'mlp',valuetype = 'mlp'):
        if policytype == 'mlp':
            self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
            self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        elif policytype == 'attention':
            self.actor = PolicyNetAttention(state_dim, hidden_dim, action_dim, action_bound).to(device)
            self.target_actor = PolicyNetAttention(state_dim, hidden_dim, action_dim, action_bound).to(device)
        if valuetype == 'mlp':
            self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
            self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device
        self.mean = torch.tensor(mean, dtype=torch.float, device=device)
        self.std = torch.tensor(std, dtype=torch.float, device=device)
    def save(self):
        model_dict = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }
        return model_dict

    def take_action(self, state,eval=False):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        state = self.zscore(state)
        
        action = self.actor(state).detach().cpu().numpy()
        if eval:
            return action
        # 给动作添加噪声，增加探索
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action
    
    def zscore(self,state):
        # zscore
        shape = state.shape
        state = state.reshape(-1,28,126)
        pose = state[...,:9]
        tgt = state[...,9:18]
        shapecode = state[...,18:]
        pose = (pose - self.mean) / self.std
        tgt = (tgt - self.mean) / self.std
        state = torch.cat([pose,tgt,shapecode],dim=-1).reshape(-1,28*126)
        assert state.shape == shape
        return state
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        gt_qvalues = torch.tensor(transition_dict['q_values'], dtype=torch.float).view(-1, 1).to(self.device)
        # zscore
        states = self.zscore(states)
        next_states = self.zscore(next_states)
        
        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        pred_q_values = self.critic(states, actions)
        critic_loss = torch.mean(F.mse_loss(pred_q_values, q_targets))
        qv_loss_l1 = torch.mean(torch.abs(pred_q_values.detach()-gt_qvalues.detach()))  # SHOW
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pred_actions = self.actor(states)
        actor_loss = -torch.mean(self.critic(states, pred_actions))
        action_loss_l1 = torch.mean(torch.abs(pred_actions.detach()-actions.detach()))  # SHOW
        action_loss_l2 = torch.mean(torch.sqrt(torch.sum(torch.square(pred_actions.detach()-actions.detach()),dim=-1))) # SHOW
        print('qv_loss_l1:',qv_loss_l1.item(),'action_loss_l1:',action_loss_l1.item(),'action_loss_l2:',action_loss_l2.item())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络
    
    def pretrain_policy(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        gt_qvalues = torch.tensor(transition_dict['q_values'], dtype=torch.float).view(-1, 1).to(self.device)
        # zscore
        states = self.zscore(states)
        next_states = self.zscore(next_states)
        
        self.critic_optimizer.zero_grad()       
        self.critic.train() 
        pred_q_values = self.critic(states, actions)
        qv_loss_l1 = torch.mean(torch.abs(pred_q_values-gt_qvalues))  # SHOW
        qv_loss_l1.backward()
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        self.actor.train()
        pred_actions = self.actor(states)
        action_loss_l1 = torch.mean(torch.abs(pred_actions-actions))  # SHOW
        action_loss_l1.backward()
        self.actor_optimizer.step()

        return qv_loss_l1.item(),action_loss_l1.item()
    
    def eval_pretrain_policy(self,transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        gt_qvalues = torch.tensor(transition_dict['q_values'], dtype=torch.float).view(-1, 1).to(self.device)
        # zscore
        states = self.zscore(states)
        next_states = self.zscore(next_states)
        
        with torch.no_grad(): 
            self.critic.eval()
            pred_q_values = self.critic(states, actions)
            qv_loss_l1 = torch.mean(torch.abs(pred_q_values-gt_qvalues))  # SHOW
            
            self.actor.eval()
            pred_actions = self.actor(states)
            action_loss_l1 = torch.mean(torch.abs(pred_actions-actions))  # SHOW


        return qv_loss_l1.item(),action_loss_l1.item()