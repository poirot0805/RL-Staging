import os
import json
import gym
import numpy as np
import torch
import random

from model import rl_utils
from model.ddpg import DDPG
import matplotlib.pyplot as plt
import pickle
pretrain_epoch = 5
actor_lr = 3e-4
critic_lr = 3e-3
num_episodes = 1800
hidden_dim = 64
gamma = 0.98
tau = 0.005  # 软更新参数
buffer_size = 1000000
minimal_size = 1000
batch_size = 64
sigma = 0.01  # 高斯噪声标准差

state_dim = 28*126
action_dim = 28*9
action_bound = 1  # 动作最大值
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device(
    "cpu")
# get mean and std of the state
stats_path = r"/home/mjy/teeth/RL/train_stats_context.pkl"
with open(stats_path, "rb") as fh:
    train_stats = pickle.load(fh)
mean = train_stats["mean"]  # (28,9)
std = train_stats["std"]    # (28,9)
assert mean.shape == (28, 9)
assert std.shape == (28, 9)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
# 加载teacher数据到replaybuffer
teacher_dataroot = r"/datasets/mjy/teacher_data"
teacher_cnt = 0
teachers=[]
for f in os.listdir(teacher_dataroot):
    if f.endswith(".json"):
        teachers.append(f)
teachers.sort()
for f in teachers:
    if teacher_cnt >= 900:
        break
    teacher_cnt += 1
    with open(os.path.join(teacher_dataroot, f)) as fh:
            data = json.load(fh)
            transition_dict = data["transition"]
            states = transition_dict["states"]
            actions = transition_dict["actions"]
            rewards = transition_dict["rewards"]
            next_states = transition_dict["next_states"]
            dones = transition_dict["dones"]
            for i in range(len(states)):
                state = np.array(states[i]).flatten()
                action = np.array(actions[i]).flatten()
                reward = rewards[i]
                next_state = np.array(next_states[i]).flatten()
                done = dones[i]
                replay_buffer.add(state, action, reward, next_state, done)


agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device,mean,std)
# pretrain 
for i in range(pretrain_epoch):
    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
    transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
    agent.update(transition_dict)
bvh_folder =r"/home/mjy/teeth/datasets/teeth10k/train"
return_list = rl_utils.my_train_off_policy_agent(bvh_folder, agent, num_episodes,
                                              replay_buffer, minimal_size,
                                              batch_size)
# mean
mean_return = np.mean(return_list)
print(f"mean return:{mean_return}")

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format("OrthoStaging"))
plt.savefig(r"/home/mjy/teeth/RL/ddpg.png")


mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format("OrthoStaging"))
plt.savefig(r"/home/mjy/teeth/RL/ddpg_smooth.png")