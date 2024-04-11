import os
import json
import gym
import numpy as np
import torch
import random

from model import rl_utils
from model.sac import SACContinuous
import matplotlib.pyplot as plt

# env_name = 'Pendulum-v0'
# env = gym.make(env_name)
pretrain_epoch = 100
state_dim = 28*126
action_dim = 28*9
action_bound = 1  # 动作最大值
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

actor_lr = 3e-4
critic_lr = 3e-3
alpha_lr = 3e-4
num_episodes = 1800
hidden_dim = 1024
gamma = 0.99
tau = 0.005  # 软更新参数
buffer_size = 1000000
minimal_size = 1000
batch_size = 64
target_entropy = -action_dim  # 目标熵
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device(
    "cpu")

replay_buffer = rl_utils.ReplayBuffer(buffer_size)
# 加载teacher数据到replaybuffer
teacher_dataroot = r"/home/mjy/teeth/RL/teacher_data"
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

agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound,
                      actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                      gamma, device)
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
plt.title('SAC on {}'.format("OrthoStaging"))
plt.savefig(r"/home/mjy/teeth/RL/sac.png")


mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('SAC on {}'.format("OrthoStaging"))
plt.savefig(r"/home/mjy/teeth/RL/sac_smooth.png")