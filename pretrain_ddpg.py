import os
import sys
import json
import gym
import numpy as np
import torch
import random

from model import rl_utils
from model.ddpg import DDPG
import matplotlib.pyplot as plt
import pickle
# 重新生成teacher数据，只包含trans一项reward
policytype = sys.argv[1]
print(f"policy type:{policytype}")
pretrain_epoch = 50
actor_lr = 1e-3
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

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device(
    "cpu")
# get mean and std of the state
stats_path = r"/home/mjy/teeth/RL/train_stats_context.pkl"
with open(stats_path, "rb") as fh:
    train_stats = pickle.load(fh)
mean = train_stats["mean"]  # (28,9)
std = train_stats["std"]    # (28,9)
assert mean.shape == (28, 9)
assert std.shape == (28, 9)
replay_buffer = []#rl_utils.ReplayBuffer(buffer_size)
eval_buffer = []
# 加载teacher数据到replaybuffer
teacher_dataroot = r"/datasets/mjy/teacher_data_onereward"
teacher_cnt = 0
teachers=[]
for f in os.listdir(teacher_dataroot):
    if f.endswith(".json"):
        teachers.append(f)
teachers.sort()
print("loading teacher data")
action_stats=[]
for f in teachers:
    with open(os.path.join(teacher_dataroot, f)) as fh:
            data = json.load(fh)
            transition_dict = data["transition"]
            states = transition_dict["states"]
            actions = transition_dict["actions"]
            rewards = transition_dict["rewards"]
            next_states = transition_dict["next_states"]
            dones = transition_dict["dones"]
            qvalues = transition_dict['q_values']
            action_stats.append(np.mean(np.array(actions),axis=(0,1)))
            for i in range(len(states)):
                state = np.array(states[i]).flatten()
                action = np.array(actions[i]).flatten()
                reward = rewards[i]
                qv = qvalues[i]
                next_state = np.array(next_states[i]).flatten()
                done = dones[i]
                if teacher_cnt<6400:
                    replay_buffer.append((state, action, reward, next_state, done, qv))
                else:
                    eval_buffer.append((state, action, reward, next_state, done, qv))
    teacher_cnt += 1
action_stats = np.array(action_stats)
print("action stats:",np.mean(action_stats,axis=0),np.std(action_stats,axis=0))
agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device,mean,std,policytype=policytype)

return_list_a=[]
return_list_c=[]
eval_list_a=[]
eval_list_c=[]
total_size = len(replay_buffer)
eval_size = len(eval_buffer)
print("total size",total_size)
     
#-------------------------------------------------加载eval结束--------------------------------------------------------------
best_return = -np.inf
best_eval_return = -np.inf
agent_name = type(agent).__name__
for i in range(pretrain_epoch):
    print(f"pretrain epoch:{i}")
    random.shuffle(replay_buffer)
    temp_a=0
    temp_c=0
    for iter in range(total_size//batch_size):
        start_idx = iter*batch_size
        end_idx = min((iter+1)*batch_size, total_size)
        
        transitions = replay_buffer[start_idx:end_idx]
        state, action, reward, next_state, done, qv = zip(*transitions)
        transition_dict = {'states': np.array(state), 'actions': np.array(action), 'next_states': np.array(next_state), 'rewards': reward, 'dones': done, 'q_values': qv}
        c_loss,a_loss=agent.pretrain_policy(transition_dict)
        temp_a+=a_loss
        temp_c+=c_loss
    return_list_a.append(temp_a/(total_size//batch_size))
    return_list_c.append(temp_c/(total_size//batch_size))
    print(f"epoch:{i},actor loss:{return_list_a[-1]},critic loss:{return_list_c[-1]}")
    
    tmp_mean = return_list_a[-1]
    if tmp_mean> best_return:
            best_return = tmp_mean
            print(f"save best model with return:{best_return} in epoch:{i}")
            rl_utils.save_checkpoint(r"/home/mjy/teeth/RL/checkpt/",agent,i,i,suffix=f"best_{agent_name}{policytype}_pretrain.pth")
    sum_eval_a=0
    sum_eval_c=0
    for iter in range(eval_size//batch_size):
        start_idx = iter*batch_size
        end_idx = min((iter+1)*batch_size, eval_size)
        transitions = eval_buffer[start_idx:end_idx]
        state, action, reward, next_state, done, qv = zip(*transitions)
        transition_dict = {'states': np.array(state), 'actions': np.array(action), 'next_states': np.array(next_state), 'rewards': reward, 'dones': done, 'q_values': qv}
        c_loss,a_loss=agent.eval_pretrain_policy(transition_dict)
        sum_eval_a+=a_loss
        sum_eval_c+=c_loss
    eval_list_a.append(sum_eval_a/(eval_size//batch_size))
    eval_list_c.append(sum_eval_c/(eval_size//batch_size))
    print(f"[eval] epoch:{i},actor loss:{eval_list_a[-1]},critic loss:{eval_list_c[-1]}")
    tmp_mean = eval_list_a[-1]
    if tmp_mean> best_eval_return:
            best_eval_return = tmp_mean
            print(f"save best model with return:{best_eval_return} in epoch:{i}")
            rl_utils.save_checkpoint(r"/home/mjy/teeth/RL/checkpt/",agent,i,i,suffix=f"best_{agent_name}{policytype}_eval_pretrain.pth")


episodes_list = list(range(len(return_list_a)))
plt.figure()
plt.plot(episodes_list, return_list_a, label='train actor loss')
plt.plot(episodes_list, eval_list_a, label='eval actor loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('actor loss')
plt.title('DDPG on {}'.format("OrthoStaging"))
plt.savefig(r"/home/mjy/teeth/RL"+f"/ddpg_{agent_name}{policytype}_actor.png")
plt.close()

plt.figure()
plt.plot(episodes_list, return_list_c, label='train critic loss')
plt.plot(episodes_list, eval_list_c, label='eval critic loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('critic loss')
plt.title('DDPG on {}'.format("OrthoStaging"))
plt.savefig(r"/home/mjy/teeth/RL"+f"/ddpg_{agent_name}{policytype}_critic.png")
plt.close()