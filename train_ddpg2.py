import os
import sys
import json
import argparse

import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import random

from model import rl_utils
from model.ddpg import DDPG
import matplotlib.pyplot as plt
import pickle
# teacher checkpoint_load + env train + env eval
# python train_ddpg.py --policy_type "mlp" --gamma 0.98 --tensordir "runs/10case20e_gamma98_load2" --des "10case20e_gamma98_load2" --load_ckp > 0428_10case20e_gamma98_load2.log 2>&1 &
# python train_ddpg.py --policy_type "mlp" --gamma 0.98 --tensordir "runs/10case20e_gamma98_xload" --des "10case20e_gamma98_xload" > 0428_10case20e_gamma98_xload.log 2>&1 &
# python train_ddpg.py --policy_type "mlp" --gamma 0.0 --tensordir "runs/10case20e_gamma0_load" --des "10case20e_gamma0_load" --load_ckp > 0428_10case20e_gamma0_load.log 2>&1 &
# python train_ddpg.py --policy_type "mlp" --gamma 0.0 --tensordir "runs/10case20e_gamma0_xload" --des "10case20e_gamma0_xload" > 0428_10case20e_gamma0_xload.log 2>&1 &
# python train_ddpg.py --policy_type "mlp" --gamma 0.0 --tensordir "runs/10case20e_gamma0_xloadzs" --des "10case20e_gamma0_xloadzs" > 0428_10case20e_gamma0_xloadzs.log 2>&1 &
# python train_ddpg.py --policy_type "mlp" --gamma 0.0 --tensordir "runs/10case30e_gamma0_xloadzs_10eval" --des "10case30e_gamma0_xloadzs_10eval" -e 30 > 0428_10case30e_gamma0_xloadzs_10eval.log 2>&1 &
# python train_ddpg.py --policy_type "mlp" --gamma 0.0 --tensordir "runs/10case30e_gamma0_xloadzs_10eval_half" --des "10case30e_gamma0_xloadzs_10eval_half" -e 30 > 0428_10case30e_gamma0_xloadzs_10eval_half.log 2>&1 &
# python train_ddpg.py --policy_type "mlp" --gamma 0.98 --tensordir "runs/10case30e_gamma98_xloadzs_10eval_square_teach" --des "10case30e_gamma98_xloadzs_10eval_square_teach" -e 30 > 0429_10case30e_gamma98_xloadzs_10eval_square_teach.log 2>&1 &
# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="Process command line arguments for training configuration.")
    
# 添加参数
parser.add_argument('--policy_type', type=str, help='Type of policy, e.g., "mlp"')
parser.add_argument('--gamma', type=float, help='Discount factor for the reinforcement learning algorithm')
parser.add_argument('--tensordir', type=str, help='Directory path for TensorBoard logs')
parser.add_argument('--des', type=str, help='Description for the run')
parser.add_argument("--load_ckp",action="store_true",help="Load checkpoint.")
parser.add_argument("-e", "--epoch_num", type=int, default=20,
                        help="epoch_num (default=20)")
parser.add_argument("-c", "--case_num", type=int, default=10,
                        help="case_num (default=10)")
# 解析命令行参数
args = parser.parse_args()

# 使用参数
print(f"Policy Type: {args.policy_type}")
print(f"Gamma: {args.gamma}")
print(f"TensorBoard Directory: {args.tensordir}")
print(f"Description: {args.des}")
print(f"Load Checkpoint: {args.load_ckp}")
print(f"Epoch Num: {args.epoch_num}")
print(f"Case Num: {args.case_num}")
pretrain_epoch = 5
actor_lr = 1e-3
critic_lr = 3e-3
num_episodes = args.case_num
hidden_dim = 64
gamma = args.gamma
tau = 0.005  # 软更新参数
buffer_size = 100000
minimal_size = 1000
batch_size = 64
sigma = 0.01  # 高斯噪声标准差

state_dim = 28*126
action_dim = 28*9
action_bound = 2  # 动作最大值
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
#vis
if not os.path.exists(os.path.join(r"/home/mjy/teeth/RL",args.tensordir)):
    os.makedirs(os.path.join(r"/home/mjy/teeth/RL",args.tensordir))
writer = SummaryWriter(os.path.join(r"/home/mjy/teeth/RL",args.tensordir))

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
#------------------# 加载teacher数据到replaybuffer
teacher_dataroot = r"/home/mjy/teeth/teacher"
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
            # qvalues = transition_dict['q_values']
            # qvalues[-1]=rewards[-1]
            # for i in range(len(states)-2,-1,-1):
            #     qvalues[i] = rewards[i] + gamma * qvalues[i+1] *(1-dones[i])
            for i in range(len(states)):
                state = np.array(states[i]).flatten()
                action = np.array(actions[i]).flatten()
                action_stats.append(actions[i])
                
                reward = rewards[i]
                # qv = qvalues[i]
                next_state = np.array(next_states[i]).flatten()
                done = dones[i]
                if teacher_cnt<5:
                    replay_buffer.add(state, action, reward, next_state, done)
    teacher_cnt += 1
action_stats = np.concatenate(action_stats,axis=0)
ac_mean = np.mean(action_stats,axis=0)
ac_std = np.std(action_stats,axis=0)
print(f"action mean:{ac_mean},action std:{ac_std}")
#------------------#
ckpt_path = r"/home/mjy/teeth/RL/checkpt/best_DDPGattention_eval_pretrainq.pth" if args.policy_type=="attention" else r"/home/mjy/teeth/RL/checkpt/best_DDPGmlp_eval_pretrainq.pth"
agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device,mean,std,policytype=args.policy_type)
if args.load_ckp:
    agent.load(ckpt_path,equal=True)
# pretrain 
# for i in range(pretrain_epoch):
#     b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
#     transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
#     agent.update(transition_dict)
bvh_folder =r"/home/mjy/teeth/RL/datasets/datasets/teeth10k/train"
return_list = rl_utils.my_train_off_policy_agent(bvh_folder, agent, num_episodes,
                                              replay_buffer, minimal_size,
                                              batch_size,writer,explore_cnt=args.epoch_num,des=args.des)
writer.close()

# mean
mean_return = np.max(return_list)
print(f"max return:{mean_return}")

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(args.des))
plt.savefig(r"/home/mjy/teeth/RL/ddpg_env_train_"+args.des+".png")


mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('smooth DDPG on {}'.format(args.des))
plt.savefig(r"/home/mjy/teeth/RL/ddpg_smooth_env_train_"+args.des+".png")