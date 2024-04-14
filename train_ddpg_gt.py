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

pretrain_epoch = 100
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
# 加载teacher数据到replaybuffer
teacher_dataroot = r"/datasets/mjy/teacher_data"
teacher_cnt = 0
teachers=[]
for f in os.listdir(teacher_dataroot):
    if f.endswith(".json"):
        teachers.append(f)
teachers.sort()
for f in teachers:
    if teacher_cnt >= 6400:
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
                replay_buffer.append((state, action, reward, next_state, done))


agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device,mean,std)

return_list=[]
total_size = len(replay_buffer)
print("total size",total_size)
# -------------------------------------------------------------------加载eval-------------------------------------------------------
bvh_folder = r"/home/mjy/teeth/datasets/teeth10k/train"
num_episodes = 50


data_root = [os.path.join(bvh_folder, "complete"), os.path.join(bvh_folder, "incomplete")]

json_root=r"/home/mjy/teeth"# r"E:\PROJECTS\tooth_spatial_temporal_transformer"
mode=['test','train','val']
remove_dict={}
for subdir in mode:
            json_path=os.path.join(json_root,f"removeStatus3_{subdir}.json")
            with open(json_path) as fh:
                remove_dict.update(json.load(fh))
    # tooth data使用了json格式作为数据源
bvh_files = []

PROJECT_DIR = r"/home/mjy/teeth"
error_cases=['datasets/teeth10k/train/complete/C01002722632.json', 'datasets/teeth10k/train/complete/C01002722812.json', 'datasets/teeth10k/train/complete/C01002724937.json', 'datasets/teeth10k/train/complete/C01002726883.json', 'datasets/teeth10k/train/complete/C01002728672.json', 'datasets/teeth10k/train/complete/C01002737908.json', 'datasets/teeth10k/train/complete/C01002739797.json', 'datasets/teeth10k/train/complete/C01002739809.json', 'datasets/teeth10k/train/complete/C01002740294.json', 'datasets/teeth10k/train/complete/C01002742285.json', 'datasets/teeth10k/train/complete/C01002742814.json', 'datasets/teeth10k/train/complete/C01002743376.json', 'datasets/teeth10k/train/complete/C01002748270.json', 'datasets/teeth10k/train/complete/C01002752736.json', 'datasets/teeth10k/train/complete/C01002753894.json', 'datasets/teeth10k/train/complete/C01002757078.json', 'datasets/teeth10k/train/complete/C01002760218.json', 'datasets/teeth10k/train/complete/C01002760285.json', 'datasets/teeth10k/train/complete/C01002762513.json', 'datasets/teeth10k/train/complete/C01002764234.json', 'datasets/teeth10k/train/complete/C01002770466.json', 'datasets/teeth10k/train/complete/C01002772985.json', 'datasets/teeth10k/train/complete/C01002774123.json', 'datasets/teeth10k/train/complete/C01002774594.json', 'datasets/teeth10k/train/complete/C01002775269.json', 'datasets/teeth10k/train/complete/C01002784742.json', 'datasets/teeth10k/train/complete/C01002791706.json', 'datasets/teeth10k/train/complete/C01002792886.json', 'datasets/teeth10k/train/complete/C01002796891.json', 'datasets/teeth10k/train/complete/C01002800505.json', 'datasets/teeth10k/train/complete/C01002807805.json', 'datasets/teeth10k/train/complete/C01002809896.json', 'datasets/teeth10k/train/complete/C01002810292.json', 'datasets/teeth10k/train/complete/C01002811406.json', 'datasets/teeth10k/train/complete/C01002811855.json', 'datasets/teeth10k/train/complete/C01002812430.json', 'datasets/teeth10k/train/complete/C01002817413.json', 'datasets/teeth10k/train/complete/C01002818931.json', 'datasets/teeth10k/train/complete/C01002828437.json', 'datasets/teeth10k/train/complete/C01002828482.json', 'datasets/teeth10k/train/complete/C01002837246.json', 'datasets/teeth10k/train/complete/C01002838124.json', 'datasets/teeth10k/train/complete/C01002838337.json', 'datasets/teeth10k/train/complete/C01002840587.json', 'datasets/teeth10k/train/complete/C01002844772.json', 'datasets/teeth10k/train/complete/C01002849621.json', 'datasets/teeth10k/train/incomplete/C01002722823.json', 'datasets/teeth10k/train/incomplete/C01002725118.json', 'datasets/teeth10k/train/incomplete/C01002725736.json', 'datasets/teeth10k/train/incomplete/C01002727154.json', 'datasets/teeth10k/train/incomplete/C01002727817.json', 'datasets/teeth10k/train/incomplete/C01002728762.json', 'datasets/teeth10k/train/incomplete/C01002735973.json', 'datasets/teeth10k/train/incomplete/C01002736749.json', 'datasets/teeth10k/train/incomplete/C01002736806.json', 'datasets/teeth10k/train/incomplete/C01002737627.json', 'datasets/teeth10k/train/incomplete/C01002738954.json', 'datasets/teeth10k/train/incomplete/C01002742982.json', 'datasets/teeth10k/train/incomplete/C01002744298.json', 'datasets/teeth10k/train/incomplete/C01002744513.json', 'datasets/teeth10k/train/incomplete/C01002744715.json', 'datasets/teeth10k/train/incomplete/C01002745255.json', 'datasets/teeth10k/train/incomplete/C01002746492.json', 'datasets/teeth10k/train/incomplete/C01002746762.json', 'datasets/teeth10k/train/incomplete/C01002746784.json', 'datasets/teeth10k/train/incomplete/C01002747392.json', 'datasets/teeth10k/train/incomplete/C01002748258.json', 'datasets/teeth10k/train/incomplete/C01002750688.json', 'datasets/teeth10k/train/incomplete/C01002751746.json', 'datasets/teeth10k/train/incomplete/C01002752343.json', 'datasets/teeth10k/train/incomplete/C01002752398.json', 'datasets/teeth10k/train/incomplete/C01002756167.json', 'datasets/teeth10k/train/incomplete/C01002761703.json', 'datasets/teeth10k/train/incomplete/C01002763288.json', 'datasets/teeth10k/train/incomplete/C01002763514.json', 'datasets/teeth10k/train/incomplete/C01002764458.json', 'datasets/teeth10k/train/incomplete/C01002764650.json', 'datasets/teeth10k/train/incomplete/C01002767170.json', 'datasets/teeth10k/train/incomplete/C01002767967.json', 'datasets/teeth10k/train/incomplete/C01002770411.json', 'datasets/teeth10k/train/incomplete/C01002772389.json', 'datasets/teeth10k/train/incomplete/C01002772402.json', 'datasets/teeth10k/train/incomplete/C01002772660.json', 'datasets/teeth10k/train/incomplete/C01002775270.json', 'datasets/teeth10k/train/incomplete/C01002776709.json', 'datasets/teeth10k/train/incomplete/C01002778059.json', 'datasets/teeth10k/train/incomplete/C01002778116.json', 'datasets/teeth10k/train/incomplete/C01002781299.json', 'datasets/teeth10k/train/incomplete/C01002782256.json', 'datasets/teeth10k/train/incomplete/C01002782469.json', 'datasets/teeth10k/train/incomplete/C01002785002.json', 'datasets/teeth10k/train/incomplete/C01002787969.json', 'datasets/teeth10k/train/incomplete/C01002788634.json', 'datasets/teeth10k/train/incomplete/C01002791605.json', 'datasets/teeth10k/train/incomplete/C01002791650.json', 'datasets/teeth10k/train/incomplete/C01002792909.json', 'datasets/teeth10k/train/incomplete/C01002793517.json', 'datasets/teeth10k/train/incomplete/C01002795801.json', 'datasets/teeth10k/train/incomplete/C01002796969.json', 'datasets/teeth10k/train/incomplete/C01002799164.json', 'datasets/teeth10k/train/incomplete/C01002800279.json', 'datasets/teeth10k/train/incomplete/C01002801236.json', 'datasets/teeth10k/train/incomplete/C01002805533.json', 'datasets/teeth10k/train/incomplete/C01002808367.json', 'datasets/teeth10k/train/incomplete/C01002811237.json', 'datasets/teeth10k/train/incomplete/C01002811934.json', 'datasets/teeth10k/train/incomplete/C01002821159.json', 'datasets/teeth10k/train/incomplete/C01002823780.json', 'datasets/teeth10k/train/incomplete/C01002824747.json', 'datasets/teeth10k/train/incomplete/C01002830711.json', 'datasets/teeth10k/train/incomplete/C01002831149.json', 'datasets/teeth10k/train/incomplete/C01002834276.json', 'datasets/teeth10k/train/incomplete/C01002835435.json', 'datasets/teeth10k/train/incomplete/C01002836043.json', 'datasets/teeth10k/train/incomplete/C01002836706.json', 'datasets/teeth10k/train/incomplete/C01002840767.json', 'datasets/teeth10k/train/incomplete/C01002844996.json', 'datasets/teeth10k/train/incomplete/C01002846987.json', 'datasets/teeth10k/train/incomplete/C01002847045.json', 'datasets/teeth10k/val/complete/C01002722788.json', 'datasets/teeth10k/val/complete/C01002747516.json', 'datasets/teeth10k/val/complete/C01002774628.json', 'datasets/teeth10k/val/complete/C01002785507.json', 'datasets/teeth10k/val/complete/C01002796914.json', 'datasets/teeth10k/val/complete/C01002803328.json', 'datasets/teeth10k/val/complete/C01002815310.json', 'datasets/teeth10k/val/incomplete/C01002735210.json', 'datasets/teeth10k/val/incomplete/C01002737403.json', 'datasets/teeth10k/val/incomplete/C01002756831.json', 'datasets/teeth10k/val/incomplete/C01002763198.json', 'datasets/teeth10k/val/incomplete/C01002763390.json', 'datasets/teeth10k/val/incomplete/C01002775708.json', 'datasets/teeth10k/val/incomplete/C01002789185.json', 'datasets/teeth10k/val/incomplete/C01002801630.json', 'datasets/teeth10k/val/incomplete/C01002814870.json', 'datasets/teeth10k/val/incomplete/C01002826165.json', 'datasets/teeth10k/test/complete/C01002725466.json', 'datasets/teeth10k/test/complete/C01002726265.json', 'datasets/teeth10k/test/complete/C01002745749.json', 'datasets/teeth10k/test/complete/C01002757180.json', 'datasets/teeth10k/test/complete/C01002766258.json', 'datasets/teeth10k/test/complete/C01002767675.json', 'datasets/teeth10k/test/complete/C01002771849.json', 'datasets/teeth10k/test/complete/C01002780423.json', 'datasets/teeth10k/test/complete/C01002801146.json', 'datasets/teeth10k/test/complete/C01002847483.json', 'datasets/teeth10k/test/incomplete/C01002744748.json', 'datasets/teeth10k/test/incomplete/C01002776833.json', 'datasets/teeth10k/test/incomplete/C01002790266.json', 'datasets/teeth10k/test/incomplete/C01002796879.json', 'datasets/teeth10k/test/incomplete/C01002826705.json', 'datasets/teeth10k/test/incomplete/C01002827975.json', 'datasets/teeth10k/test/incomplete/C01002838720.json', 'datasets/teeth10k/test/incomplete/C01002845920.json']
full_error_cases=[os.path.join(PROJECT_DIR,case) for case in error_cases]
for dataset_path in data_root:
        # load bvh files that match given actors
        for f in os.listdir(dataset_path):
            f = os.path.abspath(os.path.join(dataset_path, f))
            if f.endswith(".json"):
                if f in full_error_cases:
                    continue
                bvh_files.append(f)

bvh_files_sample = random.sample(bvh_files, num_episodes)

env_list=[]
import data.bvh as bvh
from data.utils_np import *

for i in range(num_episodes):
                bvh_path=bvh_files_sample[i]
                rootpath,basepath=os.path.split(bvh_path)
                basename,_=os.path.splitext(basepath)
                remove_list=remove_dict[basepath]['remove_ids']

                geo_code=bvh.load_feature_npy(basename,True,remove_list=remove_list)
                anim = bvh.load_tooth_json(bvh_path, start=0,remove_list=remove_list)
                positions = anim.positions  # (seq,joint,3)
                rotations = anim.rotations  # (seq,joint,3,3)
                remove_list = anim.missing_list
                r6d = matrix9D_to_6D(rotations)

                first_state=np.concatenate([positions[0],r6d[0],positions[-1],r6d[-1],geo_code],axis=-1)

                convex_dict=rl_utils.getConvexDict(basename,remove_list)
                env = gym.make('OrthoEnv',first_step=first_state,convex_hulls=convex_dict,epsilon=0)
                env_list.append(env)
                
#-------------------------------------------------加载eval结束--------------------------------------------------------------
best_return = -np.inf
agent_name = type(agent).__name__
for i in range(pretrain_epoch):
    print(f"pretrain epoch:{i}")
    random.shuffle(replay_buffer)
    for iter in range(total_size//batch_size):
        start_idx = iter*batch_size
        end_idx = min((iter+1)*batch_size, total_size)
        
        transitions = replay_buffer[start_idx:end_idx]
        state, action, reward, next_state, done = zip(*transitions)
        transition_dict = {'states': np.array(state), 'actions': np.array(action), 'next_states': np.array(next_state), 'rewards': reward, 'dones': done}
        agent.update(transition_dict)
    temp_return_list = []
    done_cases = 0
    eval_cnt=0
    for env in env_list:
        print(f"eval case:{eval_cnt}")
        eval_cnt+=1
        _state = env.reset()
        bump_cnt=0
        done = False
        episode_return = 0

        while not done:
            state = _state.flatten()
            action = agent.take_action(state)[0]
            _action = action.reshape(28,9)
            _next_state, reward, done, _,info = env.step(_action)
            print(info)
            # next_state =_next_state.flatten()
            _state = _next_state
            episode_return += reward
            bump_cnt+=1
            if bump_cnt>150:
                            break
        if done:
            done_cases += 1
        print(f"episode return:{episode_return} for case:{eval_cnt-1}")
        temp_return_list.append(episode_return)
    tmp_mean = np.mean(temp_return_list)
    if tmp_mean> best_return:
            best_return = tmp_mean
            print(f"save best model with return:{best_return} in epoch:{i}")
            rl_utils.save_checkpoint(r"/home/mjy/teeth/RL/checkpt/",agent,i,i,suffix=f"best_{agent_name}_gt_train.pth")
    print(f"[eval] epoch:{i},mean return:{tmp_mean},best return:{best_return}")
    print(f"done cases:{done_cases}/{num_episodes}")
    return_list.append(tmp_mean)
for env in env_list:
    env.close()


# bvh_folder =r"/home/mjy/teeth/datasets/teeth10k/train"
# return_list = rl_utils.my_train_off_policy_agent(bvh_folder, agent, num_episodes,
#                                               replay_buffer, minimal_size,
#                                               batch_size)
# mean
mean_return = np.mean(return_list)
print(f"mean return:{mean_return}")

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Epoch')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format("OrthoStaging"))
plt.savefig(r"/home/mjy/teeth/RL/ddpg.png")


mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Epoch')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format("OrthoStaging"))
plt.savefig(r"/home/mjy/teeth/RL/ddpg_smooth.png")