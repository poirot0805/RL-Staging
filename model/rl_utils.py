import os
import json
from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import hppfcl
import gym
from data.utils_np import *
import data.bvh as bvh
import time
up_ids = [i for i in range(17, 10, -1)] \
    + [i for i in range(21, 28)] 
down_ids = [i for i in range(47, 40, -1)] \
    + [i for i in range(31, 38)]
ids = up_ids+down_ids
oid = {id: i for i, id in enumerate(ids)}   # dict{41:第0颗牙} 28颗牙

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        assert action.shape[0]==252
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)
    
def loadConvexMesh(file_name: str):
    loader = hppfcl.MeshLoader()
    bvh: hppfcl.BVHModelBase = loader.load(file_name)
    bvh.buildConvexHull(True, "Qt")
    return bvh.convex

def getConvexDict(client,remove_idx):
    stl_dir = os.path.join(r'/datasets/mjy/teeth/data_10000_image_with_model',client)
    up_meshes = []
    down_meshes = []
    up_list = []
    down_list = []
    for id in up_ids:
                # for each tooth
                if not os.path.exists(f'{stl_dir}/models/{id}._Root.stl'):
                    continue
                index_id = oid[id]
                if index_id in remove_idx:
                    continue
                mesh = loadConvexMesh(f'{stl_dir}/models/{id}._Root.stl')
                up_meshes.append(mesh)
                up_list.append(id)
    for id in down_ids:
                # for each tooth
                if not os.path.exists(f'{stl_dir}/models/{id}._Root.stl'):
                    continue
                index_id = oid[id]
                if index_id in remove_idx:
                    continue
                mesh = loadConvexMesh(f'{stl_dir}/models/{id}._Root.stl')
                down_meshes.append(mesh)
                down_list.append(id)
    convex_dict = {
        "up_meshes":up_meshes,
        "up_list":up_list,
        "down_meshes":down_meshes,
        "down_list":down_list
    }
    return convex_dict
def save_checkpoint(checkpoint_path, model, epoch, iteration, suffix=""):
    checkpoint_path = checkpoint_path + suffix
    model_dict = model.save()
    checkpoint = {
        "epoch": epoch,
        "iteration": iteration,
        "model": model_dict
    }
    torch.save(checkpoint, checkpoint_path)
    print("Save checkpoint to {}.".format(checkpoint_path))
    
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def my_train_off_policy_agent(bvh_folder,agent, num_episodes, replay_buffer, minimal_size, batch_size,explore_cnt=10):
    return_list = []
    # 加载num条原始数据
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

    bvh_files.sort()
    total_len=len(bvh_files)
    best_return_batch =-1000000000
    best_return =-1000000000
    agent_name = type(agent).__name__
    basename_list =[]
    first_state_list =[]
    remove_dict_list =[]
    assert total_len>= num_episodes
    for i in range(num_episodes):
                bvh_path=bvh_files[i]
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
                basename_list.append(basename)
                first_state_list.append(first_state)
                remove_dict_list.append(remove_list)
    with tqdm(total=explore_cnt,desc='Training agent...') as pbar:
        for epoch in range(explore_cnt):
            temp_return_list=[]
            for i in range(num_episodes):
                basename=basename_list[i]
                first_state=first_state_list[i]
                remove_list=remove_dict_list[i]
                start_time = time.time()
                convex_dict=getConvexDict(basename,remove_list)
                env = gym.make('OrthoEnv',first_step=first_state,convex_hulls=convex_dict,epsilon=0)
                end_time = time.time()
                # print(f"epoch:{epoch},episode:{i},load time:{end_time-start_time}")
                bump_cnt=0

                episode_return = 0
                _state = env.reset() # (28,126)
                done = False
                while not done:
                        state = _state.flatten()
                        action = agent.take_action(state)[0]
                        _action = action.reshape(28,9)
                        _next_state, reward, done, _,info = env.step(_action)
                        print(info)
                        next_state =_next_state.flatten()
                        replay_buffer.add(state, action, reward, next_state, done)  # replaybuffer里面放扁平化的数据
                        _state = _next_state
                        episode_return += reward
                        bump_cnt+=1
                        if replay_buffer.size() > minimal_size:
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                            agent.update(transition_dict)
                        if bump_cnt>150:
                            break
                temp_return_list.append(episode_return)
                env.close()
                if i%50==0:
                    tmp_mean = np.mean(temp_return_list)
                    if tmp_mean> best_return_batch:
                        best_return_batch = tmp_mean
                        save_checkpoint(r"/home/mjy/teeth/RL/checkpt/",agent,epoch,i,suffix=f"bestbatch_{agent_name}.pth")
                pbar.set_postfix({'episode': '%d' % i, 'return': '%.3f' % episode_return})
            tmp_mean = np.mean(temp_return_list)
            if tmp_mean> best_return:
                best_return = tmp_mean
                save_checkpoint(r"/home/mjy/teeth/RL/checkpt/",agent,epoch,0,suffix=f"best_{agent_name}.pth")
            return_list.append(tmp_mean)
            pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
                