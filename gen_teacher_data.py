"""
description: generate teacher data
0. 遍历目录下的所有json文件
1. 从json文件中读取数据
2. 读取model文件，生成convex_hulls{"up_meshes"，"up_list","down_meshes","down_list"}
3. 对于每一条数据，计算其action,reward：
    数据格式transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
4. 将数据保存到teacher_data文件夹下，每一条数据保存为一个文件
"""

import os
import json
import numpy as np
import torch
from data.utils_np import *
import data.bvh as bvh
import gym
import hppfcl

up_ids = [i for i in range(17, 10, -1)] \
    + [i for i in range(21, 28)] 
down_ids = [i for i in range(47, 40, -1)] \
    + [i for i in range(31, 38)]
ids = up_ids+down_ids
oid = {id: i for i, id in enumerate(ids)}   # dict{41:第0颗牙} 28颗牙

def loadConvexMesh(file_name: str):
    loader = hppfcl.MeshLoader()
    bvh: hppfcl.BVHModelBase = loader.load(file_name)
    bvh.buildConvexHull(True, "Qt")
    return bvh.convex

def simulation(pos,r9d,geo_code,client,remove_idx):
    r6d = matrix9D_to_6D(r9d)

    first_state=np.concatenate([pos[0],r6d[0],pos[-1],r6d[-1],geo_code],axis=-1)
    
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
    env = gym.make('OrthoEnv',first_step=first_state,convex_hulls=convex_dict,epsilon=0)
    env.reset()
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], "q_values":[]}
    step_num = pos.shape[0]
    check_error_premature=False
    check_error_nonstop=False
    total_return=0
    for i in range(step_num):
        j = i+1 if i<step_num-1 else step_num-1
        current_state = np.concatenate([pos[i],r6d[i],pos[-1],r6d[-1],geo_code],axis=-1)
        next_state = np.concatenate([pos[j],r6d[j],pos[-1],r6d[-1],geo_code],axis=-1)
        action = next_state[:,:9]-current_state[:,:9]
        _,reward,done,_,info=env.step(action)
        print(info)
        total_return+=reward
        if done and i<step_num-1:
            check_error_premature=True
            print(f"client:{client} gets target before final step!!!")
        if done==False and i ==step_num-1:
            check_error_nonstop=True
            print(f"client:{client} in env check done wrong!!!")
        transition_dict['states'].append(current_state.tolist())
        transition_dict['actions'].append(action.tolist())
        transition_dict['next_states'].append(next_state.tolist())
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)
    for i in range(step_num):
        if i==0:
            transition_dict['q_values'].append(total_return)
        else:
            last_q=transition_dict['q_values'][i-1]
            transition_dict['q_values'].append(last_q-transition_dict['rewards'][i-1])
    json_path = os.path.join(r"/datasets/mjy/teacher_data_onereward",client+".json")
    print(json_path)
    with open(json_path, "w") as fh:
        data = {
            "transition":transition_dict,
            "remove_idx":remove_idx,
            "total_return":total_return
        }
        json.dump(data, fh)
    return check_error_premature,check_error_nonstop

def load_raw_data(bvh_folder):
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
    names=[]

    current_id=0
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
    sim_error_prem=[]
    sim_error_nonstop=[]
    for bvh_path in bvh_files:
            print("{}/{} Processing file {}".format(current_id,total_len,bvh_path))
            rootpath,basepath=os.path.split(bvh_path)
            basename,_=os.path.splitext(basepath)
            remove_list=remove_dict[basepath]['remove_ids']

            geo_code=bvh.load_feature_npy(basename,True,remove_list=remove_list)  # 【28，seq，9】

            anim = bvh.load_tooth_json(bvh_path, start=0,remove_list=remove_list)
            positions = anim.positions  # (seq,joint,3)
            rotations = anim.rotations  # (seq,joint,3,3)
            
            remove_list = anim.missing_list

            current_id+=1
            
            e1,e2=simulation(positions,rotations,geo_code,basename,remove_list)
            if e1:
                sim_error_prem.append(basename)
            if e2:
                sim_error_nonstop.append(basename)
    print("target before final step!")
    print(sim_error_prem)
    print("non stop!")
    print(sim_error_nonstop)

if __name__ == "__main__":
    load_raw_data(r"/home/mjy/teeth/datasets/teeth10k/train")