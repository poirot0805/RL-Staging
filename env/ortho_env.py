import numpy as np
import gym
from gym import spaces
from data import utils_np
import hppfcl

"""_summary_
1. 动作空间由50个苹果的9维变化向量组成
2. 状态由50个苹果的9维姿态，50个苹果的9维目标姿态和50个苹果的100个形状采样点组成(9+9+100*3)
3. 可以参考“BipedalWalker-v3”环境做出修改
4.奖励由以下部分组成：
4.1. 基础的 
-平移奖励：如果苹果向目标位置移动，则给予正奖励；如果远离目标位置，则给予负奖励。
-旋转奖励：同理，接近目标朝向给予正激励
4.2. 约束性 
-完成奖励：当某颗苹果到达指定位置且朝向正确时，给予一次性的较大的正奖励
-碰撞惩罚：当两颗苹果的距离小于某个阈值时，给予负奖励
-步数奖励：每多走一步给予负的奖励，使总步数尽量小
4.3. 协同 当所有苹果都到指定为止和朝向时，给予额外的团队奖励
"""
NUM_TEETH=28
STATE_DIM=126

up_ids = [i for i in range(17, 10, -1)] \
    + [i for i in range(21, 28)] 
down_ids = [i for i in range(47, 40, -1)] \
    + [i for i in range(31, 38)]
ids = up_ids+down_ids
oid = {id: i for i, id in enumerate(ids)}   # dict{41:第0颗牙} 28颗牙

class OrthoEnv(gym.Env):
    def __init__(self,first_step,convex_hulls,beta=10,contact_threshold=0.3,prize=200,epsilon=0.6):
        
        # 定义动作空间和观察空间
        self.action_space = spaces.Box(low=-1, high=1, shape=(NUM_TEETH, 9), dtype=np.float32)  # 每个苹果的位置和姿态变化向量(平移和旋转
        self.observation_space = spaces.Box(low=-50, high=50, shape=(NUM_TEETH, STATE_DIM), dtype=np.float32)  
        
        self.first_step=first_step
        self.first_dis = np.linalg.norm(first_step[:, :3] - first_step[:, 9:12], axis=1)    # 初始状态与目标位置的距离
        self.first_angle_error = self.get_angle_error_np(first_step[:, 3:9], first_step[:, 12:18])  #初始状态与目标朝向的角度差
        self.convex_hulls=convex_hulls  # 碰撞检测用的凸包{"up_meshes"，"up_list","down_meshes","down_list"}
        
        self.beta=beta  # 旋转奖励的系数
        self.contact_threshold=contact_threshold    # 碰撞惩罚的阈值
        self.prize=prize    # 完成奖励
        self.epsilon=epsilon    # 随机初始化的概率
        # 初始化状态
        self.state = first_step
        self.last_state = None


    def reset(self):
        # 重置环境到初始状态
        ka = np.random.random()
        if ka < self.epsilon:
            # 以一定概率随机初始化前9维
            self.state[:, :3] = self.first_step[:, :3] + np.random.normal(ka, 5, (NUM_TEETH, 3))
            euler = np.random.normal(ka, 20, (NUM_TEETH, 3))    # 用欧拉角完成扰动
            R = utils_np.euler_to_matrix(euler)
            mat9d = utils_np.matrix6D_to_9D(self.first_step[:, 3:9])
            self.state[:, 3:9] = utils_np.matrix9D_to_6D(np.matmul(mat9d, R))
            self.first_dis = np.linalg.norm(self.state[:, :3] - self.state[:, 9:12], axis=1)
            self.first_angle_error = self.get_angle_error_np(self.state[:, 3:9], self.state[:, 12:18])
        else:
            self.state = self.first_step
        self.last_state = None
        return self.state
    
    def step(self,action):
        done = False
        reward = 0
        self.last_state = self.state.copy()
        # 用简化的逻辑更新状态
        self.state[:, :9] += action  # 假设前9维为当前姿态，直接加上动作向量
        
        # 计算奖励
        reward_trans = self.calculate_translation_reward()
        reward_rot = self.calculate_rotation_reward()
        reward_collision = self.calculate_collision_penalty()*self.beta
        reward_smooth = self.calculate_smooth_reward()
        reward+=(reward_trans+reward_rot+reward_collision+reward_smooth)
        reward -= 1  # 每执行一步减少的奖励
        
        done = self.check_done()

        # 当所有苹果都到达指定位置和姿态时，给予额外奖励
        if done:
            reward += self.prize  # 假定的团队奖励
        
        return self.state, reward, done, False, {"info": f"trans:{reward_trans}, rot:{reward_rot}, smooth:{reward_smooth},collision:{reward_collision}"}

    def calculate_translation_reward(self):
        # 根据苹果向目标位置移动的情况计算奖励
        current_positions = self.state[:, :3]
        target_positions = self.state[:, 9:12]
        distance = np.linalg.norm(current_positions - target_positions, axis=1)
        reward = np.sum(self.first_dis - distance)
        
        return reward

    def calculate_rotation_reward(self,beta=10):
        # 根据苹果接近目标朝向的情况计算奖励
        beta = self.beta
        current_rotations = self.state[:, 3:9]
        target_rotations = self.state[:, 12:18]
        angle_error = self.get_angle_error_np(current_rotations, target_rotations)
        reward = np.sum(self.first_angle_error - angle_error)*beta
        return reward
    def calculate_smooth_reward(self,beta=10):
        beta = self.beta
        current_positions = self.state[:, :3]
        last_positions = self.last_state[:, :3]
        current_rotations = self.state[:, 3:9]
        last_rotations = self.last_state[:, 3:9]
        reward = 0
        dis = np.linalg.norm(current_positions - last_positions, axis=1)
        delta = dis-0.5
        angle_error = self.get_angle_error_np(current_rotations, last_rotations)
        deltar =  angle_error - np.pi/60
        reward = 0-np.sum(delta)-np.sum(deltar)*beta
        return reward
        
    def calculate_collision_penalty(self):
        # 计算碰撞惩罚
        reward=0
        up_meshes = self.convex_hulls["up_meshes"]
        up_list = self.convex_hulls["up_list"]
        down_meshes = self.convex_hulls["down_meshes"]
        down_list = self.convex_hulls["down_list"]
        rot_data = utils_np.matrix9D_to_quat(utils_np.matrix6D_to_9D(self.state[:, 3:9]))
        for i in range(1,len(up_meshes)):
                T1 = hppfcl.Transform3f()
                T2 = hppfcl.Transform3f()
                index1 = oid[up_list[i]]
                index2 = oid[up_list[i-1]]
                T1.setTranslation(np.array(self.state[index1][:3]))
                T2.setTranslation(np.array(self.state[index2][:3]))
                T1.setQuatRotation(hppfcl.Quaternion(w=rot_data[index1][0],x=rot_data[index1][1],y=rot_data[index1][2],z=rot_data[index1][3]))
                T2.setQuatRotation(hppfcl.Quaternion(w=rot_data[index2][0],x=rot_data[index2][1],y=rot_data[index2][2],z=rot_data[index2][3]))
                reward+=self.hppfcl_check(up_meshes[i], T1, up_meshes[i-1], T2,index1)
        for i in range(1,len(down_meshes)):
                T1 = hppfcl.Transform3f()
                T2 = hppfcl.Transform3f()
                index1 = oid[down_list[i]]
                index2 = oid[down_list[i-1]]
                T1.setTranslation(np.array(self.state[index1][:3]))
                T2.setTranslation(np.array(self.state[index2][:3]))
                T1.setQuatRotation(hppfcl.Quaternion(w=rot_data[index1][0],x=rot_data[index1][1],y=rot_data[index1][2],z=rot_data[index1][3]))
                T2.setQuatRotation(hppfcl.Quaternion(w=rot_data[index2][0],x=rot_data[index2][1],y=rot_data[index2][2],z=rot_data[index2][3]))
                reward+=self.hppfcl_check(down_meshes[i], T1, down_meshes[i-1], T2,index1)
        return reward
    
    def check_done(self):
        # 检查是否所有苹果都到达了指定位置和姿态
        if np.linalg.norm(self.state[:, :3] - self.state[:, 9:12], axis=1).max() < 0.2 and \
            self.get_angle_error_np(self.state[:, 3:9], self.state[:, 12:18]).max()*90 < np.pi:
            return True
        return False

    def get_angle_error_np(self,a,b):
        a = utils_np.matrix6D_to_9D(a)       
        b = utils_np.matrix6D_to_9D(b)  
        # Matmul in numpy
        
        rm = np.matmul(np.swapaxes(a, -2, -1), b)  # Adjusted axis for numpy's transpose
        tr = np.zeros(28)

        for k in range(28):
            tr[k] = np.trace(rm[k])
        # Numpy's arccos and clip functions are directly equivalent to torch's acos and clamp
        res = np.arccos(np.clip((tr - 1) / 2, -1.0, 1.0))
        return res
    
    def hppfcl_check(self,shape1, T1, shape2, T2, thres=0.6):
        thres = self.contact_threshold
        col_req = hppfcl.CollisionRequest()
        col_res = hppfcl.CollisionResult()
        hppfcl.collide(shape1, T1, shape2, T2, col_req, col_res)
        if col_res.isCollision():
            distance_req = hppfcl.DistanceRequest()
            distance_res = hppfcl.DistanceResult()
            dist = hppfcl.distance(shape1, T1, shape2, T2, distance_req, distance_res)

            distance_res.clear()
            col_res.clear()
            if abs(dist)>thres:
                return thres-abs(dist)
            else:
                return 0
        # print("distance:",distance_res.min_distance)
        col_res.clear()
        return 0
    def render(self, mode='human', close=False):
        pass
