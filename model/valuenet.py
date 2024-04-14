
import torch
from torch import nn
import torch.nn.functional as F
from data.utils_torch import matrix6D_to_9D_torch
def get_angle_error(batch, a, b,device):
    # r9d
    a = matrix6D_to_9D_torch(a)
    b = matrix6D_to_9D_torch(b)
    rm = torch.matmul(a.transpose(-2, -1), b)
    tr = torch.zeros([batch,28],device=device)
    for i in range(batch):
            for k in range(28):
                tr[i,k] = torch.trace(rm[i, k])
    res = torch.acos(torch.clamp((tr - 1) / 2, -1.0, 1.0))
    return res.unsqueeze(-1)

class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.input_dim = (state_dim + action_dim)//28
        # self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        # self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.fc_out = torch.nn.Linear(hidden_dim, 1)
        self.activation = nn.ELU()
        self.fc_layers=nn.ModuleList()
        self.fc_layers.append(nn.Linear(self.input_dim+2, hidden_dim))
        self.fc_layers.append(nn.Linear(self.input_dim+hidden_dim, hidden_dim))
        self.fc_layers.append(nn.Linear(self.input_dim+hidden_dim, hidden_dim))
        self.fc_out = nn.Linear(hidden_dim*28, 1)
        self.kappa =8.0
    def forward(self, x, a):
        # cat = torch.cat([x, a], dim=1) # 拼接状态和动作
        # x = F.relu(self.fc1(cat))
        # x = F.relu(self.fc2(x))
        # return self.fc_out(x)
        x_ = x.reshape(-1,28,126)
        a_ = a.reshape(-1,28,9)
        c1,d1 = x_[...,:3],x_[...,3:9]
        c2,d2 = x_[...,9:12],x_[...,12:18]
        r1 = torch.norm(c1 - c2, dim=-1).unsqueeze(-1)
        r2 = get_angle_error(d1.shape[0],d1,d2,x.device)
        r1 = torch.log(self.kappa*r1+1)
        r2 = torch.log(self.kappa*r2*10.0+1)
        res = torch.cat([r1,r2],dim=-1)
        input = torch.cat([x_,a_],dim=-1)
        for layer in self.fc_layers:
            if res is not None:
                res = layer(torch.cat([input,res], dim = -1))
            else:
                res = layer(input)
            res = self.activation(res)
        res = res.flatten(start_dim=1)
        return self.fc_out(res)
        """
            def calculate_rotation_reward(self,beta=10,k=8.0):
        # 根据苹果接近目标朝向的情况计算奖励
        beta = self.beta
        current_rotations = self.state[:, 3:9]
        target_rotations = self.state[:, 12:18]
        angle_error = self.get_angle_error_np(current_rotations, target_rotations)
        reward = -np.log((1+k*beta*angle_error)/(1+k*beta*self.first_angle_error))
        print("reward-max:",reward.max(),"min:",reward.min(),"mean:",reward.mean())
        print("angle_error-max:",angle_error.max(),"min:",angle_error.min(),"mean:",angle_error.mean())
        print("first_angle_error-max:",self.first_angle_error.max(),"min:",self.first_angle_error.min(),"mean:",self.first_angle_error.mean())  
        reward = np.sum(reward)
        # x = np.sum(angle_error-self.first_angle_error)*beta
        # reward = (1 / (1 + np.exp(x))-0.6)*2
        self.first_angle_error = angle_error
        return reward
        """