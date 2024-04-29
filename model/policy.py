import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

def exists(val):
    return val is not None

class TeethPositionalBias(nn.Module):
    def __init__(self, heads, **kwargs):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)
    
    def get_bias(self, i, j, device):
        i_arange = torch.arange(28, device = device)
        j_arange = torch.arange(28, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 1 j') - rearrange(i_arange, 'i -> 1 1 i 1'))
        for i in range(28):
            for j in range(28):
                bias[...,i,j] =abs((i%14)-(j%14))+abs((i//14)-(j//14))
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, qk_dots):
        h, i, j, device = *qk_dots.shape[-3:], qk_dots.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return qk_dots + self.bias[..., :i,:j]
        
        if not exists(self.bias):
            bias = self.get_bias(i, j, device)
            bias = bias * self.slopes

            num_heads_unalibied = h - bias.shape[-3]
            bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
            self.register_buffer('bias', bias, persistent = False)
        return qk_dots + self.bias[..., :i,:j]
    
class SPMultiHeadedAttention(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout=0.1,
                 pre_lnorm=True, bias=False, pe=True):
        """
        Multi-headed attention of vanilla transformer with memory mechanism.

        Args:
            n_head (int): Number of heads.
            d_model (int): Input dimension.
            d_head (int): Head dimension.
            dropout (float, optional): Dropout value. Defaults to 0.1.
            pre_lnorm (bool, optional):
                Apply layer norm before rest of calculation. Defaults to True.
                In original Transformer paper (pre_lnorm=False):
                    LayerNorm(x + Sublayer(x))
                In tensor2tensor implementation (pre_lnorm=True):
                    x + Sublayer(LayerNorm(x))
            bias (bool, optional):
                Add bias to q, k, v and output projections. Defaults to False.

        """
        super(SPMultiHeadedAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.pre_lnorm = pre_lnorm
        self.bias = bias
        self.atten_scale = 1 / math.sqrt(self.d_model)
        self.pe = pe
        self.q_linear = nn.Linear(d_model, n_head * d_head, bias=bias)
        self.k_linear = nn.Linear(d_model, n_head * d_head, bias=bias)
        self.v_linear = nn.Linear(d_model, n_head * d_head, bias=bias)
        self.out_linear = nn.Linear(n_head * d_head, d_model, bias=bias)

        self.droput_layer = nn.Dropout(dropout)
        self.atten_dropout_layer = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(d_model)
        self.alibi_pos = TeethPositionalBias(n_head)

    def forward(self, hidden, memory=None, mask=None):
        """
        Args:
            hidden (Tensor): Input embedding or hidden state of previous layer.
                Shape: (batch, seq, dim)
            memory (Tensor): Memory tensor of previous layer.
                Shape: (batch, mem_len, dim)
            mask (BoolTensor, optional): Attention mask.
                Set item value to True if you DO NOT want keep certain
                attention score, otherwise False. Defaults to None.
                Shape: (seq, seq+mem_len).
        """
        if memory is None:
            combined = hidden
        else:
            combined = torch.cat([memory, hidden], dim=1)

        if self.pre_lnorm:
            hidden = self.layer_norm(hidden)
            combined = self.layer_norm(combined)

        # shape: (batch, q/k/v_len, dim)
        #print(hidden.shape)
        q = self.q_linear(hidden)
        k = self.k_linear(combined)
        v = self.v_linear(combined)

        # reshape to (batch, q/k/v_len, n_head, d_head)
        q = q.reshape(q.shape[0], q.shape[1], self.n_head, self.d_head)
        k = k.reshape(k.shape[0], k.shape[1], self.n_head, self.d_head)
        v = v.reshape(v.shape[0], v.shape[1], self.n_head, self.d_head)

        # transpose to (batch, n_head, q/k/v_len, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # (batch, n_head, q_len, k_len)
        atten_score = torch.matmul(q, k.transpose(-1, -2)) # * self.atten_scale
        if self.pe:
            atten_score = self.alibi_pos(atten_score)
        if mask is not None:
            # apply attention mask
            atten_score = atten_score.masked_fill(mask, float("-inf"))
        atten_score = atten_score.softmax(dim=-1)
        atten_score = self.atten_dropout_layer(atten_score)

        # (batch, n_head, q_len, d_head)
        atten_vec = torch.matmul(atten_score, v)
        # (batch, q_len, n_head*d_head)
        atten_vec = atten_vec.transpose(1, 2).flatten(start_dim=-2)

        # linear projection
        output = self.droput_layer(self.out_linear(atten_vec))

        if self.pre_lnorm:
            return hidden + output
        else:
            return self.layer_norm(hidden + output)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout=0.1, pre_lnorm=True):
        """
        Positionwise feed-forward network.

        Args:
            d_model(int): Dimension of the input and output.
            d_inner (int): Dimension of the middle layer(bottleneck).
            dropout (float, optional): Dropout value. Defaults to 0.1.
            pre_lnorm (bool, optional):
                Apply layer norm before rest of calculation. Defaults to True.
                In original Transformer paper (pre_lnorm=False):
                    LayerNorm(x + Sublayer(x))
                In tensor2tensor implementation (pre_lnorm=True):
                    x + Sublayer(LayerNorm(x))
        """
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.pre_lnorm = pre_lnorm

        self.layer_norm = nn.LayerNorm(d_model)
        self.network = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        if self.pre_lnorm:
            return x + self.network(self.layer_norm(x))
        else:
            return self.layer_norm(x + self.network(x))

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.input_dim = state_dim//28
        self.fc_layers=nn.ModuleList()
        self.fc_layers.append(nn.Linear(self.input_dim, hidden_dim))
        self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc_low = nn.Linear(hidden_dim, hidden_dim//4)
        self.fc_out = nn.Linear(hidden_dim//4*28, action_dim)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值
        self.activation = nn.PReLU()
    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # return torch.tanh(self.fc2(x)) * self.action_bound
        x_ = x.reshape(-1,28,126)
        for i, layer in enumerate(self.fc_layers):
            if i==0:
                x_ = self.activation(layer(x_))
            else:
                x_ = self.activation(x_+layer(x_))
        x_ = self.activation(self.fc_low(x_))
        res = x_.flatten(start_dim=1)
        return torch.tanh(self.fc_out(res))
    
class PolicyNetAttention(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetAttention, self).__init__()
        self.input_dim = state_dim//28

        self.n_layer = 6
        self.dropout= 0.
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim,hidden_dim),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim,hidden_dim),
            nn.PReLU(),
            nn.Dropout(self.dropout)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim,9)
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.att_layers = nn.ModuleList()
        self.pff_layers = nn.ModuleList()
        
        for i in range(self.n_layer):
            self.att_layers.append(SPMultiHeadedAttention(8,hidden_dim,hidden_dim,dropout=self.dropout)) 
            self.pff_layers.append(PositionwiseFeedForward(hidden_dim, hidden_dim,dropout=self.dropout))

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # return torch.tanh(self.fc2(x)) * self.action_bound
        x_ = x.reshape(-1,28,126)
        x = self.encoder(x_)
        for i in range(self.n_layer):
            x = self.att_layers[i](x, mask=None)
            x = self.pff_layers[i](x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x.flatten(start_dim=1)
    
class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)

        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob.sum(-1)
        # normal_sample = dist.rsample()  # rsample()是重参数化采样
        # log_prob = dist.log_prob(normal_sample).sum(axis=-1)
        
        # action = torch.tanh(normal_sample)
        # # 计算tanh_normal分布的对数概率密度
        # log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        # action = action * self.action_bound
        return pi_action, logp_pi
