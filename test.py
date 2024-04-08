import numpy as np
import torch
from data.utils_np import *
import data.utils_torch as utils_torch
# **test 1
# q = np.array([0.026454919949173927,
#                 0.6598586440086365,
#                 -0.7503260374069214,
#                 0.029958751052618027]).reshape(1, 4)

# mat9d = quat_to_matrix9D(q)

# q2 = matrix9D_to_quat(mat9d)
# print(q2)

# mat9d_torch = utils_torch.quat_to_matrix9D_torch(torch.tensor(q).reshape(1, 4))

# q3 =utils_torch.matrix9D_to_quat_torch(mat9d_torch)
# print(q3)

# **test 2
# random_nums = np.random.normal(loc=0, scale=1, size=(3, 4))
# print(random_nums)
# random_nums = np.random.normal(loc=0, scale=0.1, size=(3, 4))
# print(random_nums)
# random_nums = np.random.normal(loc=0, scale=20, size=(3, 4))
# print(random_nums)

# **test 3
# a = np.ones((4,2))
# b = np.zeros((4,1))
# c = np.concatenate([a,b],axis=-1)

# print(c)

# d = a.tolist()
# print(d)
# e = np.array(d)
# print(e)

# **test4

a= torch.randn(1,1)
print(a.item())
