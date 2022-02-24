# -*- coding: utf-8 -*- 
# @Time : 2022/2/23 15:28 
# @Author : zhuhuijuan
# @File : p23_nn_loss.py
import torch
from torch import nn
from torch.nn import L1Loss

input = torch.tensor([1, 2, 3], dtype=torch.float32)
output = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(input, (1, 1, 1, 3))
targets = torch.reshape(output, (1, 1, 1, 3))
loss = L1Loss(reduction='sum')
result = loss(inputs, targets)
print(result)
loss = L1Loss(reduction='mean')
result = loss(inputs, targets)
print(result)

loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets)
print(result_mse)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, [1, 3])
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)

