# -*- coding: utf-8 -*- 
# @Time : 2022/2/23 19:29 
# @Author : zhuhuijuan
# @File : p26_model_load.py

# method 1 -> load medel
import torch

model = torch.load("vgg16_method.pth") #copy referance path 此处在同一个文件夹下 直接写
print(model)

# method 2
import torchvision.models

# vgg_dict = torch.load("vgg16_method2_dict.pth")
# # print(vgg_dict)
# vgg16 = torchvision.models.vgg16(pretrained=False)
# vgg16.load_state_dict(vgg_dict)
# print(vgg16)

# load lighte model method1
# we can't directly load lighte model by method 1 ,we must add the original lighte model structure code
# the method just let us not to init "ligthe = Ligther() "
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

#
# class Lighter(nn.Module):
#     def __init__(self):
#         super(Lighter, self).__init__()
#         self.model1 = Sequential(
#             Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
#             MaxPool2d(kernel_size=2),
#             Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
#             MaxPool2d(kernel_size=2),
#             Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
#             MaxPool2d(kernel_size=2),
#             Flatten(),
#             Linear(in_features=1024, out_features=64),
#             Linear(in_features=64, out_features=10)
#         )
#
#     def forward(self, x):
#         x = self.model1(x)
#         return x
#
#
# model = torch.load("lighte_method.pth")
# print(model)

# load model method 2  we can import the file name which import all the content  and we can directly load model
# from p26_model_save import *
# model = torch.load("lighte_method.pth")
# print(model)