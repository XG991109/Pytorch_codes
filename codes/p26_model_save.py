# -*- coding: utf-8 -*- 
# @Time : 2022/2/23 19:26 
# @Author : zhuhuijuan
# @File : p26_model_save.py
import torch
import torchvision.models
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
# save method 1 模型结构+模型参数
torch.save(vgg16, "vgg16_method.pth")
# method 2     模型参数 （官方推荐）
torch.save(vgg16.state_dict(), "vgg16_method2_dict.pth")
import torchvision.transforms
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential


# 陷阱
class Lighter(nn.Module):
    def __init__(self):
        super(Lighter, self).__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


light = Lighter()
torch.save(light, "lighte_method.pth")
