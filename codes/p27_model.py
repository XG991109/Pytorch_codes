# -*- coding: utf-8 -*- 
# @Time : 2022/2/24 15:38 
# @Author : zhuhuijuan
# @File : p27_model.py

import torch
from torch import nn


class Lighter(nn.Module):
    def __init__(self):
        super(Lighter, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 4 * 4, out_features=64),
            nn.Linear(in_features=64, out_features=10)  # 10 classes

        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    light = Lighter()
    test = torch.ones((64, 3, 32, 32))
    output = light(test)
    print(output.shape)
