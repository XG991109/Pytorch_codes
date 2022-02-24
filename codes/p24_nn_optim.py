# -*- coding: utf-8 -*- 
# @Time : 2022/2/23 16:51 
# @Author : zhuhuijuan
# @File : p24_nn_optim.py
import torch.optim
import torchvision.transforms
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=dataset_transform, download=True)
dataLoader = DataLoader(dataset, batch_size=1)


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


loss = nn.CrossEntropyLoss()
light = Lighter()
optim = torch.optim.SGD(light.parameters(),lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataLoader:
        imgs, targets = data
        outputs = light(imgs)
        # print(outputs)
        # print(targets)
        result_loss = loss(outputs, targets)
        # print(result_loss)
        optim.zero_grad() #梯度清零
        result_loss.backward()
        optim.step()
        running_loss= running_loss + result_loss
    print(running_loss)

