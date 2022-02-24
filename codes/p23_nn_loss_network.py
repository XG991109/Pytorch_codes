# -*- coding: utf-8 -*- 
# @Time : 2022/2/23 15:46 
# @Author : zhuhuijuan
# @File : p23_nn_loss_network.py
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
for data in dataLoader:
    imgs, targets = data
    outputs = light(imgs)
    # print(outputs)
    # print(targets)
    result_loss = loss(outputs, targets)
    # print(result_loss)
    result_loss.backward()
    print("ok")
