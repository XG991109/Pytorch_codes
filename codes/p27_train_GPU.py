# -*- coding: utf-8 -*- 
# @Time : 2022/2/24 10:37 
# @Author : zhuhuijuan
# @File : p27_train_GPU.py
#  能调用cuda的地方只有 网络模型 、 数据 、 损失函数 修改 网络模型.cuda() 、 数据.cuda() 、 损失函数.cuda()
# dataset
import torch.optim
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from p27_model import *
import time
start_time = time.time()
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform=dataset_transform, download=True)
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=dataset_transform, download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)

print("train_data_size:{}".format(train_data_size))
print("test_data_size:{}".format(test_data_size))

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
# create network model
light = Lighter()
if torch.cuda.is_available():
    light = light.cuda()
# Loss function
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
# optimizer
learning_rate = 1e-2
optimizer = torch.optim.SGD(light.parameters(), learning_rate)

# record train and test steps
total_train_step = 0
total_test_step = 0

total_accuracy = 0
# train epoch
epoch = 10
writer = SummaryWriter("./logs_train")

for i in range(epoch):
    print("-----the {} epoch train starting ---- ".format(i + 1))
    # train step start
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        output = light(imgs)
        loss = loss_fn(output, targets)

        # optimizer the model
        optimizer.zero_grad()  # origin the grad  每次开始前都要初始化梯度
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("train steps :{} , Loss:{}".format(total_train_step,
                                                     loss.item()))  # loss.item means the real number not tensor  ,这是一种标准写法
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # test step start
    total_test_loss = 0  # to sum all test loss
    with torch.no_grad():  # under this condition , we can't optimizer the grad
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs =  imgs.cuda()
                targets = targets.cuda()
            outputs = light(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("the {} epoch , total_test_loss:{} ".format(i + 1, total_test_loss))
    print("the whole test_dataset accuracy rate:{}".format(total_accuracy / test_data_size))

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("total_accuracy", total_accuracy,total_test_step)
    total_test_step += 1

    torch.save(light, "model_pth/lighte_{}.pth".format(i + 1))
    # print(time.time()-start_time) # 一轮时间要花费的时间
    print("****the model has been saved")
writer.close()
