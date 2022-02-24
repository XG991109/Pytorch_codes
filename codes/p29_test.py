# -*- coding: utf-8 -*- 
# @Time : 2022/2/24 18:06 
# @Author : zhuhuijuan
# @File : p29_test.py
import torch
import torchvision
from PIL import Image
from p27_model import *

image_path = "../imgs/airplane.png"
# 加 convert('RGB')的原因是因为 PNG 是4个通道，除了RGB以外还有一个透明通道，convert('RGB‘)只保留颜色通道 以适应所有图片类型
image = Image.open(image_path).convert('RGB')  # 此时的image是PIL 类型 ， 接下来还要转化为tensor 类型
print(image)

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])
image = data_transform(image)
print(image.shape)

light_model = torch.load("model_pth/lighte_10.pth")
print(light_model)
image = torch.reshape(image, (1, 3, 32, 32))
  # 将模型设为测试类型
# 并且采用没有梯度的设置
light_model.eval()
with torch.no_grad():
    output = light_model(image.cuda())  # 用GPU训练的模型， 再test时的数据，也要转换为cuda类型；CPU训练的模型，同理
print(output)
print(output.argmax(1))