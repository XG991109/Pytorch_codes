# -*- coding: utf-8 -*- 
# @Time : 2022/2/23 17:04 
# @Author : zhuhuijuan
# @File : p25_model_pretrained.py
import torchvision
from torch import nn

vgg16_true = torchvision.models.vgg16(pretrained=True) #参数已经是训练过的
vgg16_false = torchvision.models.vgg16(pretrained=False)

print(vgg16_true)
#修改模型 添加
# vgg16_true.add_module("add_linear",nn.Linear(1000,10)) #若要在全局添加，即在最后vgg_true.add_module
vgg16_true.classifier.add_module("add_linear",nn.Linear(1000,10)) # 若需要在对应层添加，加上对应层名称vgg16_true.classifier.add_module
print(vgg16_true)
#修改模型 修改指定层
print(vgg16_false)
vgg16_false.classifier[6] =nn.Linear(4096,10)
print(vgg16_false)