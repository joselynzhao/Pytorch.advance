#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWARE:PyCharm
@FILE:cifar-10.py
@TIME:2019/9/17 下午2:51
@DES:
'''

import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

show = ToPILImage #将tensor转换为image,方便可视化


#定义对数据的预处理
transform = transforms.Compose([transforms.ToTensor(),  #转为tensor
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5) #归一化
                                                     )])



#训练集
trainset = tv.datasets.CIFAR10(
    root='/home/joselyn/data/',
    train=True,
    download=True,
    transform=transform)

trainloader = t.utils.???
