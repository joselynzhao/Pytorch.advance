#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWARE:PyCharm
@FILE:main.py
@TIME:2019/9/17 上午11:13
@DES:
'''


import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        #执行父类的构造函数
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,6,5) #1是输入通道,6是输出通道,5是卷积核变长
        self.conv2 = nn.Conv2d(6,16,5)

        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)


    def forward(self,x):
        # 卷积 \激活\此话
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)

        # reshape

        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

net =  Net()
print(net)