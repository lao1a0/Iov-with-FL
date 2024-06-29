# -*- coding:utf-8 -*-
"""
@Time: 2022/03/03 12:23
@Author: KI
@File: model.py
@Motto: Hungry And Humble
"""
from torch import nn


class SimpleClassifier(nn.Module):
    def __init__(self,name, input_size=9, output_size=5):
        super(SimpleClassifier, self).__init__()
        self.name = name
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
            # 移除了 nn.softmax，因为它将在损失函数中计算
        )

    def forward(self, x):
        x = self.fc(x)
        return x

