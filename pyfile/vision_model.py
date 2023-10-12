import sys
import torch
import tensorwatch as tw
import torchvision.models
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self, num_class):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 32, 5, padding=1)  # 输入通道数为64，输出通道数为64，卷积核大小为3x3，步长为1，填充为1
        self.pool1 = nn.MaxPool2d(2)  # 最大池化层，池化核大小为2x2
        self.gap = nn.AdaptiveAvgPool2d(5)  # 全局平均池化层
        self.fc1 = nn.Linear(32, num_class)  # 全连接层 ，输入特征维度位256 ，输出特征维度位num_class
        self.relu = nn.ReLU()  # 激活函数
        self.dropout = nn.Dropout(p=0.5)  # 随机失活层
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.gap(x)
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = x.view(x.shape[0], -1)  # torch.Size([128, 32])
        x = self.softmax(self.fc1(x))
        return x
alexnet_model = CNN(5)
img=tw.draw_model(alexnet_model, [128, 3, 224, 224])
img.save('../img/cnn.png')