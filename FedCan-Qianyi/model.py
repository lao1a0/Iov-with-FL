# -*- coding:utf-8 -*-
"""
@Time: 2022/03/03 12:23
@Author: KI
@File: model.py
@Motto: Hungry And Humble
#https://blog.csdn.net/qq_56483157/article/details/133364495#t1
"""
from torch import nn
import torch


class ANN(nn.Module):
    def __init__(self, args, name):
        super(ANN, self).__init__()
        self.name = name
        self.len = 0
        self.loss = 0
        self.fc1 = nn.Linear(args.input_dim, 20)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 1)

    def forward(self, data):
        x = self.fc1(data)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        x = self.fc4(x)
        x = self.sigmoid(x)

        return x


class SimpleClassifier(nn.Module):
    def __init__(self, input_size=9, output_size=5):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
            # 移除了 nn.softmax，因为它将在损失函数中计算
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CNN(nn.Module):
    def __init__(self, num_class, name):
        super(CNN, self).__init__()
        self.name = name
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(5),
            nn.Dropout(p=0.5)

        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 32, 5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Linear(32, num_class),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.shape[0], -1)  # torch.Size([128, 32])
        x = self.block3(x)
        return x


def ResNet(name):
    ''':cvar
    返回修改好的模型，和冻结好的参数
    '''
    from torchvision.models import resnet18  # ResNet系列
    pretrain_model = resnet18(pretrained=False)
    pretrain_model.fc = nn.Linear(pretrain_model.fc.in_features, 5)  # 将全连接层改为自己想要的分类输出
    pretrained_dict = torch.load('/home/raoxy/data/BestModel/preResNetbest.pth')
    pretrained_dict.pop('fc.weight')
    pretrained_dict.pop('fc.bias')

    model_dict = pretrain_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)  # 模型参数列表进行参数更新，加载参数
    pretrain_model.load_state_dict(model_dict)  # 将满足条件的参数的 requires_grad 属性设置为False

    for name, value in pretrain_model.named_parameters():
        if (name != 'fc.weight') and (name != 'fc.bias'):
            value.requires_grad = False
    params_conv = filter(lambda p: p.requires_grad, pretrain_model.parameters())  # 要更新的参数在parms_conv当中
    # pretrain_model.name = name
    return pretrain_model, params_conv


def VGG(name):
    from torchvision.models import vgg16  # VGG系列
    from collections import OrderedDict
    pretrain_model = vgg16(pretrained=False)  # 导入了模型的框架
    pretrain_model.classifier[6] = nn.Linear(pretrain_model.classifier[6].in_features, 5)  # 将全连接层改为自己想要的分类输出
    model_dict = pretrain_model.state_dict()

    pretrained_dict = torch.load("/home/raoxy/data/BestModel/vgg16-397923af.pth")
    pretrained_dict.pop('classifier.6.bias')
    pretrained_dict.pop('classifier.6.weight')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)  # 模型参数列表进行参数更新，加载参数
    pretrain_model.load_state_dict(model_dict)
    # 将满足条件的参数的 requires_grad 属性设置为False
    for name, value in pretrain_model.named_parameters():
        if (name != 'classifier.6.weight') and (name != 'classifier.6.bias'):
            value.requires_grad = False

    upsample = nn.Upsample(size=[224, 224], mode='bilinear', align_corners=True)
    new_model = nn.Sequential(OrderedDict([
        ('laolao', upsample),
        ('features', pretrain_model.features),
        ('avgpool', pretrain_model.avgpool),
        ('classifier', nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=3),
            pretrain_model.classifier[0],
            pretrain_model.classifier[1],
            pretrain_model.classifier[2],
            pretrain_model.classifier[3],
            pretrain_model.classifier[4],
            pretrain_model.classifier[5],
            pretrain_model.classifier[6]
        ))
    ]))  # 创建一个新的模型
    params_conv = filter(lambda p: p.requires_grad, new_model.parameters())  # 要更新的参数在parms_conv当中
    # new_model.name = name
    return new_model, params_conv


def Alexnet(name):
    from torchvision.models import alexnet  # 最简单的模型
    from collections import OrderedDict
    pretrain_model = alexnet(pretrained=False)  # 导入了模型的框架
    pretrain_model.classifier[6] = nn.Linear(pretrain_model.classifier[6].in_features, 5)  # 将全连接层改为自己想要的分类输出
    model_dict = pretrain_model.state_dict()
    pretrained_dict = torch.load("/home/raoxy/data/BestModel/alexnet-owt-4df8aa71.pth")
    pretrained_dict.pop('classifier.6.bias')
    pretrained_dict.pop('classifier.6.weight')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)  # 模型参数列表进行参数更新，加载参数
    pretrain_model.load_state_dict(model_dict)
    for name, value in pretrain_model.named_parameters():
        if (name != 'classifier.6.weight') and (name != 'classifier.6.bias'):
            value.requires_grad = False

    upsample = nn.Upsample(size=[224, 224], mode='bilinear', align_corners=True)
    new_model = nn.Sequential(OrderedDict([
        ('laolao', upsample),
        ('features', pretrain_model.features),
        ('avgpool', pretrain_model.avgpool),
        ('classifier', nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=3),
            pretrain_model.classifier[0],
            pretrain_model.classifier[1],
            pretrain_model.classifier[2],
            pretrain_model.classifier[3],
            pretrain_model.classifier[4],
            pretrain_model.classifier[5],
            pretrain_model.classifier[6]
        ))
    ]))  # 创建一个新的模型
    params_conv = filter(lambda p: p.requires_grad, new_model.parameters())
    # new_model.name = name
    return new_model, params_conv
