# -*- coding:utf-8 -*-
"""
@Time: 2022/03/03 12:22
@Author: KI
@File: get_data.py
@Motto: Hungry And Humble
"""

import sys
import numpy as np
import pandas as pd
import torch
# from args import args_parser
import torchvision

sys.path.append('.')  # lao改了
from torch.utils.data import Dataset, DataLoader

# 解析命令行参数
# args = args_parser()

# 设置设备（如果有GPU则使用GPU，否则使用CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建一个包含10个任务的客户端列表
# clients_wind = ['Task1_W_Zone' + str(i) for i in range(1, 11)]
clients = ['car_' + str(i) for i in range(1, 11)]

def carHacking_Data(file_name, args, flag=2):
    data_train = '{}{}/{}'.format(args.root, file_name, 'train/')
    data_test = '{}{}/{}'.format(args.root, file_name, 'test/')
    # data_var = '{}{}/{}'.format(args.root, file_name, 'verify/')

    use_cuda = torch.cuda.is_available()

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if flag == 1:
        Dtr = DataLoader(
            torchvision.datasets.ImageFolder(root=data_train,
                                             transform=torchvision.transforms.ToTensor()),
            batch_size=args.B,
            shuffle=True, **kwargs)
        return "", Dtr
    else:
        Dtr = DataLoader(
            torchvision.datasets.ImageFolder(root=data_train,
                                             transform=torchvision.transforms.ToTensor()),
            batch_size=args.B,
            shuffle=True, **kwargs)
        Dte = DataLoader(
            torchvision.datasets.ImageFolder(root=data_test,
                                             transform=torchvision.transforms.ToTensor()),
            batch_size=args.B,
            shuffle=False, **kwargs)

        # Val = DataLoader(
        #     torchvision.datasets.ImageFolder(root=data_var,
        #                                      transform=torchvision.transforms.ToTensor()),
        #     batch_size=args.B,
        #     shuffle=False, **kwargs)
        # return Dtr, Dte, Val
        return Dtr, Dte
