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
from args import args_parser
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


def load_data(file_name):
    """
    从CSV文件加载数据并进行预处理。
    - 使用均值填充缺失值
    - 对第3到第6列进行归一化
    """
    #     df = pd.read_csv('data/Wind/Task 1/Task1_W_Zone1_10/' + file_name + '.csv', encoding='gbk')
    df = pd.read_csv('/home/raoxy/data/carhacking_dirichlet_1/' + file_name + '.csv')
    columns = df.columns
    df.fillna(df.mean(numeric_only=True), inplace=True)
    for i in range(3, 7):
        MAX = np.max(df[columns[i]])
        MIN = np.min(df[columns[i]])
        df[columns[i]] = (df[columns[i]] - MIN) / (MAX - MIN)

    return df


class MyDataset(Dataset):
    """
    自定义PyTorch数据集类，用于处理风能数据。
    """

    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def nn_seq_wind(file_name, B):
    """
    处理风能数据以用于神经网络训练。
    - 加载数据
    - 分割为训练、验证和测试集
    - 创建用于训练的序列
    """
    print('数据处理中...')
    dataset = load_data(file_name)

    # 将数据集划分为训练集（train）、验证集（val）和测试集（test）。通过切片操作，根据数据集的长度将数据按照60%、20%和20%的比例进行划分。
    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]

    def process(data):
        '''
        该函数接受一个参数 data，并返回一个 DataLoader 对象
        '''
        columns = data.columns
        wind = data[columns[2]]  # 提取 data 的列名，并将第三列的数据赋值给变量 wind
        wind = wind.tolist()  # 然后将 wind 转换为列表，并重新赋值给 wind 变量
        data = data.values.tolist()  # 接下来，将 data 转换为列表格式
        seq = []  # 创建一个空列表存储训练序列和标签
        for i in range(len(data) - 30):
            train_seq = []
            train_label = []
            for j in range(i, i + 24):
                train_seq.append(wind[j])
            for c in range(3, 7):
                train_seq.append(data[i + 24][c])
            train_label.append(wind[i + 24])

            # 前24个数据作为训练特征（客户端的wind数据），后续4个数据作为训练标签
            train_seq = torch.FloatTensor(train_seq).view(-1)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        seq = MyDataset(seq)  # 创建一个数据集对象
        seq = DataLoader(dataset=seq, batch_size=B, shuffle=False, num_workers=0)

        return seq

    def process_c(data):
        from get_data import MyDataset
        from torch.utils.data import Dataset, DataLoader
        columns = data.columns
        wind = data[columns[-1]]  # 提取 data 的列名，并将第三列的数据赋值给变量 wind label
        wind = wind.tolist()  # 然后将 wind 转换为列表，并重新赋值给 wind 变量
        data = data.values.tolist()  # 接下来，将 data 转换为列表格式
        seq = []  # 创建一个空列表存储训练序列和标签
        import torch
        for i in range(len(data)):
            train_seq = []
            train_label = []
            train_seq.append(data[i])
            train_label.append(wind[i])

            # 前24个数据作为训练特征（客户端的wind数据），后续4个数据作为训练标签
            train_seq = torch.FloatTensor(train_seq).view(-1)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        seq = MyDataset(seq)  # 创建一个数据集对象
        seq = DataLoader(dataset=seq, batch_size=B, shuffle=False, num_workers=0)

        return seq

    Dtr = process_c(train)
    Val = process_c(val)
    Dte = process_c(test)

    return Dtr, Val, Dte


def get_mape(x, y):
    """
    计算MAPE（平均绝对百分比误差）
    :param x: 真实值
    :param y: 预测值
    :return: MAPE
    """
    return np.mean(np.abs((x - y) / x))


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
