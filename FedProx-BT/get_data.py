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

sys.path.append('../')
from torch.utils.data import Dataset, DataLoader

# args = args_parser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clients_wind = ['Task1_W_Zone' + str(i) for i in range(1, 11)]


def load_data(file_name):
    """
    从CSV文件加载数据并进行预处理。
    - 使用均值填充缺失值
    - 对第3到第6列进行归一化
    """
    fn = '/home/raoxy/data/carhacking_bt/{}.csv'.format(file_name)
    df = pd.read_csv(fn)
    # columns = df.columns
    # df.fillna(df.mean(numeric_only=True), inplace=True)
    # for i in range(3, 7):
    #     MAX = np.max(df[columns[i]])
    #     MIN = np.min(df[columns[i]])
    #     df[columns[i]] = (df[columns[i]] - MIN) / (MAX - MIN)

    return df


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def nn_seq_wind(file_name, B, flag=2):
    """
    - 加载数据
    - 分割为训练、验证和测试集
    - 创建用于训练的序列
    """
    print('数据处理中...')
    dataset = load_data(file_name)

    # 将数据集划分为训练集（train）、测试集（test）。
    train = dataset[:int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]

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

            train_seq = torch.FloatTensor(train_seq).view(-1)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        seq = MyDataset(seq)  # 创建一个数据集对象
        seq = DataLoader(dataset=seq, batch_size=B, shuffle=False, num_workers=0)

        return seq

    if flag == 1:
        Dte = process_c(test)
        return "", Dte
    else:
        Dtr = process_c(train)
        Dte = process_c(test)

        return Dtr, Dte


def get_mape(x, y):
    """
    :param x:true
    :param y:pred
    :return:MAPE
    """
    return np.mean(np.abs((x - y) / x))
