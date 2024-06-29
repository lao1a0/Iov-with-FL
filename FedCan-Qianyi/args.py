# -*- coding:utf-8 -*-
"""
@Time: 2022/03/03 11:52
@Author: KI
@File: args.py
@Motto: Hungry And Humble
"""
import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_save_path', default="/home/raoxy/experimental_result/Qian_yi/", help='save file')  # 迁移实验数据保存位置
    parser.add_argument('--name', default="CNN", help='save file')  # 迁移的模型类型
    parser.add_argument('--root', default="/home/raoxy/data/carhacking/", help='save file') # 加载的训练数据集 get_data.py
    # 做模型迁移实验的时候需要 --name --root 这两个参数
    parser.add_argument('--E', type=int, default=30, help='number of rounds of training')  # 客户端上，每个客户端上的训练轮数
    parser.add_argument('--r', type=int, default=30, help='number of communication rounds')  # 中央服务器上的更新轮数
    parser.add_argument('--K', type=int, default=10, help='number of total clients')  # K 表示客户端的总数
    parser.add_argument('--C', type=float, default=0.5, help='sampling rate')  # C 控制每轮参与训练的比例
    # K*C 才是实际上参与的客户端数量，也就是说，每次随机从K中选择 K*C 个客户端
    parser.add_argument('--input_dim', type=int, default=28, help='input dimension')  # 输入的分类任务维度（暂时用不上
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')  # 初始学习率
    parser.add_argument('--B', type=int, default=50, help='local batch size')  # 客户端上，批次大小
    parser.add_argument('--mu', type=float, default=0.01, help='proximal term constant')  # 近项常数 为0等同于FedAvg
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')  # 优化器类型
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')  # 优化器上的权重衰减因子，防止过拟合
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # 训练设备类型
    parser.add_argument('--step_size', type=int, default=10, help='step size')  # 学习率动态调整的步长，每10轮执行一次衰减
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='learning rate decay per global round')  # 学习率动态调整的衰减因子，每次衰减10%
        # clients = ['Task1_W_Zone' + str(i) for i in range(1, 2)]

    args = parser.parse_args()

    return args
