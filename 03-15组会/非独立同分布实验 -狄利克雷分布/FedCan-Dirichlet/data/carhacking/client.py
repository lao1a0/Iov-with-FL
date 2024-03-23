# -*- coding:utf-8 -*-
"""
@Time: 2022/03/03 12:25
@Author: KI
@File: client.py
@Motto: Hungry And Humble
"""
import copy
from itertools import chain
import os
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from tqdm import tqdm
import pandas as pd
from get_data import nn_seq_wind


def get_val_loss(args, model, Val):
    model.eval()
    # loss_function = nn.MSELoss().to(args.device)
    loss_function = nn.CrossEntropyLoss().to(args.device)
    val_loss = []
    for (seq, label) in Val:
        with torch.no_grad():
            seq = seq.to(args.device)
            # label = label.to(args.device)
            label = label.long().squeeze().to(args.device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            val_loss.append(loss.item())

    return np.mean(val_loss)


def train(args, model, server):
    model.train()
    Dtr, Val, Dte = nn_seq_wind(model.name, args.B)
    model.len = len(Dtr)
    global_model = copy.deepcopy(server)
    lr = args.lr
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    stepLR = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    print('model: {} training...'.format(model.name))
    # loss_function = nn.MSELoss().to(args.device)
    loss_function = nn.CrossEntropyLoss().to(args.device)
    import torch.nn.functional as F
    train_loss_ = []
    val_loss_ = []
    for epoch in tqdm(range(args.E)):
        train_loss = []
        for i, (seq, label) in enumerate(Dtr, 0):
            seq = seq.to(args.device)
            label = label.long().squeeze().to(args.device)
            y_pred = model(seq)
            optimizer.zero_grad()
            # 初始化近似（≈）为0
            proximal_term = 0.0
            for w, w_t in zip(model.parameters(), global_model.parameters()):  # 对于每个模型参数和全局模型参数进行循环
                proximal_term += (w - w_t).norm(2)  # 计算模型参数和全局模型参数之间的差异，并使用L2范数来度量
            # loss = loss_function(y_pred, label) + (args.mu / 2) * proximal_term  # 计算损失函数，包括预测值和标签之间的均方误差损失以及近似项（≈）。
            loss = loss_function(y_pred, label) # + (args.mu / 2) * proximal_term  # 计算损失函数，包括预测值和标签之间的均方误差损失以及近似项（≈）。
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        stepLR.step()  # 调整学习率
        # validation
        val_loss = get_val_loss(args, model, Val)

        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            #  如果达到最小训练轮数并且验证集损失小于最小验证集损失，则更新最小验证集损失和最佳模型
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        # print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        train_loss_.append(np.mean(train_loss))
        val_loss_.append(val_loss)
        model.train()

    df = pd.DataFrame({'train_loss': train_loss_, 'val_loss': val_loss_})
    # df.to_csv("csv/{}.csv".format(model.name), index=False)
    # 'my_data.csv'是您要写入的文件名
    # 如果文件不存在，to_csv将会创建它
    df.to_csv("csv/{}.csv".format(model.name), mode='a', header=not os.path.exists("csv/{}.csv".format(model.name)), index=False)

    printFigure(df, model.name)
    return best_model

def printFigure(df, name):
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import numpy as np
    df.plot(color=['#CD0056', '#0C755F'])
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.title('model:{}'.format(name))
    # ax2.legend(loc='center right')
    # 显示图形
    plt.savefig("fig/{}.png".format(name))
    plt.clf()


def test(args, ann):
    ann.eval()
    Dtr, Val, Dte = nn_seq_wind(ann.name, args.B)
    pred = []
    y = []
    for (seq, target) in tqdm(Dte):
        with torch.no_grad():
            seq = seq.to(args.device)
            y_pred = ann(seq)
            _, y_pred = torch.max(y_pred, dim=1)
            # print(len(y_pred))#.data.tolist())
            # print(len(target))
            target = target.long()
            # print("y_pred:{},{}".format(a, len(a)))
            # print("target:{},{}".format(b, len(b)))
            pred += y_pred.data.tolist()  # 将预测值添加到列表中
            y += list(chain.from_iterable(target.data.tolist()))  # 将真实值添加到列表中

    pred = np.array(pred)
    y = np.array(y)
    print('mae:', mean_absolute_error(y, pred), 'rmse:', np.sqrt(mean_squared_error(y, pred)))
