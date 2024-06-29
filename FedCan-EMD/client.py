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
from torch import nn
from tqdm import tqdm
import pandas as pd
from get_data import carHacking_Data
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def get_val_loss(args, model, Val):
    model.eval()
    # loss_function = nn.MSELoss().to(args.device)
    loss_function = nn.CrossEntropyLoss().to(args.device)
    val_loss = []
    pred = []
    y = []
    for (seq, label) in Val:
        with torch.no_grad():
            seq = seq.to(args.device)
            # label = label.to(args.device)
            label = label.long().squeeze().to(args.device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)

            _, y_pred = torch.max(y_pred, dim=1)
            val_loss.append(loss.item())
            pred += y_pred.data.tolist()  # 将预测值添加到列表中
            y += label.data.tolist()  # 将真实值添加到列表中
    pred = np.array(y_pred.cpu())
    y = np.array(label.cpu())
    acc = accuracy_score(np.array(y), np.array(pred))
    return np.mean(val_loss), acc


def train(args, _model_, server, flag):
    if flag == "CNN":
        param = _model_.parameters()
        model = _model_
    else:
        model = _model_[0]
        param = _model_[0].parameters()

    Dtr, Dte = carHacking_Data(model.name, args)
    model.len = len(Dtr)
    global_model = copy.deepcopy(server)
    lr = args.lr

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param, lr=lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(param, lr=lr,
                                    momentum=0.9, weight_decay=args.weight_decay)

    stepLR = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    min_epochs = 1  # 最小的训练轮数
    best_model = None
    min_val_loss = 5
    print('model: {} training...'.format(model.name))
    # loss_function = nn.MSELoss().to(args.device)
    loss_function = nn.CrossEntropyLoss().to(args.device)

    train_loss_ = []
    val_loss_ = []
    val_acc_ = []

    # 计算两个张量之间的EMD距离
    def calculate_emd_distance(tensor_a, tensor_b):
        # 将张量移动到CPU并展平
        from scipy.stats import wasserstein_distance
        flattened_a = tensor_a.view(-1).cpu().detach().numpy()
        flattened_b = tensor_b.view(-1).cpu().detach().numpy()

        # 计算EMD距离
        emd = wasserstein_distance(flattened_a, flattened_b)
        return torch.tensor(emd)

    for epoch in tqdm(range(args.E)):
        train_loss = []
        model.train()
        for i, (seq, label) in enumerate(Dtr, 0):
            seq = seq.to(args.device)
            label = label.long().squeeze().to(args.device)
            y_pred = model(seq)
            optimizer.zero_grad()
            # 初始化近似（≈）为0
            proximal_term = 0.0
            # for w, w_t in zip(model.parameters(), global_model.parameters()):
            #     proximal_term += (w - w_t).norm(2)
            for w, w_t in zip(model.parameters(), global_model.parameters()):  # 对于每个模型参数和全局模型参数进行循环
                proximal_term += calculate_emd_distance(w, w_t).to(args.device)  # 计算模型参数和全局模型参数之间的差异，并使用L2范数来度量
            loss = loss_function(y_pred, label) + (args.mu / 2) * proximal_term  # 计算损失函数，包括预测值和标签之间的均方误差损失以及近似项（≈）。
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        stepLR.step()  # 调整学习率

        val_loss, val_acc = get_val_loss(args, model, Dte)

        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            #  如果达到最小训练轮数并且验证集损失小于最小验证集损失，则更新最小验证集损失和最佳模型
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        # print( 'epoch {:03d} train_loss {:.8f} val_loss {:.8f} val_acc {:.8f}'.format(epoch, np.mean(train_loss),
        # val_loss, val_acc))
        train_loss_.append(np.mean(train_loss))
        val_loss_.append(val_loss)
        val_acc_.append(val_acc)
        model.train()

    def _path(end, args):
        # 您想要检查的路径
        path = args.root_save_path + end
        # 检查路径是否存在
        if not os.path.exists(path):
            # 路径不存在，创建它
            os.makedirs(path)
        return path

    df = pd.DataFrame({'train_loss': train_loss_, 'val_loss': val_loss_, 'val_acc_': val_acc_})
    # 如果文件不存在，to_csv将会创建它
    if args.name == "CNN":
        df.to_csv(_path("CNN/csv/", args) + "{}.csv".format(model.name), mode='a',
                  header=not os.path.exists("{}{}.csv".format(_path("CNN/csv/", args), model.name)), index=False)
    elif args.name == "ResNet":
        df.to_csv(_path("ResNet/csv/", args) + "{}.csv".format(model.name), mode='a',
                  header=not os.path.exists("{}{}.csv".format(_path("ResNet/csv/", args), model.name)), index=False)
    elif args.name == "VGG":
        df.to_csv(_path("VGG/csv/", args) + "{}.csv".format(model.name), mode='a',
                  header=not os.path.exists("{}{}.csv".format(_path("VGG/csv/", args), model.name)), index=False)
    else:
        df.to_csv(_path("AlexNet/csv/", args) + "{}.csv".format(model.name), mode='a',
                  header=not os.path.exists("{}{}.csv".format(_path("AlexNet/csv/", args), model.name)), index=False)

    if flag == "CNN":
        return best_model
    else:
        return best_model, best_model.parameters()


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


def calculate_mcc(y_true, y_pred):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 提取混淆矩阵的值
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    # 计算MCC
    numerator = (TP * TN) - (FP * FN)
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    if denominator == 0:
        return 0.0  # 防止除以零
    mcc = numerator / denominator

    return mcc


def test(args, ann, flag=2):
    ann.eval()
    Dtr, Dte = carHacking_Data(ann.name, args, flag)
    pred = []
    y = []
    for (seq, target) in tqdm(Dte):
        with torch.no_grad():
            seq = seq.to(args.device)
            y_pred = ann(seq)
            y_pred = y_pred.argmax(1, keepdim=True)
            target = target.long()
            pred += y_pred.data.tolist()  # 将预测值添加到列表中
            y += target.data.tolist()  # 将真实值添加到列表中

    pred = np.array(pred)
    y = np.array(y)

    print(">车辆：{}".format(ann.name))
    acc = accuracy_score(y, pred)
    precision = precision_score(y, pred, labels=None, pos_label=1, zero_division=1, average='macro')
    recall = recall_score(y, pred, average='macro', zero_division=1)  # 'micro', 'macro', 'weighted'
    f1 = f1_score(y, pred, average='macro')
    mcc = 0  # calculate_mcc(y, pred)
    print('\t准确率-Acc:{}\n\t查准率-TP/(TP+FP):{}\n\t召回率-TP/(TP+FN):{}\n\tF1:{}\n\tmcc:{}'.format(acc, precision, recall, f1,
                                                                                             mcc))
    return acc, precision, recall, f1, mcc
