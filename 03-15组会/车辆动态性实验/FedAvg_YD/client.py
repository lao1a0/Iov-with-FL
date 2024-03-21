# -*- coding:utf-8 -*-
"""
@Time: 2022/03/03 12:25
@Author: KI
@File: client.py
@Motto: Hungry And Humble
"""
import os
import copy
from itertools import chain
import pandas as pd
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from get_data import nn_seq_wind


def get_val_loss(args, model, Val):
    model.eval()
    loss_function = nn.CrossEntropyLoss().to(args.device)
    val_loss = []
    pred = []
    y = []
    for (seq, label) in Val:
        with torch.no_grad():
            seq = seq.to(args.device)
            label = label.long().squeeze().to(args.device)
            y_pred = model(seq)
            #             y_pred = y_pred.argmax(1, keepdim=True)
            loss = loss_function(y_pred, label)
            val_loss.append(loss.item())
            pred += y_pred.data.tolist()  # 将预测值添加到列表中
            y += label.data.tolist()  # 将真实值添加到列表中
    pred = np.array(y_pred.cpu())
    y = np.array(label.cpu())
    #     acc = accuracy_score(np.array(y), np.array(pred))
    return np.mean(val_loss), 0


def train(args, model, server):
    model.train()
    Dtr, Dte = nn_seq_wind(model.name, args.B)
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
    min_epochs = 1
    best_model = copy.deepcopy(model)
    min_val_loss = 5
    print('model: {} training...'.format(model.name))
    loss_function = nn.CrossEntropyLoss().to(args.device)
    train_loss_ = []
    val_loss_ = []
    val_acc_ = []
    for epoch in tqdm(range(args.E)):
        train_loss = []
        for (seq, label) in Dtr:
            seq = seq.to(args.device)
            label = label.long().squeeze().to(args.device)
            y_pred = model(seq)
            optimizer.zero_grad()
            # compute proximal_term
            # proximal_term = 0.0
            # for w, w_t in zip(model.parameters(), global_model.parameters()):
            #     proximal_term += (w - w_t).norm(2)

            loss = loss_function(y_pred, label) # + (args.mu / 2) * proximal_term
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        stepLR.step()
        # validation
        val_loss, _ = get_val_loss(args, model, Dte)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))

        train_loss_.append(np.mean(train_loss))
        val_loss_.append(val_loss)
        model.train()

    df = pd.DataFrame({'train_loss': train_loss_, 'val_loss': val_loss_})
    # 如果文件不存在，to_csv将会创建它
    df.to_csv("csv/{}.csv".format(model.name), mode='a', header=not os.path.exists("csv/{}.csv".format(model.name)),
              index=False)
    return best_model


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
    Dtr, Dte = nn_seq_wind(ann.name, args.B, flag)
    pred = []
    y = []
    for (seq, target) in tqdm(Dte):
        with torch.no_grad():
            seq = seq.to(args.device)
            y_pred = ann(seq)
            y_pred = y_pred.argmax(1, keepdim=True)
            target = target.squeeze().long()
            pred += y_pred.tolist()  # 将预测值添加到列表中
            y += target.tolist()  # 将真实值添加到列表中

    pred = np.array(pred)
    y = np.array(y)

    acc = accuracy_score(y, pred)
    precision = precision_score(y, pred, labels=None, pos_label=1, zero_division=1, average='macro')
    recall = recall_score(y, pred, average='macro', zero_division=1)  # 'micro', 'macro', 'weighted'
    f1 = f1_score(y, pred, average='macro')
    mcc = 0  # calculate_mcc(y, pred)

    print('\t准确率-Acc:{}\n\t查准率-TP/(TP+FP):{}\n\t召回率-TP/(TP+FN):{}\n\tF1:{}\n\tmcc:{}'.format(acc, precision, recall, f1,
                                                                                             mcc))
    return acc, precision, recall, f1, mcc
