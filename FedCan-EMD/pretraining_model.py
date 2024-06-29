from torch.utils.data import DataLoader
import torchvision
from model import CNN,ResNet,VGG,Alexnet
import torch
from get_data import carHacking_Data
from client import get_val_loss
from torch.optim.lr_scheduler import StepLR
from torch import nn
from tqdm import tqdm
import pandas as pd
import copy
from itertools import chain
import os
import numpy as np
from client import test
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Args:
    def __init__(self):
        self.B = 200
        self.root = '/home/raoxy/data/carhacking_our/'
        self.optimizer = "adam"
        self.lr = 0.01
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_decay = 0.0001
        self.E = 80
        self.gamma = 0.1
        self.step_size = 50


args = Args()

# m = CNN(num_class=5).to(args.device)
# m,_ = ResNet()
# m,_ = VGG()
# m,_ = Alexnet()
# m.to(args.device)
# m.name = "car_11"


def Train(args, model, name):
    model.train()
    param = model.parameters()
    Dtr, Dte = carHacking_Data(file_name=model.name, args=args)
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
    loss_function = nn.CrossEntropyLoss().to(args.device)

    train_loss_ = []
    val_loss_ = []
    for epoch in tqdm(range(args.E)):
        train_loss = []
        for i, (seq, label) in enumerate(Dtr, 0):
            seq = seq.to(args.device)
            label = label.long().squeeze().to(args.device)
            y_pred = model(seq)
            optimizer.zero_grad()
            loss = loss_function(y_pred, label)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        stepLR.step()  # 调整学习率
        val_loss, val_acc = get_val_loss(args, model, Dte)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            #  如果达到最小训练轮数并且验证集损失小于最小验证集损失，则更新最小验证集损失和最佳模型
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print(
            'epoch {:03d} train_loss {:.8f} val_loss {:.8f} val_acc {:.8f}'.format(epoch, np.mean(train_loss), val_loss,
                                                                                   val_acc))

        train_loss_.append(np.mean(train_loss))
        val_loss_.append(val_loss)

    torch.save(best_model.state_dict(), "{}.pth".format(name + "best"))
    # torch.save(model.state_dict(), "{}.pth".format(name))
    print("模型保存：{}".format(name + "best"))
    # print("模型保存：{}".format(name))
    # df = pd.DataFrame({'train_loss': train_loss_, 'val_loss': val_loss_})
    # df.to_csv("{}.csv".format(name), mode='a', header=not os.path.exists("csv/{}.csv".format(name)), index=False)
    return best_model


class My_Train:
    def __init__(self, device):
        self.loss_train = []
        self.acc_train = []
        self.loss_test = []
        self.acc_test = []
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(device)

    def train(self, device, model, train_loader, optimizer, batch_size):
        model.train()
        train_loss = 0
        correct = 0.0
        n = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            n += target.shape[0]
            loss = self.criterion(output, target.long())
            loss.backward()
            optimizer.step()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        self.loss_train.append(train_loss)
        self.acc_train.append(correct * 1.0 / n)

        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.8f}%)'.format(
            train_loss, correct, len(train_loader) * batch_size,
                                 100.0 * correct / n))
        return copy.deepcopy(model)

    def test(self, device, model, test_loader, batch_size):
        model.eval()
        test_loss = 0
        correct = 0.0
        n = 0.0
        p = []
        y = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                n += target.shape[0]
                loss = self.criterion(output, target.long())
                test_loss += loss.item()
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                p += pred.data.tolist()
                y += target.data.tolist()

        test_loss /= len(test_loader)
        self.loss_test.append(test_loss)
        self.acc_test.append(correct * 1.0 / n)

        # acc = accuracy_score(np.array(y), np.array(p))
        # precision = precision_score(np.array(y), np.array(p), labels=None, pos_label=1, zero_division=1,
                                    # average='macro')
        # recall = recall_score(np.array(y), np.array(p), average='macro')  # 'micro', 'macro', 'weighted'
        # f1 = f1_score(np.array(y), np.array(p), average='macro')
        # print('\t准确率-Acc:{}\n\t查准率-TP/(TP+FP):{}\n\t召回率-TP/(TP+FN):{}\n\tF1:{}'.format(acc, precision, recall, f1))
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader) * batch_size,
                                100.0 * correct / n))

    def start(self, args, model, train_loader, test_loader, optimizer):
        for epoch in range(args.E):
            model = self.train(self.device, model, train_loader, optimizer, batch_size=args.B)
            self.test(self.device, model, test_loader, batch_size=args.B)
        return copy.deepcopy(model)

# name = "pth/preCNN"
# name = "pth/preResNet"
# name = "pth/preVGG"
# name = "pth/AlexNet"
# Train(args, m, name)
# test(args, m)
