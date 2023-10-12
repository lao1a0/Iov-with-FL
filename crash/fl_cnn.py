#!/usr/bin/env py
import torchvision
from torch.utils.data import DataLoader, random_split
import torch
import syft as sy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import xlwt

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'


class Arguments():
    def __init__(self):
        self.batch_size = 128
        self.test_batch_size = 1000
        self.epochs = 50
        self.lr = 0.05
        self.momentum = 0.5
        self.log_interval = 30
        self.num_class = 5
        self.save_name = 'yeo_cnn'


args = Arguments()

hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")

federated_train_loader = sy.FederatedDataLoader(
    torchvision.datasets.ImageFolder(root='../data/train_224/',
                                     transform=torchvision.transforms.ToTensor()).federate((bob, alice)),
    batch_size=args.batch_size,
    shuffle=True)

federated_test_loader = DataLoader(
    torchvision.datasets.ImageFolder(root='../data/test_224/',
                                     transform=torchvision.transforms.ToTensor()),
    batch_size=args.batch_size,
    num_workers=0,
    shuffle=False)


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


def train(model, device, federated_train_loader, optimizer):
    model.train()
    correct = 0
    sample_num = 0
    total_loss = 0
    train_batch_num = len(federated_train_loader)

    for idx, (data, target) in enumerate(federated_train_loader):
        model.send(data.location)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target.long())
        loss.backward()
        optimizer.step()

        total_loss += loss.get().data
        pred = output.argmax(1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().get()
        sample_num += len(pred)
        model.get()
    return total_loss / train_batch_num, correct.cpu().item() / sample_num


def test(model, device, federated_test_loader):
    model.eval()
    correct = 0
    total_loss = 0
    sample_num = 0
    test_batch_num = len(federated_test_loader)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(federated_test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = F.cross_entropy(output, target.long())
            total_loss += loss.data

            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum()
            sample_num += len(pred)

    return total_loss.cpu() / test_batch_num, correct.cpu().item() / sample_num


model = CNN(args.num_class).to(device)
optims = optim.SGD(model.parameters(), lr=args.lr)

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
time_list = []
timestart = time.perf_counter()
for epoch in range(1, args.epochs + 1):
    epochstart = time.perf_counter()  # 每一个epoch的开始时间
    train_loss, train_acc = train(model, device, federated_train_loader, optims)
    elapsed = (time.perf_counter() - epochstart)  # 每一个epoch的结束时间 记录训练的耗时
    test_loss, test_acc = test(model, device, federated_test_loader)
    # 保存各个指际
    train_loss_list.append(train_loss.cpu())
    train_acc_list.append(train_acc)
    test_loss_list.append(test_loss.cpu())
    test_acc_list.append(test_acc)
    time_list.append(elapsed)
    print('epoch %d, train_loss %.6f,test_loss %.6f,train_acc %.6f,test_acc %.6f,time cost %.6f' % (
        epoch, train_loss, test_loss,
        train_acc, test_acc,elapsed))

# 训练数据保存
torch.save(model.state_dict(), "../model/{}.pt".format(args.save_name))


def _change(a):
    b = []
    for i in a:
        b.append(float(i))
    return b


file_name = '../model/{}.xlsx'.format(args.save_name)

# 创建workbook和sheet对象
workboot = xlwt.Workbook(encoding='utf-8')
worksheet = workboot.add_sheet('result')  # 设置工作表的名字
# 写入Excel标题
row0 = ["Train loss", "Train acc", "Test loss", 'Test acc', 'Time']
for i in range(len(row0)):
    worksheet.write(0, i, row0[i])

test_loss_list = _change(test_loss_list)
train_loss_list = _change(train_loss_list)
train_acc_list = _change(train_acc_list)
test_acc_list = _change(test_acc_list)
time_list = _change(time_list)

length = len(test_loss_list)

for i in range(1, length + 1):
    worksheet.write(i, 0, train_loss_list[i - 1])
    worksheet.write(i, 1, train_acc_list[i - 1])
    worksheet.write(i, 2, test_loss_list[i - 1])
    worksheet.write(i, 3, test_acc_list[i - 1])
    worksheet.write(i, 4, time_list[i - 1])
workboot.save(file_name)

import matplotlib as mpl
from matplotlib import pyplot as plt


# mpl.use('nbAgg')
# mpl.style.use('seaborn-darkgrid')
import numpy as np
def plotP(test_loss, train_loss, train_acc_list, test_acc_list):
    plt.figure(figsize=(10, 10))
    x = np.linspace(0, len(train_loss), len(train_loss))
    y = np.linspace(0, len(train_acc_list), len(train_acc_list))
    plt.subplot(2, 1, 1)
    plt.plot(x, train_loss, label="train_loss")
    plt.plot(x, test_loss, label="test_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(y, train_acc_list, label="train_acc")
    plt.plot(y, test_acc_list, label="test_acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.savefig('../img/{}.png'.format(args.save_name))
    plt.show()


plotP(test_loss_list, train_loss_list, train_acc_list, test_acc_list)