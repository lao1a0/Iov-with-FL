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
print("现在是在：",device)


class Arguments():
    def __init__(self):
        self.batch_size = 128
        self.epochs = 50
        self.lr = 0.01
        self.num_class = 5
        self.data_train = '../data2/train_quantile_ransformer_224/'
        self.data_test = '../data2/test_quantile_ransformer_224/'
        self.model_name = '../model/resnet18-5c106cde.pth'
        self.save_name = 'yeo_resnet18_lr0.01_epoch50'
        # self.model_name = '../model/vgg16-397923af.pth'
        # self.save_name = 'qtn_vgg16_lr0.01_epoch50'
        # self.model_name = '../model/vgg16-397923af.pth'
        # self.save_name = 'qtn_vgg16_lr0.01_epoch50'
        self.flag = True


args = Arguments()

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


# 迁移学习
#  wget https://download.pytorch.org/models/vgg16_bn-6c64b313.pth --no-check-certificate

from torchvision.models import inception_v3  # Inception 系列


def ResNet_s(args):
    ''':cvar
    返回修改好的模型，和冻结好的参数
    '''
    from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152  # ResNet系列
    if args.flag:

        pretrain_model = resnet18(pretrained=False)
        pretrain_model.fc = nn.Linear(pretrain_model.fc.in_features, 5)  # 将全连接层改为自己想要的分类输出
        pretrained_dict = torch.load(args.model_name)

        pretrained_dict.pop('fc.weight')
        pretrained_dict.pop('fc.bias')

        model_dict = pretrain_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)  # 模型参数列表进行参数更新，加载参数
        for a, b in model_dict.items():
            print(a, b.shape)

        pretrain_model.load_state_dict(model_dict)  # 将满足条件的参数的 requires_grad 属性设置为False

        for name, value in pretrain_model.named_parameters():
            if (name != 'fc.weight') and (name != 'fc.bias'):
                value.requires_grad = False
        params_conv = filter(lambda p: p.requires_grad, pretrain_model.parameters())  # 要更新的参数在parms_conv当中
        # pretrain_model = resnet18(pretrained=False, num_classes=5)
    else:
        pretrain_model = resnet18(pretrained=False, num_classes=5)
        params_conv = pretrain_model.parameters()

    return pretrain_model, params_conv

model, params_conv = ResNet_s(args)
# print(model)
# print([*zip(model.parameters())])
model = model.to(device)
optims = optim.SGD(params_conv, lr=args.lr)

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
        train_acc, test_acc, elapsed))

# 训练数据保存
torch.save(model.state_dict(), "../model/{}.pt".format(args.save_name))