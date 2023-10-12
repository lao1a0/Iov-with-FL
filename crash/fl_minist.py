import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import syft as sy  # <-- NEW: import the Pysyft library

"""
Part 06 - Federated Learning on MNIST using a CNN
http://localhost:8888/notebooks/git-home/github/PySyft/examples/tutorials/Part%2006%20-%20Federated%20Learning%20on%20MNIST%20using%20a%20CNN.ipynb
"""

"""
本例演示联邦学习CNN
"""


class Arguments():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 10
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 30
        self.save_model = False


# pysyft的hook
hook = sy.TorchHook(torch)
# 创建虚拟节点
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")

# 配置参数
args = Arguments()
# 使用cuda
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
# 设置worker
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# 联邦数据，数据分布在不同工作节点上
federated_train_loader = sy.FederatedDataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))
                                                 ]))
        .federate((bob, alice)),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False,
                   transform=transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))
                                                 ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


# 深度网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        print(x.size())
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# 训练
def train(args, model, device, federated_train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(federated_train_loader):
        # 把模型发给联邦学习节点
        model.send(data.location)
        data, target = data.to(device), target.to(device)
        # 把grad清零
        optimizer.zero_grad()
        output = model(data)
        # 计算损失
        loss = F.nll_loss(output, target)
        # 计算梯度
        loss.backward()
        optimizer.step()
        # 从远程节点更新模型
        model.get()
        if batch_idx % args.log_interval == 0:
            loss = loss.get()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(federated_train_loader) * args.batch_size,
                       100. * batch_idx / len(federated_train_loader), loss.item()))


# 测试
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    # 模型初始化
    model = Net().to(device)
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # 求解
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, federated_train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
    # 训练结果存盘
    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")

# class CNN(nn.Module):
# def __init__(self, input_shape, num_class):
#     super(CNN, self).__init__()#         定义网络结构，包括卷积层，池化层，全连接层等
#     self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,stride=1, padding=1)
#     # 输入通道数为input_shape[0]，输出通道数为64，卷积核大小为3x3，步长为1，填充为1
#     self.conv2 = nn.Conv2d(64, 64, 3, padding=1) # 输入通道数为64，输出通道数为64，卷积核大小为3x3，步长为1，填充为1
#     self.pool1 = nn.MaxPool2d(2) # 最大池化层，池化核大小为2x2
#     self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 输入通道数为64，输出通道数为128，卷积核大小为3x3，步长为1，填充为1
#     self.conv4 = nn.Conv2d(128, 128, 3, padding=1) # 输入通道数为128，输出通道数为128，卷积核大小为3x3，步长为1，填充为1
#     self.pool2 = nn.MaxPool2d(2) # 最大池化层，池化核大小为2x2
#     self.conv5 = nn.Conv2d(128, 64, 3 ,padding=1) # 输入通道数为128，输出通道数为256，卷积核大小为3x3，步长为1，填充为1
#     self.conv6 = nn.Conv2d(64 ,32 , 3 ,padding=1) # 输入通道数位256 ，输出通道数位256 ，卷积核大小位3x3 ，步长位1 ，填充位1
#     self.gap = nn.AdaptiveAvgPool2d(5) # 全局平均池化层
#     self.fc1 = nn.Linear(800,num_class) # 全连接层 ，输入特征维度位256 ，输出特征维度位num_class
#     self.relu = nn.ReLU() # 激活函数
#     self.dropout = nn.Dropout(p=0.5) # 随机失活层
#     self.softmax = nn.Softmax()
# def forward(self,x):
#     # print(x.size())
#     x=self.relu(self.conv1(x))
#     x=self.relu(self.conv2(x))
#     x=self.pool1(x)
#     x=self.relu(self.conv3(x))
#     x=self.relu(self.conv4(x))
#     x=self.pool2(x)
#     x=self.relu(self.conv5(x))
#     x=self.relu(self.conv6(x))
#     # print(x.size())
#     x=self.gap(x)
#     x=self.dropout(x)
#     # print(x.shape[0])
#     x = x.view(x.shape[0], -1)
#     # print(x.shape)
#     x=self.softmax(self.fc1(x))
#     return x
# def train(args, model, device, federated_train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(federated_train_loader):
#         # print(data.shape)
#         model.send(data.location) # <-- NEW: send the model to the right location
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         model.get() # <-- NEW: get the model back
#         if batch_idx % args.log_interval == 0:
#             loss = loss.get() # <-- NEW: get the loss back
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * args.batch_size, len(federated_train_loader) * args.batch_size,
#                 100. * batch_idx / len(federated_train_loader), loss.item()))
