# -*- coding:utf-8 -*-
"""
@Time: 2022/03/03 12:50
@Author: KI
@File: server.py
@Motto: Hungry And Humble

最重要的一个代码
"""
import copy
import random
from get_data import carHacking_Data
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from model import CNN, ResNet, VGG, Alexnet
from client import train, test, get_val_loss


class FedProx:
    def __init__(self, args):
        self.args = args
        self.clients = ['car_' + str(i) for i in range(1, args.K + 1)]
        self.global_loss_acc = {"accuracy": [], "loss": []}
        self.bt_non_idd = {f"car_{i}": {"accuracy": [], "precision": [], "recall": [], "f1": [], "mcc": []} for i in
                           range(1, args.K + 1)}
        self.nns = []
        self.nn = CNN(num_class=5, name="car_12").to(args.device)  # nn：全局模型，即服务器端的模型
        self.state_dict = torch.load('/home/raoxy/FedCan/pth/CNN.pth')
        self.nn.load_state_dict(self.state_dict)
        for i in range(self.args.K):
            temp = copy.deepcopy(self.nn)
            temp.name = self.clients[i]
            self.nns.append(temp)  # nns：每个客户端模型的集合

        # self.nn = ResNet(name="car_12")
        # self.nn_RestNet18[0].to(args.device)
        # self.nns_RestNet18 = []
        # for i in range(self.args.K):
        #     temp = copy.deepcopy(self.nn_RestNet18[0])
        #     temp.name = self.args.clients[i]
        #     self.nns_RestNet18.append((temp, self.nn_RestNet18[1]))  # nns_RestNet18：每个客户端模型的集合

        # self.nn_VGG16 = VGG()
        # self.nn_VGG16[0].to(args.device)
        # self.nns_VGG16 = []
        # for i in range(self.args.K):
        #     temp = copy.deepcopy(self.nn_VGG16[0])
        #     temp.name = self.args.clients[i]
        #     self.nns_VGG16.append((temp, self.nn_VGG16[1]))  # nns_VGG16：每个客户端模型的集合

        # self.nn_AlexNet = Alexnet()
        # self.nn_AlexNet[0].to(args.device)
        # self.nns_AlexNet = []
        # for i in range(self.args.K):
        #     temp = copy.deepcopy(self.nn_AlexNet[0])
        #     temp.name = self.args.clients[i]
        #     self.nns_AlexNet.append((temp, self.nn_AlexNet[1]))  # nns_AlexNet：每个客户端模型的集合

    def server(self, name):
        '''关键函数1：服务端的主函数'''
        for t in tqdm(range(self.args.r)):
            print('round', t + 1, ':')
            m = np.max([int(self.args.C * self.args.K), 1])  # 计算随机选择的客户端数量 m，参数 C 控制每轮参与训练的比例，参数 K 表示客户端的总数
            index = random.sample(range(0, self.args.K), m)  # random.sample 方法打乱m个客户端的顺序
            print("idex：{}".format(index))
            self.dispatch(index, flag=name)  # 将全局模型参数分发给被选择的客户端模型
            self.client_update(index, flag=name)  # 客户端使用收到的全局模型参数进行本地模型训练，并更新自己的模型参数
            self.aggregation(index, flag=name)  # 根据客户端的更新情况对全局模型参数进行聚合，以得到新的全局模型参数
            # self.global_test(name)
            _, Val = carHacking_Data(self.nn.name, self.args, flag=1)
            gl, ga = get_val_loss(self.args, self.nn, Val)
            self.global_loss_acc['accuracy'].append(ga)
            self.global_loss_acc['loss'].append(gl)

        df = pd.DataFrame(self.global_loss_acc)
        df.to_csv("fig/global_loss_acc.csv", index=False)
        print("保存全局模型在car 12上的acc和loss成功：fig/global_loss_acc.csv")
        for car, data in self.bt_non_idd.items():
            df = pd.DataFrame(data)
            df.to_csv(f"csv/F1_{car}.csv", index=False)
            print(f"csv/F1_{car}.csv 保存成功。")

        torch.save(self.nn.state_dict(), "pth/{}.pth".format(name))
        print("保存全局模型：{} 成功".format("pth/{}.pth".format(name)))

        for i in self.nns:
            torch.save(i.state_dict(), "model/{}.pth".format(i.name))
            print("保存客户端模型：{} 成功".format("model/{}.pth".format(i.name)))
        if name == "CNN":
            return self.nn
        elif name == "ResNet":
            return self.nn_RestNet18[0]
        elif name == "VGG":
            return self.nn_VGG16[0]
        else:
            return self.nn_AlexNet[0]

    def aggregation(self, index, flag):
        ''''
        关键函数2：客户端聚合函数
            index：一个数字列表，每一个item是一个客户端的编号
        '''

        def aggregation_model(nn, nns, f):
            if f == "CNN":
                s = 0  # 计算被选中客户端模型的总样本数量，并将其存储在变量 s 中，以便后续加权平均使用
                for j in index:
                    s += nns[j].len
                params = {}  # 存储每一个客户端聚合后的参数，初始值为全局模型参数的零张量，以便稍后进行累加计算。
                for k, v in nns[0].named_parameters():
                    params[k] = torch.zeros_like(v.data)

                for j in index:
                    # 对选中的客户端模型的参数进行加权累加。使用加权平均的方法，其中权重是选中客户端的样本数量与总样本数量的比值。
                    # 这意味着训练样本更多的客户端对全局模型参数的影响权重更大。
                    for k, v in nns[j].named_parameters():
                        params[k] += v.data * (nns[j].len / s)

                for k, v in nn.named_parameters():  # 将计算得到的聚合参数复制到全局模型中，使全局模型得以数据处理中更新
                    v.data = params[k].data.clone()

            else:
                s = 0
                for j in index:
                    s += nns[j][0].len
                params = {}
                for k, v in nns[0][0].named_parameters():
                    params[k] = torch.zeros_like(v.data)

                for j in index:
                    for k, v in nns[j][0].named_parameters():
                        params[k] += v.data * (nns[j][0].len / s)

                for k, v in nn[0].named_parameters():
                    v.data = params[k].data.clone()

            return nn, nns

        if flag == "CNN":
            self.nn, self.nns = aggregation_model(self.nn, self.nns, f="CNN")
        elif flag == "ResNet":
            self.nn_RestNet18, self.nns_RestNet18 = aggregation_model(self.nn_RestNet18, self.nns_RestNet18, f="ResNet")
        elif flag == "VGG":
            self.nn_VGG16, self.nns_VGG16 = aggregation_model(self.nn, self.nns, f="VGG16")
        elif flag == "AlexNet":
            self.nn_AlexNet, self.nns_AlexNet = aggregation_model(self.nn, self.nns, f="AlexNet")

    def dispatch(self, index, flag):
        '''
        将全局模型的参数分发给选中的客户端模型，以确保每个客户端在开始训练之前都具有最新的全局模型参数。
         同时迭代选中的客户端模型 self.nns[j] 的参数和全局模型 self.nn 的参数。意味着对于客户端模型和全局模型都会迭代对应的参数
        '''
        if flag == "CNN":
            for j in index:  # 遍历被选中的客户端模型的索引列表
                for old_params, new_params in zip(self.nns[j].parameters(), self.nn.parameters()):
                    old_params.data = new_params.data.clone()  # 将全局模型的参数值（new_params.data）复制到对应的客户端模型参数
        elif flag == "ResNet":
            for j in index:
                for old_params, new_params in zip(self.nns_RestNet18[j][0].parameters(),
                                                  self.nn_RestNet18[0].parameters()):
                    old_params.data = new_params.data.clone()
        elif flag == "VGG":
            for j in index:
                for old_params, new_params in zip(self.nns_VGG16[j][0].parameters(), self.nn_VGG16[0].parameters()):
                    old_params.data = new_params.data.clone()
        elif flag == "AlexNet":
            for j in index:
                for old_params, new_params in zip(self.nns_AlexNet[j][0].parameters(), self.nn_AlexNet[0].parameters()):
                    old_params.data = new_params.data.clone()

    def client_update(self, index, flag):
        '''
        对选中的客户端模型进行训练，并将其更新为全局模型的最新参数
        每个被选中的客户端模型，代码调用了名为 train 的函数，该函数接收客户端模型 self.nns[k]、全局模型 self.nn 和参数 self.args
        作为输入。被选中的客户端模型会使用全局模型来进行训练，并将训练后的模型更新为最新参数。
        '''
        if flag == "CNN":
            for k in index:
                self.nns[k] = train(self.args, self.nns[k], self.nn, flag="CNN")
        elif flag == "ResNet":
            for k in index:
                self.nns_RestNet18[k] = train(self.args, self.nns_RestNet18[k], self.nn_RestNet18[0], flag="ResNet")
        elif flag == "VGG":
            for k in index:
                self.nns_VGG16[k] = train(self.args, self.nns_VGG16[k], self.nn_VGG16[0], flag="VGG16")
        elif flag == "AlexNet":
            for k in index:
                self.nns_AlexNet[k] = train(self.args, self.nns_AlexNet[k], self.nn_AlexNet[0], flag="AlexNet")

    def global_test(self, name):
        # 在全局模型上进行测试，针对每个客户端模型进行测试，并记录测试结果
        if name == "CNN":
            model = copy.deepcopy(self.nn)
        elif name == "ResNet":
            model = copy.deepcopy(self.nn_RestNet18[0])
        elif name == "VGG":
            model = copy.deepcopy(self.nn_VGG16[0])
        else:
            model = copy.deepcopy(self.nn_AlexNet[0])
        model.eval()
        for client in self.clients:
            model.name = client
            acc, precision, recall, f1, mcc = test(self.args, model)
            self.bt_non_idd[client]["accuracy"].append(acc)
            self.bt_non_idd[client]["precision"].append(precision)
            self.bt_non_idd[client]["recall"].append(recall)
            self.bt_non_idd[client]["f1"].append(f1)
            self.bt_non_idd[client]["mcc"].append(mcc)
