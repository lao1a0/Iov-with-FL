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

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from model import CNN, ResNet, VGG, Alexnet
from client import train, test, get_val_loss
from get_data import carHacking_Data


class FedProx:
    def __init__(self, args):
        self.args = args
        self.global_loss_acc = {"accuracy": [], "loss": []}
        self.bt_non_idd = {f"car_{i}": {"accuracy": [], "precision": [], "recall": [], "f1": [], "mcc": []} for i in
                           range(1, args.K + 1)}  # 存储每个round，全局模型在car上的F1 score 这些值
        self.clients = ['car_' + str(i) for i in range(1, args.K + 1)]

        self.nn = CNN(num_class=5, name="car_12").to(args.device)  # nn：全局模型，即服务器端的模型
        self.state_dict = torch.load('/home/raoxy/data/BestModel/preCNNbest.pth')
        self.nn.load_state_dict(self.state_dict)
        self.nns = []
        for i in range(self.args.K):
            temp = copy.deepcopy(self.nn)
            temp.name = self.clients[i]
            self.nns.append(temp)  # nns：每个客户端模型的集合

        self.nn_RestNet18 = ResNet(name="car_12")
        self.nn_RestNet18[0].to(args.device)
        self.nn_RestNet18[0].name = "car_12"
        self.nns_RestNet18 = []
        for i in range(self.args.K):
            temp = copy.deepcopy(self.nn_RestNet18[0])
            temp.name = self.clients[i]
            self.nns_RestNet18.append((temp, self.nn_RestNet18[1]))  # nns_RestNet18：每个客户端模型的集合

        self.nn_VGG16 = VGG(name="car_12")
        self.nn_VGG16[0].to(args.device)
        self.nn_VGG16[0].name="car_12"
        self.nns_VGG16 = []
        for i in range(self.args.K):
            temp = copy.deepcopy(self.nn_VGG16[0])
            temp.name = self.clients[i]
            self.nns_VGG16.append((temp, self.nn_VGG16[1]))  # nns_VGG16：每个客户端模型的集合

        self.nn_AlexNet = Alexnet(name="car_12")
        self.nn_AlexNet[0].to(args.device)
        self.nn_AlexNet[0].name="car_12"
        self.nns_AlexNet = []
        for i in range(self.args.K):
            temp = copy.deepcopy(self.nn_AlexNet[0])
            temp.name = self.clients[i]
            self.nns_AlexNet.append((temp, self.nn_AlexNet[1]))  # nns_AlexNet：每个客户端模型的集合

    def server(self, name):
        '''关键函数1：服务端的主函数'''
        for t in tqdm(range(self.args.r)):
            print('round', t + 1, ':')
            m = np.max([int(self.args.C * self.args.K), 1])
            index = random.sample(range(0, self.args.K), m)
            print("idex：{}".format(index))
            self.dispatch(index, flag=name)
            self.client_update(index, flag=name)
            self.aggregation(index, flag=name)

            # 测试每个客户端上的精度 F1/acc/prec这些
            self.global_test(name)

            # 测试全局模型的精度 loss/acc这些
            if name == "CNN":
                _, Val = carHacking_Data(self.nn.name, self.args, flag=1)
            elif name == "ResNet":
                _, Val = carHacking_Data(self.nn_RestNet18[0].name, self.args, flag=1)
            elif name == "VGG":
                _, Val = carHacking_Data(self.nn_VGG16[0].name, self.args, flag=1)
            else:
                _, Val = carHacking_Data(self.nn_AlexNet[0].name, self.args, flag=1)
            gl, ga = get_val_loss(self.args, self.nn, Val)
            self.global_loss_acc['accuracy'].append(ga)
            self.global_loss_acc['loss'].append(gl)

        self.save_model_and_file(name)

    def save_model_and_file(self, name):
        '''保存训练好的模型'''
        self.args.root_save_path = "/home/raoxy/experimental_result/Qian_yi/"
        if name == "CNN":
            self.save_file(self.args.root_save_path + "CNN/")
            torch.save(self.nn.state_dict(), self.args.root_save_path + "CNN/pth/{}.pth".format(name))
            print("保存全局模型：{} 成功".format(self.args.root_save_path + "CNN/pth/{}.pth".format(name)))
            for i in self.nns:
                torch.save(i.state_dict(), self.args.root_save_path + "CNN/model/{}.pth".format(i[0].name))
                print("保存客户端模型：{} 成功".format(self.args.root_save_path + "CNN/model/{}.pth".format(i[0].name)))

        elif name == "ResNet":
            self.save_file(self.args.root_save_path + "ResNet/")
            torch.save(self.nn_RestNet18[0].state_dict(), self.args.root_save_path + "ResNet/pth/{}.pth".format(name))
            print("保存全局模型：{} 成功".format(self.args.root_save_path + "ResNet/pth/{}.pth".format(name)))
            for i in self.nns_RestNet18:
                torch.save(i[0].state_dict(), self.args.root_save_path + "ResNet/model/{}.pth".format(i[0].name))
                print("保存客户端模型：{} 成功".format(self.args.root_save_path + "ResNet/model/{}.pth".format(i[0].name)))

        elif name == "VGG":
            self.save_file(self.args.root_save_path + "VGG/")
            torch.save(self.nn_VGG16[0].state_dict(), self.args.root_save_path + "VGG/pth/{}.pth".format(name))
            print("保存全局模型：{} 成功".format(self.args.root_save_path + "VGG/pth/{}.pth".format(name)))
            for i in self.nns_VGG16:
                torch.save(i[0].state_dict(), self.args.root_save_path + "VGG/model/{}.pth".format(i[0].name))
                print("保存客户端模型：{} 成功".format(self.args.root_save_path + "VGG/model/{}.pth".format(i[0].name)))

        else:
            self.save_file(self.args.root_save_path + "AlexNet/")
            torch.save(self.nn_AlexNet[0].state_dict(), self.args.root_save_path + "AlexNet/pth/{}.pth".format(name))
            print("保存全局模型：{} 成功".format(self.args.root_save_path + "AlexNet/pth/{}.pth".format(name)))
            for i in self.nns_AlexNet:
                torch.save(i[0].state_dict(), self.args.root_save_path + "AlexNet/model/{}.pth".format(i[0].name))
                print("保存客户端模型：{} 成功".format(self.args.root_save_path + "AlexNet/model/{}.pth".format(i[0].name)))

    def save_file(self, path):
        '''保存训练的时候输出的文件'''

        for car, data in self.bt_non_idd.items():
            df = pd.DataFrame(data)
            df.to_csv(path + "csv/F1_{}.csv".format(car), index=False)
            print("保存客户端模型在本地验证集上的评价指标(F1)：{}csv/F1_{}.csv 保存成功".format(path, car))

        df = pd.DataFrame(self.global_loss_acc)
        df.to_csv(path + "csv/global_loss_acc.csv", index=False)
        print("保存全局模型在Car12上的评价指标成功：{}csv/global_loss_acc.csv".format(path))

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
            self.nn_VGG16, self.nns_VGG16 = aggregation_model(self.nn_VGG16, self.nns_VGG16, f="VGG16")
        elif flag == "AlexNet":
            self.nn_AlexNet, self.nns_AlexNet = aggregation_model(self.nn_AlexNet, self.nns_AlexNet, f="AlexNet")

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
