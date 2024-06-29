# -*- coding:utf-8 -*-
"""
@Time: 2022/03/03 12:50
@Author: KI
@File: server.py
@Motto: Hungry And Humble
"""
import copy
import random
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from model import SimpleClassifier
from client import train, test,get_val_loss
from get_data import nn_seq_wind

class FedProx:
    def __init__(self, args):
        self.global_loss_acc = {"accuracy": [], "loss": []}
        self.args = args
        self.nn = SimpleClassifier(name='car_12').to(args.device)
        self.nns = []
        self.bt_non_idd = {f"car_{i}": {"accuracy": [], "precision": [], "recall": [], "f1": [], "mcc": []} for i in
                           range(1, args.K + 1)}
        self.clients = ['car_' + str(i) for i in range(1, self.args.K + 1)]
        for i in range(self.args.K):
            temp = copy.deepcopy(self.nn)
            temp.name = self.clients[i]
            self.nns.append(temp)

    def server(self):
        name = "CNN"
        for t in tqdm(range(self.args.r)):
            print('round', t + 1, ':')
            # sampling
            m = np.max([int(self.args.C * self.args.K), 1])
            index = random.sample(range(0, self.args.K), m)  # st
            print("idex：{}".format(index))
            self.dispatch(index)
            # local updating
            self.client_update(index)
            # aggregation
            self.aggregation(index)

            # self.global_test(name)
            _, Val = nn_seq_wind(self.nn.name, self.args.B, flag=1)
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

        torch.save(self.nn.state_dict(), "model/{}.pth".format(name))
        print("保存全局模型：{}成功".format("model/{}.pth".format(name)))
        for i in self.nns:
            torch.save(i.state_dict(), "model/{}.pth".format(i.name))
            print("保存客户端模型：{} 成功".format("model/{}.pth".format(i.name)))
        return self.nn

    def aggregation(self, index):
        s = 0
        for j in index:
            # normal
            s += self.nns[j].len

        params = {}
        for k, v in self.nns[0].named_parameters():
            params[k] = torch.zeros_like(v.data)

        for j in index:
            for k, v in self.nns[j].named_parameters():
                params[k] += v.data * (self.nns[j].len / s)

        for k, v in self.nn.named_parameters():
            v.data = params[k].data.clone()

    def dispatch(self, index):
        for j in index:
            for old_params, new_params in zip(self.nns[j].parameters(), self.nn.parameters()):
                old_params.data = new_params.data.clone()

    def client_update(self, index):  # update nn
        for k in index:
            self.nns[k] = train(self.args, self.nns[k], self.nn)

    def global_test(self, name):
        model = copy.deepcopy(self.nn)
        model.eval()
        for client in self.clients:
            model.name = client
            acc, precision, recall, f1, mcc = test(self.args, model)
            self.bt_non_idd[client]["accuracy"].append(acc)
            self.bt_non_idd[client]["precision"].append(precision)
            self.bt_non_idd[client]["recall"].append(recall)
            self.bt_non_idd[client]["f1"].append(f1)
            self.bt_non_idd[client]["mcc"].append(mcc)
