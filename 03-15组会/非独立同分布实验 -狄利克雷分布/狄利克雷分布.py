import pandas as pd
from sklearn.preprocessing import PowerTransformer
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import minmax_scale
import cv2
import os
from PIL import Image
import os
import random
import shutil
import warnings
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import datasets
from torch.utils.data import ConcatDataset

'''
find ./train/4 -type f | wc -l
'''
warnings.filterwarnings("ignore")

def deliver_data_to_car_dirichlet(root, alpha, n_clients=15):
    df = pd.read_csv("/home/raoxy/data/Car_Hacking_100.csv").sample(frac=1).reset_index(drop=True)
    n_classes = len(df.Label.value_counts())  # 获得分类数目
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    client = [pd.DataFrame(columns=[str(i) for i in df.columns]) for i in range(n_clients)]

    def dirichlet_non_iid(df, client):
        num_rows = len(df)
        for ld in label_distribution:  # 一共循环5次，5个类别
            np.random.shuffle(ld)
            split_sizes = (num_rows * ld).astype(int)  # 计算每个客户端的分割行数
            if split_sizes.sum() != num_rows:  # 调整最后一行的数量
                split_sizes[-1] += num_rows - split_sizes.sum()
            splits = []
            start_idx = 0
            for size in split_sizes:
                end_idx = start_idx + size
                splits.append(df.iloc[start_idx:end_idx])
                start_idx = end_idx

            # 验证每个分割的长度
            for i, split in enumerate(splits):
                client[i] = pd.concat([client[i], split])
            return client

    client = dirichlet_non_iid(df[df["Label"] == "R"].reset_index(drop=True), client)
    client = dirichlet_non_iid(df[df["Label"] == "RPM"].reset_index(drop=True), client)
    client = dirichlet_non_iid(df[df["Label"] == "gear"].reset_index(drop=True), client)
    client = dirichlet_non_iid(df[df["Label"] == "DoS"].reset_index(drop=True), client)
    client = dirichlet_non_iid(df[df["Label"] == "Fuzzy"].reset_index(drop=True), client)
    for i in range(len(client)):
        path = root + "car_{}/".format(i + 1)
        if not os.path.exists(path):
            os.makedirs(path)
        client[i].to_csv(path + "car_{}.csv".format(i + 1),index=False)

    R = []
    Rpm = []
    gear = []
    dos = []
    Fuzzy = []
    for i in range(len(client)):
        R.append(client[i].Label.value_counts()[0])
        Rpm.append(client[i].Label.value_counts()[1])
        gear.append(client[i].Label.value_counts()[2])
        dos.append(client[i].Label.value_counts()[3])
        Fuzzy.append(client[i].Label.value_counts()[4])
    ldf = pd.DataFrame({
        "R": R,
        "gear": gear,
        "DoS": dos,
        "RPM": Rpm,
        "Fuzzy": Fuzzy
    }).T
    ldf.columns = ["RSU " + str(i) for i in range(1, n_clients+1)]
    ldf = ldf.T
    df_normalized = ldf.div(ldf.max(axis=0), axis=1)
    df_normalized.to_csv("csv/dirichlet_{}.csv".format(alpha),index=False)


def load_can_data(name, root, f):
    numeric_features = ["CAN ID", "DATA[0]", "DATA[1]", "DATA[2]", "DATA[3]", "DATA[4]", "DATA[5]", "DATA[6]",
                        "DATA[7]"]
    df = pd.read_csv("{}{}/{}.csv".format(root, name, name))  # 每个客户端自己的训练数据保存路径：root/客户端名/客户端名.csv
    pre_Data = My_Pre(df, numeric_features)
    df = pre_Data.getItem()
    sd = Split_Data(root)
    sd.split(df, name, flag=f)
    df.to_csv("{}{}/pre_{}.csv".format(root, name, name), index=False)
    print("{} 生成成功！".format("{}{}/pre_{}.csv".format(root, name, name)))


if __name__ == '__main__':
    '''
    这里将数据集划分为20份了，但是只处理了12份
    carhacking_raw：没有任何预处理的原始图像
    carhacking_our：欠采样+yeo处理
    '''
    # root = '/home/raoxy/data/carhacking_raw/'
    # root = '/home/raoxy/data/carhacking_our/'
    # root = '/home/raoxy/data/carhacking_our_bt/'

    root = "/home/raoxy/data/carhacking_dirichlet_1/"
    # deliver_data_to_car_dirichlet(root, alpha=1, n_clients=15)

    # root = "/home/raoxy/data/carhacking_dirichlet_05/"
    # deliver_data_to_car_dirichlet(root, alpha=0.5, n_clients=15)

    # deliver_data_to_car(root, 15)  # 下发给区域内的12辆车
    #
    # for i in range(1, 12):
    #     load_can_data("car_" + str(i), root, f=2)

    for i in range(12, 13):
        load_can_data("car_" + str(i), root, f=1)
