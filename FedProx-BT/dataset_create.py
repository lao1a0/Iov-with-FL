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
import matplotlib.pyplot as plt  # 导入绘图包

'''
find ./train/4 -type f | wc -l
'''
warnings.filterwarnings("ignore")


class My_Pre:
    def __init__(self, df, numeric_features):
        self.df = df
        self.df = self._NearMiss(df, numeric_features)
        self._print_info("_NearMiss处理后的")
        self.df = self._yeo_johnson(self.df, numeric_features)
        self._print_info("_yeo_johnson处理后的")
        self.df = self._minmax_scale(self.df, numeric_features)

    def _yeo_johnson(self, df, numeric_features):
        scaler = PowerTransformer(method="yeo-johnson")
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
        return df

    def _NearMiss(self, df, numeric_features):
        nm_1 = NearMiss()  # 形参默认 version=1, 即采用 NearMiss-1
        X_resampled, y_resampled = nm_1.fit_resample(df[numeric_features], df["Label"])
        return pd.merge(X_resampled, y_resampled, left_index=True, right_index=True)

    def _minmax_scale(self, df, numeric_features):
        df[numeric_features] = minmax_scale(df[numeric_features])
        df[numeric_features] = df[numeric_features].apply(lambda x: (x * 255))
        return df

    def _print_info(self, massage):
        print(massage)
        print(self.df.describe())

    def getItem(self):
        return self.df


class Split_Data:
    def __init__(self, root):
        self.root = root

    def generate_image(self, DF, ditrr, image_p=""):
        count = 0  # 定义一个计数器，用来记录读取的数据个数
        ims = []  # 定义一个空列表，用来存储读取的数据
        image_path = ditrr + image_p  # 定义一个变量，用来表示图片的保存路径，由两个字符串拼接而成
        os.makedirs(image_path, exist_ok=True)  # 调用os模块的makedirs函数，创建图片的保存路径，如果路径已存在，则不报错

        for i in range(0, len(DF)):
            count = count + 1  # 每遍历一个索引，计数器加一
            if count <= 27:  # 如果计数器小于等于27，说明还没有读取足够的数据
                im = DF.iloc[i].values  # 用iloc函数根据索引获取数据框的一行数据，返回一个数组
                ims = np.append(ims, im)  # 用np.append函数将数组添加到列表中
            else:
                ims = np.array(ims).reshape(9, 9, 3)  # 用np.array函数将列表转换为一个一维数组，然后用reshape函数将数组重塑为一个9x9x3的三维数组
                array = np.array(ims, dtype=np.uint8)  # 用np.array函数将三维数组转换为一个无符号8位整数类型的数组，这是图片的数据类型
                new_image = Image.fromarray(array)  # 用Image模块的fromarray函数将数组转换为一张图片
                new_image.save(image_path + str(i) + ".png")  # 用save函数将图片保存到指定的路径，图片的名字由路径、索引和后缀拼接而成
                # 图像放大
                #             img = cv2.imread(image_path+str(i)+".png")
                #             img = cv2.resize(img, (224, 224))
                #             cv2.imwrite(image_path+str(i)+".png", img)
                #             new_image = Image.fromarray(array)
                #             new_image.save(image_path+str(i)+".png")
                count = 0  # 重置计数器为0，准备读取下一批数据
                ims = []  # 重置列表为空，准备存储下一批数据

    def data_classification(self, df):
        df0 = df[df["Label"] == "R"].drop(["Label"], axis=1)
        df1 = df[df["Label"] == "RPM"].drop(["Label"], axis=1)
        df2 = df[df["Label"] == "gear"].drop(["Label"], axis=1)
        df3 = df[df["Label"] == "DoS"].drop(["Label"], axis=1)
        df4 = df[df["Label"] == "Fuzzy"].drop(["Label"], axis=1)
        return df0, df1, df2, df3, df4

    def count_img(self, root_dir):
        file_count = 0
        # 遍历目录及其所有子目录
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                # 计数器加一，因为我们找到了一个文件
                file_count += 1
        print(f"在目录 {root_dir} 及其子目录中共找到 {file_count} 个文件。")
        # allimgs = []
        # for subdir in os.listdir(Train_Dir):
        #     for filename in os.listdir(os.path.join(Train_Dir, subdir)):
        #         filepath = os.path.join(Train_Dir, subdir, filename)
        #         allimgs.append(filepath)
        return file_count

    def split_imag_data(self, Numbers, source_dir, target_dir):
        '''
        allimgs：当前图片总数
        Numbers：目标移动图片总数
        源路径：Train_Dir
        目标路径：Val_Dir
        '''
        # 获取源目录中所有文件的列表
        all_files = os.listdir(source_dir)
        image_extensions = ('.png')
        all_images = [file for file in all_files if file.lower().endswith(image_extensions)]
        # 确保源目录中有足够的图片可以移动
        if len(all_images) < Numbers:
            print(f"\t\t源目录中的图片数量({len(all_images)})少于想要移动的数量({Numbers})。")
        else:
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            selected_images = random.sample(all_images, Numbers)
            for image in selected_images:
                # 构建源文件和目标文件的完整路径
                source_image_path = os.path.join(source_dir, image)
                target_image_path = os.path.join(target_dir, image)
                # 移动文件
                shutil.move(source_image_path, target_image_path)
            print(f"\t\t成功从 {source_dir} 移动 {Numbers} 张图片到 {target_dir}。")

    def split(self, df, name, flag):
        # 数据按照label分类：先全部放到train里面
        # df = pd.read_csv(save_csv)
        Train_Dir = self.root + name + "/train/"
        Test_Dir = self.root + name + "/test/"

        df0, df1, df2, df3, df4 = self.data_classification(df)
        self.generate_image(df0, ditrr=Train_Dir, image_p="0/")  # normal
        self.generate_image(df1, ditrr=Train_Dir, image_p="1/")
        self.generate_image(df2, ditrr=Train_Dir, image_p="2/")
        self.generate_image(df3, ditrr=Train_Dir, image_p="3/")  # dos attack
        self.generate_image(df4, ditrr=Train_Dir, image_p="4/")  # fuzzy attack

        # 数据集划分
        if flag == 3:
            '''划分为：train vs test vs verify  6:2:2'''
            allimgs = self.count_img(Train_Dir)  # 更新一下现在的训练集
            Numbers, Numbers_ = allimgs // 5, allimgs // 5
            print("测试集 | {} ({:.4f})  \n验证集 | {} ({:.4f})\n训练集 | {} ({:.4f})".format(Numbers, Numbers / allimgs,
                                                                                     Numbers_,
                                                                                     Numbers_ / allimgs,
                                                                                     allimgs - Numbers - Numbers,
                                                                                     (
                                                                                             allimgs - Numbers - Numbers_) / allimgs))
            for i in range(5):
                d = Train_Dir + str(i) + "/"
                td = Test_Dir + str(i) + "/"
                allimgs = self.count_img(d)

                Numbers = allimgs // 5
                self.split_imag_data(Numbers, d, td)
            print("\t\tFinish creating test set")
            Verify_Dir = self.root + name + "/verify/"
            for i in range(5):
                d = Train_Dir + str(i) + "/"
                td = Verify_Dir + str(i) + "/"
                allimgs = self.count_img(d)

                Numbers = allimgs // 5
                self.split_imag_data(Numbers, d, td)
            print("Finish creating verify set")
        elif flag == 2:
            '''划分为：train vs test'''
            for i in range(5):
                d = Train_Dir + str(i) + "/"
                td = Test_Dir + str(i) + "/"
                allimgs = self.count_img(d)
                print("总图片数量：", allimgs)
                Numbers = allimgs // 5
                self.split_imag_data(Numbers, d, td)
            print("Finish creating test set")
        else:
            print("{}划分完毕！".format(Train_Dir))
        print("*" * 50)


class Show_Info:
    def __init__(self):
        pass

    def data_per_category(self,df,s):
        '''画的是饼图'''
        import matplotlib.pyplot as plt  # 导入绘图包
        a = df.Label.value_counts() / s * 100
        a.to_numpy()
        print(a)
        plt.figure(figsize=(8, 8))
        plt.pie(a, labels=['R', 'RPM', 'gear', 'DoS', 'Fuzzy'],
                autopct='%3.1f%%',
                colors=["#8CB9C0", "#D98481", "#7892B5", "#91B5A9", "#EDCA7F"])
        plt.title('The amount of data per category')  # 加标题
        plt.show()

    def getfileName(self, directory_path):
        files = os.listdir(directory_path)
        random_number = random.randint(1, 800)
        return files[random_number]

    def class_figure(self, root):
        '''画出一行5个图，每个图是来自这个分类'''
        img = []
        for i in range(5):
            filedir = '{}{}'.format(root, "train/{}/".format(i))
            img.append(Image.open('{}{}'.format(filedir, self.getfileName(filedir))))

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 5, 1)
        plt.imshow(img[0])
        plt.title("Normal")
        plt.subplot(1, 5, 2)
        plt.imshow(img[1])
        plt.title("RPM Spoofing")
        plt.subplot(1, 5, 3)
        plt.imshow(img[2])
        plt.title("Gear Spoofing")
        plt.subplot(1, 5, 4)
        plt.imshow(img[3])
        plt.title("DoS Attack")
        plt.subplot(1, 5, 5)
        plt.imshow(img[4])
        plt.title("Fuzzy Attack")
        plt.show()

    def data_distribution(self, df, title):
        '''给的是一个csv，查看每一类的分布，画的是柱状图'''
        plt.figure(figsize=(15, 5))
        for i in range(9):
            plt.hist(df.iloc[:, i], bins=15, alpha=0.8, label=str(df.columns.tolist()[i]))
        plt.xlabel("Numerical value")
        plt.ylabel("Numbers")
        plt.tick_params(top=False, right=False)
        plt.title(title)
        plt.legend()
        plt.show()

    def count_img(self, root, name):
        # 指定目录路径
        s = []
        a = 0
        for i in range(5):
            directory_path = root + name + '/' + str(i) + '/'
            files = os.listdir(directory_path)
            s.append("类型{}：{}个".format(i, len(files)))
            a += len(files)
        return {name: [a, s]}

    def plot_losses(self, df, i):
        '''画的是折线图'''
        # 创建一个 figure 和一个 ax 对象
        fig, ax1 = plt.subplots()
        # 绘制 train_loss，使用 ax1 和蓝色
        ax1.plot(df.index, df['train_loss'], label='Train Loss', color='#CD0056')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Train Loss')
        ax1.tick_params(axis='y')

        # 创建一个共享相同 x 轴的第二个 ax 对象
        ax2 = ax1.twinx()
        ax2.plot(df.index, df['val_loss'], label='Validation Loss', color='#0C755F', alpha=0.8)
        ax2.set_ylabel('Validation Loss')
        ax2.tick_params(axis='y')
        plt.grid(True)
        # 添加图例
        fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        # 添加标题
        plt.title('Car:{} Training and Validation Losses'.format(i))

        # 显示图像
        plt.show()

    def eval_model(self, m, Val, device, title=""):
        '''评价分类模型的效果，画的是混淆矩阵'''
        m.eval()
        pred = []
        y = []

        for (seq, target) in tqdm(Val):
            with torch.no_grad():
                seq = seq.to(device)
                y_pred = m(seq)
                y_pred = y_pred.argmax(1, keepdim=True)
                target = target.long()
                pred += y_pred.data.tolist()  # 将预测值添加到列表中
                y += target.data.tolist()  # 将真实值添加到列表中

        pred = np.array(pred)
        y = np.array(y)

        acc = accuracy_score(y, pred)
        precision = precision_score(y, pred, labels=None, pos_label=1, zero_division=1, average='macro')
        recall = recall_score(y, pred, average='macro',zero_division="warn")  # 'micro', 'macro', 'weighted'
        f1 = f1_score(y, pred, average='macro')
        print('\t准确率-Acc:{}\n\t查准率-TP/(TP+FP):{}\n\t召回率-TP/(TP+FN):{}\n\tF1:{}'.format(acc, precision, recall, f1))

        _fig_title = title
        from matplotlib import pyplot as plt
        from sklearn.metrics import confusion_matrix
        plt.figure(figsize=(15, 15))
        label_names = ['R', 'RPM', 'gear', 'DoS', 'Fuzzy']
        confusion = confusion_matrix(y, pred, labels=[i for i in range(len(label_names))])
        plt.matshow(confusion, cmap=plt.cm.Greens)  # Greens, Blues, Oranges, Reds
        plt.colorbar()
        for i in range(len(confusion)):
            for j in range(len(confusion)):
                plt.annotate(confusion[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.xticks(range(len(label_names)), label_names)
        plt.yticks(range(len(label_names)), label_names)
        plt.title("{}".format(_fig_title))
        plt.show()


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


def deliver_data_to_car(root, num_splits):
    '''
    将拥有的can数据发放给50辆车
    '''
    df = pd.read_csv("/home/raoxy/data/Car_Hacking_100.csv").sample(frac=1).reset_index(drop=True)
    overlap = 2
    split_size = ceil(len(df) / num_splits)
    split_dfs = []

    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = start_idx + split_size
        if i > 0:
            start_idx = max(0, start_idx - overlap)
        end_idx = min(len(df), end_idx)

        split_dfs.append(df.iloc[start_idx:end_idx])
        directory = root + 'car_{}/'.format(i + 1)
        if not os.path.exists(directory):
            os.makedirs(directory)
        split_dfs[i].to_csv('{}car_{}.csv'.format(directory, i + 1), index=False)
        print("{}car_{}.csv 数据分发完毕".format(directory, i + 1))


if __name__ == '__main__':
    '''
    这里将数据集划分为20份了，但是只处理了12份
    carhacking_raw：没有任何预处理的原始图像
    carhacking_our：欠采样+yeo处理
    '''
    # root = '/home/raoxy/data/carhacking_raw/'
    # root = '/home/raoxy/data/carhacking_our/'
    root = '/home/raoxy/data/carhacking_our_bt/'
    deliver_data_to_car(root, 15)  # 下发给区域内的12辆车

    for i in range(10, 11):
        load_can_data("car_" + str(i), root, f=2)

    # for i in range(12, 13):
    #     load_can_data("car_" + str(i), root, f=1)
