#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import os
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

Train_Dir = '../data2/train_quantile_ransformer/'
Test_Dir = '../data2/test_quantile_ransformer/'
# Verify_Dir = '../data/verify_quantile_ransformer/'

# 读取数据集
DF_R = pd.read_csv('../Car_Hacking_5%.csv')
# 2.将数据转换为图片
numeric_features = DF_R.dtypes[DF_R.dtypes != 'object'].index
qtn_df = DF_R.copy(deep=True)

# ## 标准化
from sklearn.preprocessing import QuantileTransformer

scaler = QuantileTransformer(output_distribution='normal')
qtn_df[numeric_features] = scaler.fit_transform(qtn_df[numeric_features])

# ## 降采样
X = qtn_df[numeric_features]
y = qtn_df['Label']

from imblearn.under_sampling import NearMiss

nm_1 = NearMiss()  # 形参默认 version=1, 即采用 NearMiss-1
X_resampled, y_resampled = nm_1.fit_resample(X, y)

qtn_df = pd.merge(X_resampled, y_resampled, left_index=True, right_index=True)
print(qtn_df.Label.value_counts())

# ## 归一化
from sklearn.preprocessing import minmax_scale

qtn_df[numeric_features] = minmax_scale(qtn_df[numeric_features])
qtn_df[numeric_features] = qtn_df[numeric_features].apply(lambda x: (x * 255))

# ## 分类
df = qtn_df.copy(deep=True)
df0 = df[df['Label'] == 'R'].drop(['Label'], axis=1)
df1 = df[df['Label'] == 'RPM'].drop(['Label'], axis=1)
df2 = df[df['Label'] == 'gear'].drop(['Label'], axis=1)
df3 = df[df['Label'] == 'DoS'].drop(['Label'], axis=1)
df4 = df[df['Label'] == 'Fuzzy'].drop(['Label'], axis=1)


def generate_image(DF, ditrr=Train_Dir, image_p=""):
    ''':cvar
    Generate 9*9 color images for class 0 (Normal)
    Change the numbers 9 to the number of features n in your dataset if you use a different dataset, reshape(n,n,3)
    '''
    count = 0
    ims = []
    image_path = ditrr + image_p
    os.makedirs(image_path)

    for i in range(0, len(DF)):
        count = count + 1
        if count <= 27:
            im = DF.iloc[i].values
            ims = np.append(ims, im)
        else:
            ims = np.array(ims).reshape(9, 9, 3)
            array = np.array(ims, dtype=np.uint8)
            new_image = Image.fromarray(array)
            new_image.save(image_path + str(i) + '.png')
            count = 0
            ims = []


generate_image(df0, image_p="0/")  # normal
generate_image(df1, image_p="1/")
generate_image(df2, image_p="2/")
generate_image(df3, image_p="3/")  # dos attack
generate_image(df4, image_p="4/")  # fuzzy attack

import os
import cv2
import random
import shutil
import warnings


def count_img(Train_Dir):
    allimgs = []
    for subdir in os.listdir(Train_Dir):
        for filename in os.listdir(os.path.join(Train_Dir, subdir)):
            filepath = os.path.join(Train_Dir, subdir, filename)
            allimgs.append(filepath)
    return allimgs


allimgs = count_img(Train_Dir)

print('总图片数量：', len(allimgs))

len_all = len(allimgs)
Numbers = len_all // 5  # size of test set (20%)
# Numbers_ = len_all // 5  # size of verify set (20%)

def mymovefile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.move(srcfile, dstfile)


def split_imag_data(allimgs, Numbers, Train_Dir, Val_Dir):
    val_imgs = random.sample(allimgs, Numbers)
    for img in val_imgs:
        dest_path = img.replace(Train_Dir, Val_Dir)
        mymovefile(img, dest_path)


split_imag_data(allimgs, Numbers, Train_Dir, Test_Dir)
print('Finish creating test set')
# allimgs = count_img(Train_Dir)  # 更新一下现在的训练集
# split_imag_data(allimgs, Numbers_, Train_Dir, Verify_Dir)
# print('Finish creating verify set')


def get_224(folder, dstdir):
    imgfilepaths = []
    for root, dirs, imgs in os.walk(folder):
        for thisimg in imgs:
            thisimg_path = os.path.join(root, thisimg)
            imgfilepaths.append(thisimg_path)
    for thisimg_path in imgfilepaths:
        dir_name, filename = os.path.split(thisimg_path)
        dir_name = dir_name.replace(folder, dstdir)
        new_file_path = os.path.join(dir_name, filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        img = cv2.imread(thisimg_path)
        img = cv2.resize(img, (224, 224))
        cv2.imwrite(new_file_path, img)
    print('Finish resizing'.format(folder=folder))


DATA_DIR_224 = '../data2/train_quantile_ransformer_224/'
DATA_DIR2_224 = '../data2/test_quantile_ransformer_224/'
# DATA_DIR3_224 = '../data2/verify_quantile_ransformer_224/'

get_224(folder=Train_Dir, dstdir=DATA_DIR_224)
get_224(folder=Test_Dir, dstdir=DATA_DIR2_224)
# get_224(folder=Verify_Dir, dstdir=DATA_DIR3_224)

num1 = len(count_img(DATA_DIR_224))
num2 = len(count_img(DATA_DIR2_224))
# num3 = len(count_img(DATA_DIR3_224))

# print('训练集中的图片数量：{} ({})'.format(num1, num1 / (num1+num2+num3)))
# print('测试集中的图片数量：{} ({})'.format(num2, num2 / (num1+num2+num3)))
# print('验证集中的图片数量：{} ({})'.format(num3, num3 / (num1+num2+num3)))
print('训练集中的图片数量：{} ({})'.format(num1, num1 / (num1+num2)))
print('测试集中的图片数量：{} ({})'.format(num2, num2 / (num1+num2)))