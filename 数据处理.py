# 读取数据集

这个数据集来源于：https://docs.google.com/forms/d/e/1FAIpQLScLxUSEoEbmM2w8UhN3D388TzQMqCLCAvxdR2hu8O-YoZbgIQ/formResponse?pli=1


| 攻击类型 | 总数 | 正常数量 | 恶意数量 |
| :---------- | :------------ | :------------------- | :--------------------- |
| DoS Attack                  | 3,665,771 | 3,078,250 | 587,521 |
| Fuzzy Attack                | 3,838,860 | 3,347,013 | 491,847 |
| Spoofing the drive gear     | 4,443,142 | 3,845,890 | 597,252 |
| Spoofing the RPM gauze      | 4,621,702 | 3,966,805 | 654,897 |
| GIDS: Attack-free  (normal) | 988,987   | 988,872   |         |

如果已经处理好了一份`Car_Hacking_100%.csv`数据集,可以直接跳到`2 数据探索部分`加载这个csv,`3 划分数据集`部分提供了5中不同的数据集处理方案

#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

## 干净数据

import pandas as pd

# 定义列名
column_names = ['Timestamp', 'CAN ID', 'DLC', 'DATA[0]', 'DATA[1]', 'DATA[2]', 'DATA[3]', 'DATA[4]', 'DATA[5]', 'DATA[6]', 'DATA[7]', 'Flag']
csv_files = ['~/data/DoS_dataset.csv', '~/data/RPM_dataset.csv','~/data/gear_dataset.csv','~/data/Fuzzy_dataset.csv']

all_data = pd.DataFrame()

for file in csv_files:
    df = pd.read_csv(file,names=column_names)
    all_data = pd.concat([all_data, df])

all_data=all_data.loc[all_data['Flag'] == 'R']
all_data

all_data=all_data.drop_duplicates() # 去掉重复值 
all_data

all_data.loc[:, ['CAN ID','DATA[0]', 'DATA[1]', 'DATA[2]', 'DATA[3]', 'DATA[4]', 'DATA[5]', 'DATA[6]', 'DATA[7]']].max()

all_data.loc[:, ['CAN ID','DATA[0]', 'DATA[1]', 'DATA[2]', 'DATA[3]', 'DATA[4]', 'DATA[5]', 'DATA[6]', 'DATA[7]']].min()

all_data=all_data.loc[:, ['CAN ID','DATA[0]', 'DATA[1]', 'DATA[2]', 'DATA[3]', 'DATA[4]', 'DATA[5]', 'DATA[6]', 'DATA[7]','Flag']]
new_header = ['CAN ID', 'DATA[0]', 'DATA[1]', 'DATA[2]', 'DATA[3]', 'DATA[4]', 'DATA[5]', 'DATA[6]', 'DATA[7]', 'Label']
all_data.columns = new_header

columns_to_convert = ['CAN ID','DATA[0]', 'DATA[1]', 'DATA[2]', 'DATA[3]', 'DATA[4]', 'DATA[5]', 'DATA[6]', 'DATA[7]']
all_data[columns_to_convert]=all_data[columns_to_convert].applymap(lambda x: int(x, 16) if isinstance(x, str) else int(str(x),16))

all_data

all_data.to_csv('~/data/normal_dataset.csv',index=False)

## 4种攻击

import pandas as pd

def read_raw_data(tag_name,data_name,new_data_name):
    column_names = ['Timestamp', 'CAN ID', 'DLC', 'DATA[0]', 'DATA[1]', 'DATA[2]', 'DATA[3]', 'DATA[4]', 'DATA[5]', 'DATA[6]', 'DATA[7]', 'Flag']
    df=pd.read_csv(data_name, names=column_names)
    print(df.Flag.value_counts())
    print('min:\n{}\t max:\n{}'.format(df.loc[df['Flag'] == 'T'].loc[:, ['CAN ID','DATA[0]', 'DATA[1]', 'DATA[2]', 'DATA[3]', 'DATA[4]', 'DATA[5]', 'DATA[6]', 'DATA[7]']].min()
    ,df.loc[df['Flag'] == 'T'].loc[:, ['CAN ID','DATA[0]', 'DATA[1]', 'DATA[2]', 'DATA[3]', 'DATA[4]', 'DATA[5]', 'DATA[6]', 'DATA[7]']].max()))
    df_evil=df.loc[df['Flag'] == 'T'].loc[:, ['CAN ID','DATA[0]', 'DATA[1]', 'DATA[2]', 'DATA[3]', 'DATA[4]', 'DATA[5]', 'DATA[6]', 'DATA[7]','Flag']]

    df_evil.loc[:, 'Flag'] = tag_name
    new_header = ['CAN ID', 'DATA[0]', 'DATA[1]', 'DATA[2]', 'DATA[3]', 'DATA[4]', 'DATA[5]', 'DATA[6]', 'DATA[7]', 'Label']
    df_evil.columns = new_header

    columns_to_convert = ['CAN ID','DATA[0]', 'DATA[1]', 'DATA[2]', 'DATA[3]', 'DATA[4]', 'DATA[5]', 'DATA[6]', 'DATA[7]']
    df_evil[columns_to_convert]=df_evil[columns_to_convert].applymap(lambda x: int(x, 16) if isinstance(x, str) else int(str(x),16))
    df_evil.to_csv(new_data_name,index=False)
    print(df_evil)

read_raw_data('DoS','~/data/DoS_dataset.csv','~/data/DoS_dataset_attack.csv')

read_raw_data('RPM','~/data/RPM_dataset.csv','~/data/RPM_dataset_attack.csv')

read_raw_data('gear','~/data/gear_dataset.csv','~/data/gear_dataset_attack.csv')

read_raw_data('Fuzzy','~/data/Fuzzy_dataset.csv','~/data/Fuzzy_dataset_attack.csv')

## 合并所有的数据

import pandas as pd

csv_files = ['DoS_dataset_attack.csv', 'Fuzzy_dataset_attack.csv', 'gear_dataset_attack.csv','RPM_dataset_attack.csv','normal_dataset.csv']
all_data = pd.DataFrame()

for file in csv_files:
    df = pd.read_csv('~/data/'+file)
    all_data = pd.concat([all_data, df])

all_data

all_data.to_csv('~/data/Car_Hacking_100%.csv',index=False)
# 现在，all_data包含了所有文件的数据

# 数据探索

## 查看每一类的数据量

原来的数据量是818,440条，现在扩充后是13,724,466条

import pandas as pd

df = pd.read_csv('~/data/Car_Hacking_100%.csv')
df

df.Label.value_counts()/13724466*100

import matplotlib.pyplot as plt                #导入绘图包

a=df.Label.value_counts()/13724466*100
a.to_numpy()
plt.figure(figsize=(8,8))
plt.pie(a,labels=['R','RPM','gear','DoS','Fuzzy'], autopct='%3.1f%%')  #以时间为标签，总计成交笔数为数据绘制饼图，并显示3位整数一位小数
plt.title('The amount of data per category')             #加标题
plt.show()

## 原始特征直方图

蓝色的是car id `CAN ID : identifier of CAN message in HEX (ex. 043f)` 感觉没有什么用

# from PIL import Image, ImageStat
import matplotlib.pyplot as plt

plt.figure(figsize=(15,5))

for i in range(1,9):
    plt.hist(df.iloc[:,i], bins=15,alpha = 0.5,label=str(df.columns.tolist()[i]))
plt.xlabel("Numerical value")
plt.ylabel("Numbers")
plt.tick_params(top=False, right=False)
plt.title('primitive')
plt.legend()
plt.show()

## QQ曲线

import statsmodels.api as sm
import matplotlib.pyplot as plt
for i in range(8):
    sm.qqplot(df.iloc[:,i], line='s')

# 划分数据集

##  QuantileTransformer转换

def showPicture(name_):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,5))
    for i in range(9):
        plt.hist(yeo_df[numeric_features].iloc[:,i], bins=15,alpha = 0.5,label=str(yeo_df.columns.tolist()[i]))
    plt.xlabel("Numerical value")
    plt.ylabel("Numbers")
    plt.tick_params(top=False, right=False)
    plt.title(name_)
    plt.legend()
    plt.show()

### 标准化

qt_df=pd.read_csv('~/data/Car_Hacking_100%.csv')
qt_df=qt_df.sample(frac=0.02)
qt_df

from sklearn.preprocessing import QuantileTransformer
scaler = QuantileTransformer()
qt_df[numeric_features] = scaler.fit_transform(qt_df[numeric_features])

qt_df[numeric_features].describe()

 showPicture('QuantileTransformer')

### 降采样

X=qt_df[numeric_features]
y=qt_df['Label']

from imblearn.under_sampling import NearMiss
nm_1 = NearMiss() #形参默认 version=1, 即采用 NearMiss-1
X_resampled, y_resampled = nm_1.fit_resample(X, y)

qt_df=pd.merge(X_resampled,y_resampled,left_index=True,right_index=True)
print(qt_df.Label.value_counts())

 showPicture('QuantileTransformer-nearmiss')

### 归一化

qt_df[numeric_features] = qt_df[numeric_features].apply(lambda x: (x*255))

 showPicture('QuantileTransformer-nearmiss-scale')

### 分类

df0=qt_df[qt_df['Label']=='R'].drop(['Label'],axis=1)
df1=qt_df[qt_df['Label']=='RPM'].drop(['Label'],axis=1)
df2=qt_df[qt_df['Label']=='gear'].drop(['Label'],axis=1)
df3=qt_df[qt_df['Label']=='DoS'].drop(['Label'],axis=1)
df4=qt_df[qt_df['Label']=='Fuzzy'].drop(['Label'],axis=1)

def generate_image(DF,ditrr, image_p):
    import os
    count=0
    ims = []
    image_path = ditrr + image_p
    os.makedirs(image_path)

    for i in range(0, len(DF)):
        count=count+1
        if count<=27:
            im=DF.iloc[i].values
            ims=np.append(ims,im)
        else:
            ims=np.array(ims).reshape(9,9,3)
            array = np.array(ims, dtype=np.uint8)
            new_image = Image.fromarray(array)
            new_image.save(image_path+str(i)+'.png')
            count=0
            ims = []

generate_image(df0,ditrr='/home/raoxy/data/train_qt/',image_p = "0/") # normal021
generate_image(df1,ditrr='/home/raoxy/data/train_qt/',image_p = "1/")
generate_image(df2,ditrr='/home/raoxy/data/train_qt/',image_p = "2/")
generate_image(df3,ditrr='/home/raoxy/data/train_qt/',image_p = "3/") # dos attack
generate_image(df4,ditrr='/home/raoxy/data/train_qt/',image_p = "4/") # fuzzy attack

import os
import matplotlib.pyplot as plt
from PIL import Image
# Read the images for each category, the file name may vary (27.png, 83.png...)
img1 = Image.open('/home/raoxy/data/train_qt/0/27.png')
img2 = Image.open('/home/raoxy/data/train_qt/1/83.png')
img3 = Image.open('/home/raoxy/data/train_qt/2/27.png')
img4 = Image.open('/home/raoxy/data/train_qt/3/27.png')
img5 = Image.open('/home/raoxy/data/train_qt/4/27.png')

plt.figure(figsize=(10, 10)) 
plt.subplot(1,5,1)
plt.imshow(img1)
plt.title("Normal")
plt.subplot(1,5,2)
plt.imshow(img2)
plt.title("RPM Spoofing")
plt.subplot(1,5,3)
plt.imshow(img3)
plt.title("Gear Spoofing")
plt.subplot(1,5,4)
plt.imshow(img4)
plt.title("DoS Attack")
plt.subplot(1,5,5)
plt.imshow(img5)
plt.title("Fuzzy Attack")
plt.show()  

## yeo-johnson转换

import pandas as pd
import numpy as np

yeo_df=pd.read_csv('~/data/Car_Hacking_100%.csv')
np.random.shuffle(yeo_df.values)

# 保存到新的csv文件中
yeo_df.to_csv('~/data/Car_Hacking_100%.csv', index=False)
yeo_df

### 标准化

from sklearn.preprocessing import PowerTransformer
numeric_features=['CAN ID', 'DATA[0]', 'DATA[1]', 'DATA[2]', 'DATA[3]', 'DATA[4]', 'DATA[5]', 'DATA[6]', 'DATA[7]']
scaler = PowerTransformer(method='yeo-johnson')
yeo_df[numeric_features]=scaler.fit_transform(yeo_df[numeric_features])

yeo_df[numeric_features].describe()

def showPicture(name_):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,5))
    for i in range(9):
        plt.hist(yeo_df[numeric_features].iloc[:,i], bins=15,alpha = 0.5,label=str(yeo_df.columns.tolist()[i]))
    plt.xlabel("Numerical value")
    plt.ylabel("Numbers")
    plt.tick_params(top=False, right=False)
    plt.title(name_)
    plt.legend()
    plt.show()

 showPicture('yeo-johnson')

### 降采样

X=yeo_df[numeric_features]
y=yeo_df['Label']

X

from imblearn.under_sampling import NearMiss
nm_1 = NearMiss() #形参默认 version=1, 即采用 NearMiss-1
X_resampled, y_resampled = nm_1.fit_resample(X, y)

yeo_df=pd.merge(X_resampled,y_resampled,left_index=True,right_index=True)
print(yeo_df.Label.value_counts())

 showPicture('yeo-johnson-nearmiss')

### 归一化

from sklearn.preprocessing import minmax_scale
yeo_df[numeric_features]=minmax_scale(yeo_df[numeric_features])
yeo_df[numeric_features] = yeo_df[numeric_features].apply(lambda x: (x*255))

 showPicture('yeo-johnson-nearmiss-maxmin')

### 分类

def data_classification(df):
    df0=df[df['Label']=='R'].drop(['Label'],axis=1)
    df1=df[df['Label']=='RPM'].drop(['Label'],axis=1)
    df2=df[df['Label']=='gear'].drop(['Label'],axis=1)
    df3=df[df['Label']=='DoS'].drop(['Label'],axis=1)
    df4=df[df['Label']=='Fuzzy'].drop(['Label'],axis=1)
    return df0,df1,df2,df3,df4

df0,df1,df2,df3,df4=data_classification(yeo_df.copy(deep=True))

def generate_image(DF,ditrr='/home/raoxy/data/train_yeo/', image_p = ""):
    import os
    from PIL import Image
    import warnings

    count=0
    ims = []
    image_path = ditrr + image_p
    os.makedirs(image_path,exist_ok=True)

    for i in range(0, len(DF)):
        count=count+1
        if count<=27:
            im=DF.iloc[i].values
            ims=np.append(ims,im)
        else:
            ims=np.array(ims).reshape(9,9,3)
            array = np.array(ims, dtype=np.uint8)
            new_image = Image.fromarray(array)
            new_image.save(image_path+str(i)+'.png')
            count=0
            ims = []

generate_image(df0,image_p = "0/") # normal
generate_image(df1,image_p = "1/")
generate_image(df2,image_p = "2/")
generate_image(df3,image_p = "3/") # dos attack
generate_image(df4,image_p = "4/") # fuzzy attack

import os
import matplotlib.pyplot as plt
from PIL import Image
# Read the images for each category, the file name may vary (27.png, 83.png...)
img1 = Image.open('/home/raoxy/data/train_yeo/0/27.png')
img2 = Image.open('/home/raoxy/data/train_yeo/1/83.png')
img3 = Image.open('/home/raoxy/data/train_yeo/2/27.png')
img4 = Image.open('/home/raoxy/data/train_yeo/3/27.png')
img5 = Image.open('/home/raoxy/data/train_yeo/4/27.png')

plt.figure(figsize=(10, 10)) 
plt.subplot(1,5,1)
plt.imshow(img1)
plt.title("Normal")
plt.subplot(1,5,2)
plt.imshow(img2)
plt.title("RPM Spoofing")
plt.subplot(1,5,3)
plt.imshow(img3)
plt.title("Gear Spoofing")
plt.subplot(1,5,4)
plt.imshow(img4)
plt.title("DoS Attack")
plt.subplot(1,5,5)
plt.imshow(img5)
plt.title("Fuzzy Attack")
plt.show()  

### 数据集划分

# import os
# import cv2
# import random
# import shutil
# import warnings
import os
# import cv2
import random
import shutil
import warnings
warnings.filterwarnings("ignore")

Train_Dir='/home/raoxy/data/train_yeo/'

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
Numbers_ = len_all // 5  # size of verify set (20%)

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

Verify_Dir = '/home/raoxy/data/verify_yeo/'

allimgs = count_img(Train_Dir) # 更新一下现在的训练集
split_imag_data(allimgs, Numbers_, Train_Dir, Verify_Dir)
print('Finish creating verify set')

### 图像放大

def get_224(folder, dstdir):
    import cv2
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


DATA_DIR_224 = '/home/raoxy/data/train_yeo_224/'
DATA_DIR2_224 ='/home/raoxy/data/test_yeo_224/'
DATA_DIR3_224 ='/home/raoxy/data/verify_yeo_224/'

get_224(folder=Train_Dir, dstdir=DATA_DIR_224)
get_224(folder=Test_Dir, dstdir=DATA_DIR2_224)
get_224(folder=Verify_Dir, dstdir=DATA_DIR3_224)

num1 = len(count_img(DATA_DIR_224))
num2 = len(count_img(DATA_DIR2_224))
num3 = len(count_img(DATA_DIR3_224))

print('训练集中的图片数量：{} ({})'.format(num1, num1 / (num1+num2+num3)))
print('测试集中的图片数量：{} ({})'.format(num2, num2 / (num1+num2+num3)))
print('验证集中的图片数量：{} ({})'.format(num3, num3 / (num1+num2+num3)))

%%bash

conda install opencv-python

## QuantileTransformer-正则转换

def showPicture(name_):
    from PIL import Image, ImageStat
    import math
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2

    plt.figure(figsize=(15,5))
    for i in range(9):
        plt.hist(qtn_df[numeric_features].iloc[:,i], bins=15,alpha = 0.5,label=str(qtn_df.columns.tolist()[i]))
    plt.xlabel("Numerical value")
    plt.ylabel("Numbers")
    plt.tick_params(top='off', right='off')
    plt.title(name_)
    plt.legend()
    plt.show()

### 标准化

qtn_df=DF_R.copy(deep=True)

from sklearn.preprocessing import QuantileTransformer
scaler = QuantileTransformer(output_distribution='normal')
qtn_df[numeric_features] = scaler.fit_transform(qtn_df[numeric_features])

qtn_df[numeric_features].describe()

showPicture('QuantileTransformer-normal')

### 降采样

X=qtn_df[numeric_features]
y=qtn_df['Label']

from imblearn.under_sampling import NearMiss
nm_1 = NearMiss() #形参默认 version=1, 即采用 NearMiss-1
X_resampled, y_resampled = nm_1.fit_resample(X, y)

qtn_df=pd.merge(X_resampled,y_resampled,left_index=True,right_index=True)
print(qtn_df.Label.value_counts())

showPicture('QuantileTransformer-normal-nearmiss')

### 归一化

from sklearn.preprocessing import minmax_scale
qtn_df[numeric_features]=minmax_scale(qtn_df[numeric_features])
qtn_df[numeric_features] = qtn_df[numeric_features].apply(lambda x: (x*255))

showPicture('QuantileTransformer-normal-nearmiss-maxmin')

### 分类

df=qtn_df.copy(deep=True)

df0=df[df['Label']=='R'].drop(['Label'],axis=1)
df1=df[df['Label']=='RPM'].drop(['Label'],axis=1)
df2=df[df['Label']=='gear'].drop(['Label'],axis=1)
df3=df[df['Label']=='DoS'].drop(['Label'],axis=1)
df4=df[df['Label']=='Fuzzy'].drop(['Label'],axis=1)

def generate_image(DF,ditrr='../data/train_qnt/', image_p = ""):
    ''':cvar
    Generate 9*9 color images for class 0 (Normal)
    Change the numbers 9 to the number of features n in your dataset if you use a different dataset, reshape(n,n,3)
    '''
    count=0
    ims = []
    image_path = ditrr + image_p
    os.makedirs(image_path)

    for i in range(0, len(DF)):
        count=count+1
        if count<=27:
            im=DF.iloc[i].values
            ims=np.append(ims,im)
        else:
            ims=np.array(ims).reshape(9,9,3)
            array = np.array(ims, dtype=np.uint8)
            new_image = Image.fromarray(array)
            new_image.save(image_path+str(i)+'.png')
            count=0
            ims = []

generate_image(df0,image_p = "0/") # normal
generate_image(df1,image_p = "1/")
generate_image(df2,image_p = "2/")
generate_image(df3,image_p = "3/") # dos attack
generate_image(df4,image_p = "4/") # fuzzy attack

display_image_for_each_category(drit='../data/train_qnt',one='27.png',two='83.png',three='27.png',four='27.png',five='27.png')

# from imblearn.over_sampling import SMOTE
# X_resampled, y_resampled = SMOTE().fit_resample(X, y)

# from imblearn.under_sampling import RandomUnderSampler
# rus = RandomUnderSampler(random_state=0)
# X_resampled, y_resampled = rus.fit_resample(X, y)
