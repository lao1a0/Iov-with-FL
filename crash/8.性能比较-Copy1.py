#!/usr/bin/env python
# coding: utf-8

# In[25]:


'''
    1.单幅图片验证
    2.多幅图片验证
'''
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet34
from utils import LoadData, write_result
import pandas as pd


# In[23]:


def eval(dataloader, model):
    label_list = []
    likelihood_list = []
    pred_list = []
    model.eval()
    with torch.no_grad():
        # 加载数据加载器，得到里面的X（图片数据）和y(真实标签）
        for idx, (X, y) in enumerate(dataloader):
            # 将数据转到GPU
            X = X.cuda()
            # 将图片传入到模型当中就，得到预测的值pred
            pred = model(X)
            pred_softmax = torch.softmax(pred, 1).cpu().numpy()
            # 获取可能性最大的标签
            label = torch.softmax(pred, 1).cpu().numpy().argmax()
            label_list.append(label)
            # 获取可能性最大的值（即概率）
            likelihood = torch.softmax(pred, 1).cpu().numpy().max()
            likelihood_list.append(likelihood)
            pred_list.append(pred_softmax.tolist()[0])

        return label_list, likelihood_list, pred_list


# In[26]:


'''
    加载预训练模型
'''
# 1. 导入模型结构
# model = resnet34(pretrained=False)
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
    
model=CNN(5)
# num_ftrs = model.fc.in_features    # 获取全连接层的输入
# model.fc = nn.Linear(num_ftrs, 5)  # 全连接层改为不同的输出
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 加载模型参数
model = CNN(5)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model_loc = r'D:\Desktop\shi_yan2\shi_yan_server_2023-3-27\model\yeo_cnn_lr0.01_epochs50.pt'
model_dict = torch.load(model_loc)
model.load_state_dict(model_dict)
model = model.to(device)


# In[27]:


'''
   加载需要预测的图片
'''
valid_data = LoadData( r"D:\Desktop\shi_yan2\shi_yan_server_2023-3-27\files\true_label_cnn_yeo.txt"  , train_flag=False)
test_dataloader = DataLoader(dataset=valid_data, num_workers=4, pin_memory=True, batch_size=1)


'''
  获取结果
'''
# 获取模型输出
label_list, likelihood_list, pred =  eval(test_dataloader, model)

# 将输出保存到exel中，方便后续分析
# label_names = ["daisy", "dandelion","rose","sunflower","tulip"]     # 可以把标签写在这里
label_names = ['R', 'RPM', 'gear', 'DoS', 'Fuzzy']
df_pred = pd.DataFrame(data=pred, columns=label_names)
df_pred


# In[15]:


df_pred.to_csv('pred_result.csv', encoding='gbk', index=False)
print("Done!")


# #  模型性能度量

# In[13]:


import matplotlib.pyplot as plt # pip install matplotlib
import numpy as np  # pip install numpy
from numpy import interp
from sklearn.preprocessing import label_binarize
import pandas as pd # pip install pandas

'''
读取数据

需要读取模型输出的标签（predict_label）以及原本的标签（true_label）

'''
# target_loc = "test.txt"     # 真实标签所在的文件
target_loc = r"D:\Desktop\shi_yan2\shi_yan_server_2023-3-27\files\true_label_cnn_yeo.txt"     # 真实标签所在的文件
target_data = pd.read_csv(target_loc, sep="\t", names=["loc","type"])
true_label = [i for i in target_data["type"]]

print(true_label)


# In[14]:


import pandas as pd
from sklearn.metrics import *  # pip install scikit-learn
# predict_loc = "pred_result.csv"     # 3.ModelEvaluate.py生成的文件
predict_loc=r'D:\Desktop\shi_yan2\shi_yan_server_2023-3-27\files\vgg16_yeo_lr0.01_epochs50_verify.csv'
predict_data = pd.read_csv(predict_loc)#,index_col=0)
predict_data


# In[15]:


predict_label = predict_data.to_numpy().argmax(axis=1)
predict_score = predict_data.to_numpy().max(axis=1)


# In[16]:


predict_label


# In[17]:


predict_score


# In[18]:


import torch, gc
gc.collect()
torch.cuda.empty_cache()


# ##  精度，查准率，召回率，F1-Score

# In[19]:


# 精度，准确率， 预测正确的占所有样本种的比例
accuracy = accuracy_score(true_label, predict_label)
print("精度: ",accuracy)


# In[19]:


# 查准率P（准确率），precision(查准率)=TP/(TP+FP)

precision = precision_score(true_label, predict_label, labels=None, pos_label=1, average='macro') # 'micro', 'macro', 'weighted'
print("查准率P: ",precision)

# 查全率R（召回率），原本为对的，预测正确的比例；recall(查全率)=TP/(TP+FN)
recall = recall_score(true_label, predict_label, average='macro') # 'micro', 'macro', 'weighted'
print("召回率: ",recall)

# F1-Score
f1 = f1_score(true_label, predict_label, average='macro')     # 'micro', 'macro', 'weighted'
print("F1 Score: ",f1)


# ## 混淆矩阵

# In[ ]:


label_names = ["daisy", "dandelion","rose","sunflower","tulip"]
confusion = confusion_matrix(true_label, predict_label, labels=[i for i in range(len(label_names))])


plt.matshow(confusion, cmap=plt.cm.Oranges)   # Greens, Blues, Oranges, Reds
plt.colorbar()


# In[20]:


for i in range(len(confusion)):
    for j in range(len(confusion)):
        plt.annotate(confusion[j,i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.xticks(range(len(label_names)), label_names)
plt.yticks(range(len(label_names)), label_names)
plt.title("Confusion Matrix")
plt.show()


# ## ROC曲线（多分类）
# 
# 在多分类的ROC曲线中，会把目标类别看作是正例，而非目标类别的其他所有类别看作是负例，从而造成负例数量过多，
# 虽然模型准确率低，但由于在ROC曲线中拥有过多的TN，因此AUC比想象中要大

# In[21]:


n_classes = len(label_names)
# binarize_predict = label_binarize(predict_label, classes=[i for i in range(n_classes)])
binarize_predict = label_binarize(true_label, classes=[i for i in range(n_classes)])

# 读取预测结果
predict_score = predict_data.to_numpy()


# In[22]:


# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(binarize_predict[:,i], [socre_i[i] for socre_i in predict_score])
    roc_auc[i] = auc(fpr[i], tpr[i])

print("roc_auc = ",roc_auc)


# In[23]:


all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw = 2
plt.figure()


# In[24]:


plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)


for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of {0} (area = {1:0.2f})'.format(label_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class receiver operating characteristic ')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




