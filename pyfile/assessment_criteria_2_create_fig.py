import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import LoadData, write_result
import pandas as pd
import os
from sklearn.metrics import *  # pip install scikit-learn
import matplotlib.pyplot as plt  # pip install matplotlib
import numpy as np  # pip install numpy
from numpy import interp
from sklearn.preprocessing import label_binarize
import torchvision

class Arguments():
    def __init__(self):
        # self.model_name = 'cnn'
        self.model_name = 'alexnet'
        self.data_name =  'yeo' # yeo quantile_ransformer
        self.verify_root = r"../data2/test_{}_224/".format(self.data_name)
        self.predicted_label_txt = '../files/predicted_label_{}_{}.txt'.format(self.model_name, self.data_name)
        self.true_label_txt = '../files/true_label_{}_{}.txt'.format(self.model_name, self.data_name)  # 正确标签的位置
        self.model_weight = "../model/{}_{}_lr0.01_epoch50.pt".format(self.data_name, self.model_name)
        self.save_name = '{}_{}_lr0.01_epochs50_test'.format(self.model_name, self.data_name)
        self.confusion_name = '{}_{}_lr0.01_epochs50_test_confusion'.format(self.model_name, self.data_name)
        self._fig_title='{}_{}'.format(self.model_name,self.data_name)
        self.model_name = 'alexnet'
        self.data_name = 'yeo' # yeo quantile_ransformer
        self.roc_name = '{}_{}_lr0.01_epochs50_verify_roc'.format(self.model_name, self.data_name)


args = Arguments()

target_data = pd.read_csv(args.true_label_txt, sep="\t", names=["loc", "type"])
true_label = [i for i in target_data["type"]]
# print(true_label)
predict_loc ='../files/{}.csv'.format(args.save_name)    # 3.ModelEvaluate.py生成的文件
predict_data = pd.read_csv(predict_loc)#,index_col=0)
predict_label = predict_data.to_numpy().argmax(axis=1)
predict_score = predict_data.to_numpy().max(axis=1)
# predict_score = predict_data.to_numpy().max(axis=1)
# # 精度，准确率， 预测正确的占所有样本种的比例
print(args.save_name)
print(len(predict_label))
accuracy = accuracy_score(true_label, predict_label)
print("精度: ",accuracy)
# 查准率P（准确率），precision(查准率)=TP/(TP+FP)
precision = precision_score(true_label, predict_label, labels=None, pos_label=1, average='macro') # 'micro', 'macro', 'weighted'
print("查准率P: ",precision)
# 查全率R（召回率），原本为对的，预测正确的比例；recall(查全率)=TP/(TP+FN)
recall = recall_score(true_label, predict_label, average='macro') # 'micro', 'macro', 'weighted'
print("召回率: ",recall)
# F1-Score
f1 = f1_score(true_label, predict_label, average='macro')     # 'micro', 'macro', 'weighted'
print("F1 Score: ",f1)
# 混淆矩阵
label_names = ['R', 'RPM',  'gear', 'DoS', 'Fuzzy']
confusion = confusion_matrix(true_label, predict_label, labels=[i for i in range(len(label_names))])
plt.matshow(confusion, cmap=plt.cm.Oranges)   # Greens, Blues, Oranges, Reds
plt.colorbar()
for i in range(len(confusion)):
    for j in range(len(confusion)):
        plt.annotate(confusion[j,i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.xticks(range(len(label_names)), label_names)
plt.yticks(range(len(label_names)), label_names)
plt.title("{} Confusion Matrix".format(args._fig_title))
# plt.show()
plt.savefig('../img/{}_Confusion_Matrix.png'.format(args._fig_title),bbox_inches="tight")
print('执行完毕，生成文件：../img/{}_Confusion_Matrix.png'.format(args._fig_title))
# ROC曲线
n_classes = len(label_names)
# binarize_predict = label_binarize(predict_label, classes=[i for i in range(n_classes)])
binarize_predict = label_binarize(true_label, classes=[i for i in range(n_classes)])
# 读取预测结果
predict_score = predict_data.to_numpy()
# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(binarize_predict[:,i], [socre_i[i] for socre_i in predict_score])
    roc_auc[i] = auc(fpr[i], tpr[i])
print("roc_auc = ",roc_auc)
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
plt.plot(fpr["macro"], tpr["macro"],
         label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)


for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of {0} (area = {1:0.2f})'.format(label_names[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('{} Multi-class receiver operating characteristic'.format(args._fig_title))
plt.legend(loc="lower right")
# plt.show()
plt.savefig('../img/{}_roc_fig.png'.format(args._fig_title),bbox_inches="tight")
print('执行完毕生成文件：../img/{}_roc_fig.png'.format(args._fig_title))