

现在的设计思路：

1. 对数据集进行KSVD矩阵分解，留一半进行全局模型更新
2. 全局模型更新部分的最后一层全连接层进行AutoML数据增强

[wkcn/CaffeSVD: 使用SVD、K-Means、降低权值精度的方法压缩Cifar-10神经网络的全连接层 (github.com)](https://github.com/wkcn/CaffeSVD)