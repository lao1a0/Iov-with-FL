# 工作安排

- [ ] 看完[[2104.07586\] See through Gradients: Image Batch Recovery via GradInversion (arxiv.org)](https://ar5iv.labs.arxiv.org/html/2104.07586?_immersive_translate_auto_translate=1)这篇论文

- [ ] 复现[[2104.07586\] See through Gradients: Image Batch Recovery via GradInversion (arxiv.org)](https://ar5iv.labs.arxiv.org/html/2104.07586?_immersive_translate_auto_translate=1)

- [ ] 复现[LiYangHart/Intrusion-Detection-System-Using-CNN-and-Transfer-Learning: Code for intrusion detection system development using CNN models and transfer learning (github.com)](https://github.com/LiYangHart/Intrusion-Detection-System-Using-CNN-and-Transfer-Learning)

- [ ] 复现[invertinggradients/environment.yml at master · JonasGeiping/invertinggradients (github.com)](https://github.com/JonasGeiping/invertinggradients/blob/master/environment.yml)

- [ ] 看完[Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion](https://blog.csdn.net/FengF2017/article/details/115698179)

- [ ] 复现[NVlabs/DeepInversion](https://github.com/NVlabs/DeepInversion/tree/master)

- [x] 在cifar-10上重新做一次实验


 ```python
 nohup "jupyter nbconvert --to html --execute fl_CIFAR100_LeNet.ipynb"  > jp.log 2>&1 &
 
 HrtGzAWte9t&mFw#fD #MoB的密码
 
 pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple
 pip install tensorboardX -i https://pypi.tuna.tsinghua.edu.cn/simple
 ```



# cifar-10 实验

> 重新修改了一下代码，进行500轮的训练。

- LeNet加了联邦学习与不加的效果都是差不多，在测试集上能够达到65%左右的准确率（[noFL 64%](https://github.com/lao1a0/Iov-with-FL/blob/main/12-22组会/LeNet_CIFAR10_fl_no.ipynb);[FL 63%](https://github.com/lao1a0/Iov-with-FL/blob/main/12-22组会/LeNet_CIFAR10_fl.ipynb)）

<img src="./img/1.png" style="zoom: 80%;" />

>  用了联邦学习后，在测试集上的准确率会比不用的稍微高一点（右侧第二对）土黄色后期深绿色的高。
>
> 用了联邦学习后，在测试集上的损失值会比不用的稍微高一点（右侧第三队）深蓝色后期比洋红色高。
>
> 轮收敛速度的话，联邦学习会比不用的快（loss ，蓝色是用了联邦平均的 ； acc 绿色是用了联邦平均的）

- ResNet18会产生过拟合的现象，能够达到80%左右的准确率（）

