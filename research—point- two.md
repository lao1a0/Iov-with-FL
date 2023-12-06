
# 研究点二：

## 任务安排

- [ ] 先把中期提到的这几个论文的实验先复现了

## 参考文献

### Deep Leakage from Gradients

**论文总结：**

形式上，给定一个机器学习模型$F()$及其权重$w$ ，如果我们有一对输入和标签的梯度$∇ w$，我们可以获得训练数据？传统观点认为答案是否定的，但我们证明这实际上是可能的。

在这项工作中，我们演示了来自梯度的深度泄漏（DLG）：共享梯度可能会泄漏私有训练数据。我们提出了一种优化算法，可以在几次迭代中获得训练输入和标签。为了执行攻击，
1. 随机生成一对“虚拟”输入和标签，然后执行向前和向后传播。
2. 从虚拟数据导出虚拟梯度之后，优化虚拟输入和标签，==最小化虚拟梯度和真实的梯度之间的距离==，而不是像典型训练中那样优化模型权重（如图2所示）。
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://github.com/lao1a0/Iov-with-FL/assets/46106062/57e5633c-13ff-4bf0-afb0-4ebfbcf95a0b" width = "80%" alt=""/><br/>
	<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 5px;">图2：DLG算法的概述。要更新的变量用粗体边框标记。当正常参与者使用其私有训练数据计算 $∇ w$以更新参数时，恶意攻击者更新其虚拟输入和标签以最小化梯度距离。当优化完成时，恶意用户能够从诚实的参与者那里获得训练集
  	</div>
</center>
3. 匹配梯度使虚拟数据接近原始数据（图5）。
4. 当优化完成时，私有训练数据（包括输入和标签）将完全显示。

我们的“深度”泄漏是一个优化过程，==不依赖于任何生成模型==;因此，DLG不需要任何其他关于训练集的额外先验，相反，它可以从共享梯度中推断标签，并且DLG产生的结果（图像和文本）是确切的原始训练样本，而不是合成的相似替代品。

深度泄漏对多节点机器学习系统提出了严峻的挑战。在集中式分布式训练中（图0（a）），通常不存储任何训练数据的参数服务器能够窃取所有参与者的本地训练数据。对于分散式分布式训练（图0（b）），情况变得更糟，因为任何参与者都可以窃取其邻居的私人训练数据。

**防御策略：**

为了防止深度泄漏，展示了三种防御策略：梯度扰动，低精度和梯度压缩。对于梯度扰动，发现尺度高于 $10^{-2}$的高斯和拉普拉斯噪声都是很好的防御。当==半精度攻击无法防御时，梯度压缩成功地防御了攻击，修剪后的梯度大于20%==。

**算法详细介绍：**

为了从梯度中恢复数据，我们首先随机初始化虚拟输入$𝐱′$和标签输入$𝐲′$ 。然后，我们将这些“虚拟数据”输入模型并获得“虚拟梯度”。

$$
\nabla W^{'}=\frac{\partial\ell(F(\mathbf{x}^{'},W),\mathbf{y}^{'})}{\partial W}
$$

优化接近原始的虚拟梯度也使虚拟数据接近真实的训练数据（图5中所示的趋势）。给定某一步的梯度，我们通过最小化以下目标来获得训练数据

$$

\mathbf{x'}^*,\mathbf{y'}^*=\underset{\overset{i}{\operatorname*{x'},y'}}{\operatorname*{\arg\min}}\|\nabla W^{'}-\nabla W\|^2=\underset{\overset{x',y'}{\operatorname*{x'},y'}}{\operatorname*{\arg\min}}\|\frac{\partial\ell(F(\mathbf{x'},W),\mathbf{y'})}{\partial W}-\nabla W\|^2

$$

距离$\left\|\nabla W^{'}-\nabla W\right\|^{2},$相对于伪输入$𝐱′$是可微的，并且标签$𝐲′$因此可以使用标准的基于梯度的方法来优化。注意，此优化需要 $2^{nd}$ 阶导数。我们做了一个温和的假设，即$F$是二次可微的，这适用于大多数现代机器学习模型（例如，大多数神经网络）和任务。


### See Through Gradients: Image Batch Recovery via GradInversion

原文：https://openaccess.thecvf.com/content/CVPR2021/html/Yin_See_Through_Gradients_Image_Batch_Recovery_via_GradInversion_CVPR_2021_paper.html
中文：https://blog.csdn.net/qq_34206952/article/details/116712207

## 三种防御：

**PRECODE - A Generic Model Extension to Prevent Deep Gradient Leakage**

**Protect Privacy from Gradient Leakage Attack in Federated Learning**

### **[Soteria](https://github.com/jeremy313/Soteria)**


**Provable-Defense-against-Privacy-Leakage-in-Federated-Learning-from-Representation-Perspective**

代码：https://github.com/jeremy313/Soteria/blob/main/DLG_attack/Defend_DLG.ipynb

防御的攻击：

- [Inverting Gradients - How easy is it to break Privacy in Federated Learning?](https://github.com/JonasGeiping/invertinggradients#inverting-gradients---how-easy-is-it-to-break-privacy-in-federated-learning)
- [Deep Leakage From Gradients ](https://github.com/mit-han-lab/dlg#deep-leakage-from-gradients-arxiv-webside)

源码: [jeremy313/Soteria: Official implementation of "Provable Defense against Privacy Leakage in Federated Learning from Representation Perspective" (github.com)](https://github.com/jeremy313/Soteria)

论文介绍：通过扰动数据表示，使得攻击者难以从共享的梯度信息中重建原始数据，同时保持联邦学习的性能。具体来说，
提出了一种针对 FL 中模型反转攻击的防御方法。核心思想是 <span class="burk">学习扰动数据表示</span> ，使得重构数据的质量严重下降，同时保持 FL 的性能。此外，推导出了应用我们的防御后 FL 的认证鲁棒性保证和 FedAvg 的收敛性保证。在 MNIST 和 CIFAR10 上进行了实验，以抵御 DLG 攻击和 GS 攻击。在不牺牲准确率的情况下，结果表明，与基线防御方法相比，提出的防御方法可以将重构数据和原始数据之间的均方误差增加高达 160 倍对于 DLG 攻击和 GS 攻击。Soteria包括以下几个步骤：

- 步骤一：使用一个生成对抗网络（GAN）来生成扰动后的数据表示。GAN由一个生成器和一个判别器组成，生成器的目标是生成与原始数据表示相似但不完全相同的数据表示，判别器的目标是区分真实的数据表示和生成的数据表示。通过对抗训练，生成器可以学习到一个扰动函数，使得生成的数据表示具有一定的隐私保护能力。
- 步骤二：使用扰动后的数据表示作为联邦学习的输入。每个参与者使用自己的本地数据集和生成器来生成扰动后的数据表示，并用其替换原始的数据表示。然后，每个参与者使用扰动后的数据表示来训练自己的本地模型，并将本地模型的梯度信息上传到中心服务器。中心服务器使用梯度信息来更新全局模型，并将全局模型发送回每个参与者。
- 步骤三：使用全局模型进行预测或者评估。每个参与者可以使用全局模型来对自己或者其他人的数据进行预测或者评估。由于全局模型是基于扰动后的数据表示训练的，因此它具有一定的鲁棒性和泛化能力。

 ![avatar](/home/raoxy/img/1697845-20231011123023792-122113034.png)

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="~/img/1697845-20231011123023792-122113034.png" width = "80%" alt=""/><br/>
	<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 5px;">图：WAF的发展历程
  	</div>
</center>

Soteria方法可以有效地防御模型反演攻击，因为它满足了以下几个条件：

- 条件一：扰动函数是不可逆的，即无法从扰动后的数据表示恢复出原始的数据表示。这是因为扰动函数是基于GAN生成器学习的，而GAN生成器是一个非线性映射，且存在多对一或者一对多的情况。
- 条件二：扰动函数是随机的，即对于同一个原始数据表示，每次生成的扰动后的数据表示都不相同。这是因为扰动函数是基于GAN生成器和随机噪声向量结合产生的，而随机噪声向量每次都不同。
- 条件三：扰动函数是可控制的，即可以根据不同的隐私需求调整扰动程度。这是因为扰动函数是基于GAN生成器和随机噪声向量结合产生的，而随机噪声向量可以控制其维度和分布。

我们提供了针对DLG攻击和GS攻击的防御实现。我们的代码是基于 [DLG original repo](https://github.com/mit-han-lab/dlg) and [GS original repo](https://github.com/JonasGeiping/invertinggradients).

```
pytorch=1.2.0
torchvision=0.4.0
matplotlib
pyhton=3.6
juypter
```

- DLG attack

对于DLG攻击，可以通过改变中的百分位数参数来改变防御的剪枝率

```
thresh = np.percentile(deviation_f1_x_norm_sum.flatten().cpu().numpy(), 1)
```
我们还提供了模型压缩防御的实现。您可以取消相应代码的注释来尝试它。

- GS attack

对于GS攻击，可以通过在论文上运行再现汽车图像的结果

```
python reconstruct_image.py --target_id=-1 --defense=ours --pruning_rate=60 --save_image
```

您可以通过运行来尝试模型压缩防御

```
python reconstruct_image.py --target_id=-1 --defense=prune --pruning_rate=60 --save_image
```

考虑到计算效率，我们在代码中使用$\frac{||r||}{||d(f(r))/dX||}$来近似$||\frac{1}{r(d(f(r))/dX)}||$。你可以编辑代码直接计算$||\frac{1}{r(d(f(r))/dX)}||$，这样可以用更高的计算成本获得更好的防御效果

## 三种攻击：

**Using Highly Compressed Gradients in Federated Learning for Data Reconstruction Attacks**

**Auditing Privacy Defenses in Federated Learning via Generative Gradient Leakage**

**Inverting gradients - how easy is it to break privacy in federated learning?**
