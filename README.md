# 指令快查

服务器1：

- 10.126.62.102:22
- 用户名：zhaojia-raoxy
- 密码：raoxy

服务器2：

- 密码：raoxy@123

```python
# conda
conda create -n your_env_name python=x.x
conda activate Soteria
conda deactivate

# 联网
curl http:\/\/10.10.43.3\/drcom\/login\?callback=dr1558050177253\&DDDDD=22125303\&upass=Xrq@9686\&0MKKey=123456\&R1=0\&R3=0\&R6=0\&para=00\&v6ip=\&\_=1558050050455
```
## 后台挂机

```
ps -aux|grep "jupyter-notebook"
nohup jupyter notebook  > jp.log 2>&1 &
[1] 25212
```

## 配置git代理

```
git config --global --get http.proxy
git config --global --get https.proxy
git config --global http.proxy http://127.0.0.1:10811
git config --global https.proxy http://127.0.0.1:10811

git config --global --unset http.proxy
git config --global --unset https.proxy

conda info --env
```

## 换源

[Anaconda更换清华源、中科大源-CSDN博客](https://blog.csdn.net/OuDiShenmiss/article/details/106380852)

## 配置jupyter

### 安装教程

[Linux 服务器上部署搭建 Jupyter notebook【详细教程】_jupyter linux-CSDN博客](https://blog.csdn.net/W_nihao_123456/article/details/108421145)

- 首先，您需要在服务器上安装jupyter notebook，您可以使用pip或conda命令来安装，例如：
```bash
pip install jupyter
```
或者
```bash
conda install jupyter
```
- 然后，您需要生成一个配置文件，用于设置jupyter notebook的一些参数，例如允许远程访问的ip地址，端口号，密码等。您可以使用以下命令来生成配置文件：
```bash
jupyter notebook --generate-config
```
- 接着，您需要编辑配置文件，您可以使用vim或其他文本编辑器来修改，配置文件的默认位置是`~/.jupyter/jupyter_notebook_config.py`。您需要修改以下几项：
```
c = get_config()  #noqa
c.NotebookApp.password = u'sha1:86e016e40af4:03ec979f434933c35c647637eaab87e4832a26ad'
c.NotebookApp.ip='0.0.0.0'
c.NotebookApp.port = 8888	#随便指定一个闲置端口
c.NotebookApp.open_browser = False	#禁止自动打开浏览器
c.NotebookApp.allow_remote_access = True	#远程访问
c.NotebookApp.allow_root = True
```
- 最后，您需要启动jupyter notebook服务，您可以使用以下命令来启动：
```bash
jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```
- 如果您想让jupyter notebook服务在后台持续运行，您可以使用screen或nohup命令来实现，例如：
```bash
nohup jupyter notebook > jupyter.log 2>&1 &
```
- 现在，您可以在本地浏览器中输入服务器的ip地址和端口号来访问jupyter notebook，例如`http://123.456.789.0:8888`，然后输入您设置的密码来登录。

### 安装扩展

[玩转Jupyter Notebook2-(推荐16个超实用插件) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/258976438?utm_oi=803714813804044288)

```
jupyter notebook password
conda install jupyter_contrib_nbextensions
raoxy@bjtucs-ubuntu:~$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
raoxy@bjtucs-ubuntu:~$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
raoxy@bjtucs-ubuntu:~$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
raoxy@bjtucs-ubuntu:~$ conda config --set show_channel_urls yes
raoxy@bjtucs-ubuntu:~$ conda install -c conda-forge jupyter_contrib_nbextensions
```

[jupyter扩展插件Nbextensions的安装、使用](https://blog.csdn.net/zhongkeyuanchongqing/article/details/117560437)

## 切换conda 虚拟环境

### 方法一:

```

conda activate my-conda-env    # this is the environment for your project and code
conda install ipykernel
conda deactivate

conda activate base      # could be also some other environment
conda install nb_conda_kernels
jupyter notebook
```
推荐指数： ⭐️⭐️⭐️⭐️⭐️

注意：这里的 conda install nb_conda_kernels 是在 base 环境下操作的。安装好后，打开 jupyter notebook 就会显示所有的 conda 环境啦，点击随意切换。

————————————————

原文链接：https://blog.csdn.net/u014264373/article/details/119390267

### 方法二:

切换虚拟环境:参考博客：https://blog.csdn.net/u014264373/article/details/86541767
```
conda activate abc	#激活虚拟环境
conda install ipykernel
python -m ipykernel install --user --name abc --display-name "Python3 abc"	#将选择的conda环境注入Jupyter Notebook
```
打开Jupyter Notebook，顶部菜单栏选择Kernel–Change kernel–Python3 abc

如果报错ImportError: cannot import name ‘generator_to_async_generator’

```
pip uninstall -y ipython prompt_toolkit
pip install ipython prompt_toolkit
```

## 安装torch

https://blog.csdn.net/qq_46311811/article/details/123524762

    ```
    conda create -n envName python=3.8 
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    ```



# 研究点一：

## 任务安排

https://mega.nz/folder/z0pnGA4a#WFEUISyS5_maabhcEI7HQA

数据集:https://ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset

- [x] 先用100%数据集来跑通畅原来的代码，看一下还能不能达到原始精度
- [ ] 找对照组
- [ ] 毕设的第五部分，还可以加入开发了一个实地的系统进行验证

# 研究点二：

## 任务安排

- [ ] 先把中期提到的这几个论文的实验先复现了

## 参考文献

### Deep Leakage from Gradients

### See Through Gradients: Image Batch Recovery via GradInversion

原文：https://openaccess.thecvf.com/content/CVPR2021/html/Yin_See_Through_Gradients_Image_Batch_Recovery_via_GradInversion_CVPR_2021_paper.html
中文：https://blog.csdn.net/qq_34206952/article/details/116712207

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://github.com/lao1a0/Iov-with-FL/assets/46106062/02aa4078-7abe-4a26-b308-3da68b1192d3" width = "80%" alt=""/><br/>
	<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 5px;">图：WAF的发展历程
  	</div>
</center>

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
