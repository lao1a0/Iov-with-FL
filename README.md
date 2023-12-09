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

# 环境安装

## 安装ananconda3

[Linux系统安装Anaconda3保姆级教程_linux安装anaconda3-CSDN博客](https://blog.csdn.net/arno_an/article/details/105229780)

[Anaconda更换清华源、中科大源-CSDN博客](https://blog.csdn.net/OuDiShenmiss/article/details/106380852)

```python
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.sjtug.sjtu.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes


conda config --remove-key channels 
conda config --show-sources
conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

vim ~/.condarc
# 复制进去
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/fastai/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
show_channel_urls: true
ssl_verify: false
```

## 安装syft

运行的环境叫做pysyft，环境安装教程

```python
# 方法一：
conda create -n pysyft python=3.7
conda activate pysyft
conda install pytorch==1.4.0 torchvision==0.5.0

# 方法二：
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn torch==1.4.0 torchvision==0.5.0
pip install syft==0.2.4 --no-dependencies
pip install lz4~=3.0.2 msgpack~=1.0.0 phe~=1.4.0 scipy~=1.4.1 syft-proto~=0.2.5.a1 tblib~=1.6.0 websocket-client~=0.57.0 pip install websockets~=8.1.0 zstd~=1.4.4.0 Flask~=1.1.1 tornado==4.5.3 flask-socketio~=4.2.1 lz4~=3.0.2 Pillow~=6.2.2 pip install requests~=2.22.0 numpy~=1.18.1
pip install protobuf==3.19.0

# 验证安装
python -c "import syft,torch"
```

## 配置git代理

```
git config --global --get http.proxy
git config --global --get https.proxy
git config --global http.proxy http://127.0.0.1:10811
git config --global https.proxy http://127.0.0.1:10811

git config --global --unset http.proxy
git config --global --unset https.proxy
```

## 安装jupyter

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
# Writing default config to: /home/zhaojia-raoxy/.jupyter/jupyter_notebook_config.py
jupyter notebook password # 配置密码
# 2ka2E68qNbgyQX4
# Enter password: 
# Verify password: 
# [NotebookPasswordApp] Wrote hashed password to /home/zhaojia-raoxy/.jupyter/jupyter_notebook_config.json
# 9a0e8a696832:5e10fda8163ded4ebfbd64d72cd2e16e37b185e6

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

安装扩展：[玩转Jupyter Notebook2-(推荐16个超实用插件) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/258976438?utm_oi=803714813804044288)

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

conda activate pysyft    # this is the environment for your project and code
conda install ipykernel
conda deactivate

conda activate base      # could be also some other environment
conda install nb_conda_kernels
jupyter notebook
```
注意：这里的 conda install nb_conda_kernels 是在 base 环境下操作的。安装好后，打开 jupyter notebook 就会显示所有的 conda 环境啦，点击随意切换。

------------------------------------------------

原文链接：https://blog.csdn.net/u014264373/article/details/119390267

### 方法二:

切换虚拟环境:参考博客：https://blog.csdn.net/u014264373/article/details/86541767
```
conda activate pysyft
conda install ipykernel

python -m ipykernel install --user --name abc --display-name "Python3 abc"	#将选择的conda环境注入Jupyter Notebook
```
打开Jupyter Notebook，顶部菜单栏选择Kernel–Change kernel–Python3 abc

如果报错ImportError: cannot import name ‘generator_to_async_generator’

```
pip uninstall -y ipython prompt_toolkit
pip install ipython prompt_toolkit
```

## 安装torch（可选）

> 当安装syft失败的时候在选择这个

https://blog.csdn.net/qq_46311811/article/details/123524762

```
conda create -n envName python=3.8 
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
