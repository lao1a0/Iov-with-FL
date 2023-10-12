import os
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
print(torch.cuda.is_available())#是否有可用的gpu
print(torch.cuda.device_count())#有几个可用的gpu
# import torch, gc
# gc.collect()
# torch.cuda.empty_cache()
# 配置device_ids，选择你想用的卡编号。
# device_ids = [0, 1]

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # model = torch.nn.DataParallel(model, device_ids)
    device = torch.device('cuda:1')
    t = torch.nn.Linear(3, 3).to(device)
    # input = torch.randn((3, 3)).requires_grad_().to("cuda:0")
    # output = t(input)
    print(t)
# kwargs = {'num_workers': 6, 'pin_memory': True} if torch.cuda.is_available() else {}
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"#声明gpu
# dev=torch.device('cuda:1')#调用哪个gpu
# a=torch.rand(100,100).to(dev)
# print(a)

# loss=torch.sum(output)
# torch.autograd.grad(loss,input,retain_graph=True)  ## 输出应该是一个gpu上的梯度矩阵
# loss.backward()

