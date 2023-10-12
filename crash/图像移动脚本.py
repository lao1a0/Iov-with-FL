#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 复制图像到另一个文件夹
# 文件所在文件夹
import os
#file_dir = '../data/train_224'
#name = '../data/train_224_sub'
file_dir = '../data/test_224'
name = '../data/test_224_sub'

file_list = os.listdir(file_dir)

import random
import shutil
for sub_dir in file_list:
    sub_imgs_dir=os.listdir(os.path.join(file_dir,sub_dir))
#     print(os.path.join(file_dir,sub_dir))
#     print(sub_imgs_dir)
#     break
#     print(os.path.join(name,sub_dir))
    len_str=len(sub_imgs_dir)
    for i in range(len_str//10):
        select_img=sub_imgs_dir[random.randint(0,len_str-1)]
#         print(os.path.join(name,sub_dir,select_img))
#         print(os.path.join(file_dir,sub_dir,select_img))
#         break
        if os.path.exists(os.path.join(name,sub_dir)):
            shutil.copy(os.path.join(file_dir,sub_dir,select_img), os.path.join(name,sub_dir,select_img))
        else:
            os.makedirs(os.path.join(name,sub_dir))
            shutil.copy(os.path.join(file_dir,sub_dir,select_img), os.path.join(name,sub_dir,select_img))

