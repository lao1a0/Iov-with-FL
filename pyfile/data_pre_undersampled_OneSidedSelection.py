#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import os
import cv2
import math
import random
import matplotlib.pyplot as plt
import shutil
from sklearn.preprocessing import QuantileTransformer
from PIL import Image
import warnings
import time
warnings.filterwarnings("ignore")


# In[12]:


#Read dataset
df=pd.read_csv('../Car_Hacking_5%.csv')


# In[13]:


# Transform all features into the scale of [0,1]
numeric_features = df.dtypes[df.dtypes != 'object'].index
X=df[numeric_features]
y=df['Label']

# # OneSidedSelection

# In[ ]:


from imblearn.under_sampling import OneSidedSelection
t1=time.time()
# fc = NeighbourhoodCleaningRule()
fc=OneSidedSelection(random_state=1)
X_resampled, y_resampled  = fc.fit_resample(X, y)
t2=time.time()
print("OneSidedSelection的时间消耗："+str(t2-t1))
print("采样结果如下：")


# In[ ]:


new_df=pd.merge(X_resampled,y_resampled,left_index=True,right_index=True)
print(new_df)
print(new_df.Label.value_counts())
print(df.Label.value_counts())