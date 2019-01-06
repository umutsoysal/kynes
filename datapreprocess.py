#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 18:43:58 2018

@author: usoysal
"""

print("Hello World")


# IMPORT THE LIBRARIES

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F


urad_frame=pd.read_csv("scripts_datasets/Datasets/mrds_v3_URAD.csv",sep=',')
success=np.zeros([len(urad_frame),1])
input_frame=np.zeros([len(urad_frame),17*22])
for i in range(len(urad_frame)):
    #print(i)
    img_name = urad_frame.iloc[i, 0]
    #print(img_name)
    im = Image.open('clipped-files/clipped/URadiometricClipBuff/'+img_name)
    imarray = np.array(im)
    imarray_flat=imarray.flatten()
    imarray_flat=imarray_flat[0:374]
    print(len(imarray_flat))
    input_frame[i,:]=imarray_flat
    if urad_frame.gold[i]==3:
        success[i]=1
    

urad_train=np.hstack((success,input_frame))

urad_train=torch.from_numpy(urad_train)
urad_train=urad_train[:,0:353]

#urad_train=urad_train.type[torch.DoubleTensor]
np.savetxt("urad_train.csv", urad_train, delimiter=",")


size = 128, 128
im.thumbnail(size, Image.ANTIALIAS)
im.save('scaledimage', "JPEG")

