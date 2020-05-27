# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:21:47 2020

@author: 20009
"""


import cv2
import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from keras import backend as K
from keras.models import Sequential 
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D



image=cv2.imread(r"C:\dog.jpg")
# ax.imshow(image)

def show(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # plt.imshow 預設圖片是 rgb 的
    plt.show()
show(image)


# create model
#Sequential 是一個多層模型
#透過 add() 函式將一層一層 layer 加上去
#data_format='channels_last' 尺寸为 (batch, rows, cols, channels)
#搭建一個 3 個 1*1 的 filters
model=Sequential()
model.add(Conv2D(3,(1,1),
          padding="same",
         data_format='channels_last',
         activation='relu',
         input_shape=image.shape))
# model.add(Conv2D(3,(3,3),
#           padding="same",
#          data_format='channels_last',
#          activation='relu'))
# model.add(Conv2D(3,(3,3),
#           padding="same",
#          data_format='channels_last',
#          activation='relu'))
# model.add(Conv2D(3,(3,3),
#           padding="same",
#          data_format='channels_last',
#          activation='relu'))
# model.add(Conv2D(3,(3,3),
#           padding="same",
#          data_format='channels_last',
#          activation='relu'))
# model.add(Conv2D(3,(3,3),
#           padding="same",
#          data_format='channels_last',
#          activation='relu'))
# model.add(Conv2D(3,(3,3),
#           padding="same",
#          data_format='channels_last',
#          activation='relu'))
# model.add(Conv2D(3,(3,3),
#           padding="same",
#          data_format='channels_last',
#          activation='relu'))
#作業: 接續搭建一個 4 個 3*3 的 filters 



print(model.summary())
#權重都是亂數值


# keras 在讀取檔案實是以 batch 的方式一次讀取多張，
#但我們這裡只需要判讀一張，
#所以透過 expand_dims() 函式來多擴張一個維度
image_batch=np.expand_dims(image,axis=0)
print(image_batch.shape)



#model.predict() 函式，得到回傳便是 feature map
image_conv=model.predict(image_batch)
img=np.squeeze(image_conv,axis=0)
print(img.shape)
plt.imshow(img)

