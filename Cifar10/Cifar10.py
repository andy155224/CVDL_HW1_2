import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv
import random

from keras.models import Sequential, load_model
from keras.datasets import cifar10
from keras.utils import np_utils,plot_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from torch import rand

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
x_train = X_train.astype('float32')/255
x_test = X_test.astype('float32')/255
y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(Y_test)

lst =[random.randrange(len(x_train)) for _ in range(9) ]
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


for i in range(9):
    ax=plt.subplot(3, 3, i+1)                     #建立 5*5 個子圖中的第 i+1 個
    ax.imshow(x_train[i], cmap='binary')      #顯示子圖
    #title=str(i) + "." + classes[y_train[i]] + str(y_train[i]) 
    ax.set_title('img', fontsize=10)            #設定標題
    ax.set_xticks([]);                                #不顯示 x 軸刻度
    ax.set_yticks([]);                                #不顯示 y 軸刻度
    i += 1                                              #樣本序號增量 1
plt.show()