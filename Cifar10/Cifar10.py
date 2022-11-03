import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv
import random
import torchvision

from keras import optimizers
from keras.models import Sequential, load_model
from keras.datasets import cifar10
from keras.utils import np_utils,plot_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from torch import rand
from torchsummary import summary
from sklearn.model_selection import train_test_split
from PIL import Image
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

class Cifar10():
    def __init__(self):
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = cifar10.load_data()
        self.x_train,  self.x_val, self.y_train, self.y_val=train_test_split(self.X_train, self.Y_train, test_size=0.2, random_state=0)

        self.x_train = self.x_train/255.0
        self.x_val = self.x_val/255.0
        self.X_test = self.X_test/255.0

        self.y_train = np_utils.to_categorical(self.y_train, num_classes=10)
        self.y_val = np_utils.to_categorical(self.y_val, num_classes=10)
        self.y_test = np_utils.to_categorical(self.Y_test, num_classes=10)

        self.classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.vgg19 = None
        
        self.imgPath = ''
        self.result = None

    def LoadImage(self, fileName):
        self.imgPath = fileName

    def ShowTrainImages(self):

        lst =[random.randrange(len(self.x_train)) for _ in range(9) ]

        fig=plt.gcf()                                           #取得 pyplot 物件參考
        fig.set_size_inches(8, 8)

        for i in range(9):
            ax = plt.subplot(3, 3, i+1)                     #建立 5*5 個子圖中的第 i+1 個
            ax.imshow(self.x_train[lst[i]], cmap='binary')      #顯示子圖
            title = self.classes[self.y_train[lst[i]].argmax()]
            ax.set_title(title, fontsize=10)            #設定標題
            ax.set_xticks([])                                #不顯示 x 軸刻度
            ax.set_yticks([])                                #不顯示 y 軸刻度
        plt.show()

    def ShowModelStructure(self):
        self.vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(32, 32, 3), classes=10)  # when training
        tf.keras.models.load_model('myModel.h5') # when demo
        print(self.vgg19.summary())

        # self.Training()     # when training


    def transform_invert(self, img, transform_train):

        if 'Normalize' in str(transform_train):
            norm_transform = list(filter(lambda x: isinstance(x, torchvision.transforms.Normalize), transform_train.transforms))
            mean = torchvision.torch.tensor(norm_transform[0].mean, dtype=img.dtype, device=img.device)
            std = torchvision.torch.tensor(norm_transform[0].std, dtype=img.dtype, device=img.device)
            img.mul_(std[:, None, None]).add_(mean[:, None, None])

        img = img.transpose(0, 2).transpose(0, 1)
        img = np.array(img) * 255

        if img.shape[2] == 3:
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        elif img.shape[2] == 1:
            img = Image.fromarray(img.astype('uint8').squeeze())
        else:
            raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img.shape[2]) )

        return img

    def ShowDataAugmentation(self):

        if(self.imgPath == ''):
            return

        img = Image.open(self.imgPath)

        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

        img = torchvision.transforms.Resize((224, 224))(img)
        train_transform1 = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                        torchvision.transforms.RandomRotation(90),
                                        torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(norm_mean, norm_std)])
        img_tensor1 = train_transform1(img)
        convertImg1 = self.transform_invert(img_tensor1, train_transform1)
        
        train_transform2 = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), 
                                        torchvision.transforms.RandomResizedCrop(size=224, scale=(0.08, 0.5), ratio=(0.75, 1.3333333333333333)),
                                        torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(norm_mean, norm_std)])
        img_tensor2 = train_transform2(img)
        convertImg2 = self.transform_invert(img_tensor2, train_transform2)

        train_transform3 = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                        torchvision.transforms.RandomVerticalFlip(p=1),
                                        torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(norm_mean, norm_std)])
        img_tensor3 = train_transform3(img)
        convertImg3 = self.transform_invert(img_tensor3, train_transform3)

        for i in range(3):
            ax = plt.subplot(3, 3, i+1)
            if i == 0:
                ax.imshow(convertImg1)
            elif i == 1:
                ax.imshow(convertImg2)
            else:
                ax.imshow(convertImg3)
            ax.set_xticks([])                                #不顯示 x 軸刻度
            ax.set_yticks([])                                #不顯示 y 軸刻度
        plt.show()

    def Training(self):
        model=tf.keras.models.Sequential()
        model.add(self.vgg19)
        model.add(Flatten())
        model.add(Dense(1024,activation = 'relu'))
        model.add(Dropout(.25))
        model.add(Dense(1024,activation = 'relu'))
        model.add(Dropout(.25))
        model.add(Dense(256,activation = 'relu'))
        model.add(Dense(10,activation = 'softmax'))

        adam = optimizers.Adam(learning_rate=0.0001,epsilon=1e-08)

        model.compile(
            optimizer = adam,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        train_datagen = ImageDataGenerator(rotation_range=10, zoom_range = 0.1, width_shift_range=0.1, height_shift_range=0.1,
                                        shear_range = 0.1, horizontal_flip=True, vertical_flip=False)
        train_datagen.fit(self.x_train)

        reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.6, patience=3, verbose=1, min_lr=0.00001)

        self.result = model.fit(
            train_datagen.flow(self.x_train, self.y_train, batch_size = 64),
            validation_data = (self.x_val, self.y_val),
            epochs = 30,
            verbose = 1,
            callbacks = [reduce]
        )

        model.save('myModel.h5')

    def ShowAccuracyAndLoss(self):

        '''acc = self.result.history['accuracy']
        val_acc = self.result.history['val_accuracy']
        loss = self.result.history['loss']
        val_loss = self.result.history['val_loss']

        
        plt.subplot(2, 1, 1)
        plt.title("Training and Validation Accuracy")
        plt.plot(acc,color = 'green',label = 'Training Acuracy')
        plt.plot(val_acc,color = 'red',label = 'Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.subplot(2, 2, 2)
        plt.title('Training and Validation Loss')
        plt.plot(loss,color = 'blue',label = 'Training Loss')
        plt.plot(val_loss,color = 'purple',label = 'Validation Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')
        plt.savefig('accAndLoss.png')
        plt.show()'''

        '''img = cv.imread('accAndLoss.png')
        img = cv.resize(img,(580,460))
        cv.imwrite('accAndLoss2.png', img)'''

        pass

    def Inference(self):

        model = load_model('myModel.h5')
        
        img = Image.open(self.imgPath)
        img = img.resize((32, 32))
        x = np.array(img, dtype='float32')
        x = np.expand_dims(x, axis=0)
        pred = model.predict(x)

        p = pred.max()
        c = pred.argmax()
        return (p,self.classes[c])





        
        
