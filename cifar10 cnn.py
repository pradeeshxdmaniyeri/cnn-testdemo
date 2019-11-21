# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:03:21 2019

@author: pradeesh
"""

from keras.datasets import cifar10
(xtrain,ytrain),(xtest,ytest)=cifar10.load_data()

from keras import Sequential
from keras.layers import Dense
from keras.layers import Activation,MaxPool2D,Flatten,Conv2D,Dropout
from keras.utils import to_categorical

ytrain=to_categorical(ytrain)
ytest=to_categorical(ytest)

xtrain=xtrain.astype('float32')
xtest=xtest.astype('float32')
xtrain/255
xtest/255
#ytest=ytest.astype('float32')
ytest
xtest
c=Sequential()
c.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(32,32,3)))
c.add(MaxPool2D(pool_size=(5,5)))
c.add(Conv2D(64,(3,3),padding='same',activation='relu'))
c.add(MaxPool2D(pool_size=(2,2)))
c.add(Flatten())
c.add(Dense(units=100,kernel_initializer='uniform',activation='relu'))
c.add(Dense(units=10,kernel_initializer='uniform',activation='sigmoid'))
c.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
c.fit(xtrain,ytrain,batch_size=10,epochs=1)
#from sklearn.metrics import r2_score
a=c.predict(xtest)
a
aa=c.evaluate(xtest,ytest)
print(aa)
len(a)
len(ytest)

import matplotlib.pyplot as plt
plt.imshow(xtrain[0])
plt.imshow(xtest[0])

