#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 20:08:34 2017

@author: ihong
"""
#%%
# =============================================================================
# 2.1 -- Hello word in dl --- MNIST
# =============================================================================
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#%% 
## train data
print('train dim :{}'.format(train_images.shape))
print(len(train_labels))
train_labels
## test data 

# %% build nn model 
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])
#%%
# reshape input 
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# preparing labels 
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# %% ready to fire ==> train 
network.fit(train_images, train_labels, epochs=5, batch_size=128)
# %% evaluate 
test_loss,test_acc = network.evaluate(test_images,test_labels)
print('test_acc:',test_acc)

#%%
# =============================================================================
# 2.2 Data representations for nn
# =============================================================================

import numpy as np 
x = np.array(12)
x.ndim # number dim : 0
print('np.array(12) ndim ==>',x.ndim)
print('np.array(12) shape ==>',x.shape) # ()
x1 = np.array([1,2,3,4,5])
print('np.array([1,2,3,4,5]) ndim ==>',x1.ndim)
print('np.array([1,2,3,4,5]) shape ==>',x1.shape) # (5,)
x = np.array([[5, 78, 2, 34, 0],
                  [6, 79, 3, 35, 1],
                  [7, 80, 4, 36, 2]])
print('np.array [[]] ndim ==> ',x.ndim)
print('np.array [[]] shape ==> ',x.shape)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print('train_images dim ==> ',train_images.ndim)
print('train images shape ==>',train_images.shape)
print('train images dtypes ==>',train_images.dtype)

# %%
## display digit


digit = train_images[4]
import matplotlib.pyplot as plt
plt.imshow(digit,cmap=plt.cm.binary)

