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

#%% 
my_slice = train_images[10:100] #train_images[10:100,:,:]
print(my_slice.shape)

my_slice = train_images[:,14:,14:]
my_slice = train_images[:,7:-7,7:-7]

# %% data batch 
batch = train_images[:128]
batch = train_images[128:256] # batch axis ,batch dimension

# %% Real world 

# (sample,features)
# (sample,timesteps, features) 
# (samples, width, height, channels )

## element wise relu 
def naive_relu(x):
    """ x is 2D numpy array"""
    assert len(x.shape) == 2
    
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] = max(x[i,j],0)
            
    return x

def naive_add(x,y):
    """ x,y are 2D numpy array""" 
    assert len(x.shape) == 2
    assert x.shape == y.shape
    
    x = x.copy() # avoid overwirting 
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] += y[i,j]

    return x

#import numpy as np 
#z = x + y
#z = np.maximum(z,0) ##

## matrix vector addition 
def naive_add_matrix_and_vector(x, y):
    # x is 2D numpy array
    # y is numpy vector 
    assert len(x.shape) == 2 
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    
    x = x.copy()
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] += y[j]
            
    return x
            
# %%
import numpy as np

# x is a random tensor with shape (64, 32, 10)
x = np.random.random((64, 3, 32, 10))
# y is a random tensor with shape (32, 10)
y = np.random.random((32, 10))

# the output has shape (64, 3, 32, 10) like x
z = np.maximum(x, y)


# %% tensor dot product 
y1 = y.transpose()
z = np.dot(x,y1) 

# or 
from keras import backend as K
#z = K.dot(x,y1)

# %% dot (naive dot product)
def naive_vector_dot(x, y):
    # x and y are Numpy vectors
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]

    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z

def naive_vector_dot(x, y):
    # x and y are Numpy vectors
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]

    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z