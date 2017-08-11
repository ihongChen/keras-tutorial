# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:32:39 2017

@author: 116952
"""


## binary , multi-class , regression
# %%
## A layer (sequential)
from keras import layers,models
# A dense layer with 32 output units
#layer = layers.Dense(32, input_shape=(784,))

#model = models.Sequential()
#model.add(layers.Dense(32, activation='relu', input_shape=(784,)))
#model.add(layers.Dense(10, activation='softmax'))


# %% Functional API
input_tensor = layers.Input(shape = (784,))
x = layers.Dense(32, activation='relu')(input_tensor)
output_tensor = layers.Dense(10,activation = 'softmax')(x)

model = models.Model(inputs = input_tensor, outputs = output_tensor)

# %% optimizer
from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='mse',
              metrics=['accuracy'])

# %% train model 
model.fit(input_tensor, target_tensor, batch_size=128, epochs=1)

