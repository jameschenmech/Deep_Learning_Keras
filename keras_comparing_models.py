# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 11:25:07 2018

@author: junch
"""
from keras.optimizers import SGD
import pandas as pd
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
data = pd.read_csv('titanic_all_numeric.csv')

predictors = data.drop(['survived'], axis=1).as_matrix()
#as_matrix() turns dataframe into an ndarrary object

n_cols = predictors.shape[1]

target = to_categorical(data.survived)

#model 1 with fewer nodes
model_1 = Sequential()

model_1.add(Dense(10, activation='relu', input_shape=(n_cols,)))

model_1.add(Dense(10, activation='relu'))

model_1.add(Dense(2, activation='softmax'))
    


#model 2 with more nodes
model_2 = Sequential()

model_2.add(Dense(100, activation='relu', input_shape=(n_cols,)))

model_2.add(Dense(100, activation='relu'))

model_2.add(Dense(2, activation='softmax'))

#model 3 with more hidden layers
model_3 = Sequential()

model_3.add(Dense(100, activation='relu', input_shape=(n_cols,)))

model_3.add(Dense(100, activation='relu'))

model_3.add(Dense(100, activation='relu'))

model_3.add(Dense(100, activation='relu'))

model_3.add(Dense(2, activation='softmax'))


#Create SGD optimizer with specified learning rate:  my_optimizer
my_optimizer = SGD(lr=0.01)

#Compile the model
model_1.compile(optimizer=my_optimizer, loss='categorical_crossentropy')

model_2.compile(optimizer=my_optimizer, loss='categorical_crossentropy')

model_3.compile(optimizer=my_optimizer, loss='categorical_crossentropy')


from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

early_stopping_monitor = EarlyStopping(patience=3)

model_1 = model_1.fit(predictors, target, validation_split=0.3, epochs=100,
          callbacks = [early_stopping_monitor], verbose =False)

model_2 = model_2.fit(predictors, target, validation_split=0.3, epochs=100,
          callbacks = [early_stopping_monitor], verbose = False)

model_3 = model_3.fit(predictors, target, validation_split=0.3, epochs=100,
          callbacks = [early_stopping_monitor], verbose = False)

plt.plot(model_1.history['val_loss'],'r',label='fewer nodes')
plt.plot(model_2.history['val_loss'],'b',label='more nodes')
plt.plot(model_3.history['val_loss'],'g',label='more layers')
plt.xlabel('Epochs')
plt.ylabel('Validation Score')
plt.legend()
plt.show()
#Consider changing activation function if the below happens:
#dying neuron problem:  once turns negative node continues being negative
#Vanishing gradients:  layers have very small slopes