# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:34:27 2018

@author: junch
"""

#import numpy as np
import pandas as pd
from keras.layers import Dense

from keras.models import Sequential

predictors = pd.read_csv('hourly_wages.csv',delimiter=',', header=0)

target = predictors['wage_per_hour']

predictors.drop(['wage_per_hour'], axis=1, inplace=True)

n_cols = predictors.shape[1]

model = Sequential() #Sequential is one way to model, other ways more complex

model.add(Dense(100, activation='relu', input_shape = (n_cols,)))#first hidden layer
#all nodes in previous layer connect to all nodes in current layer

model.add(Dense(100, activation='relu'))#second hidden layer 

model.add(Dense(1))

#choose an optimizer which controls learning rate
#as a default choose "Adam" as a start
#choose a lost function, default to Mean Squared Error
model.compile(optimizer='adam',  loss='mean_squared_error')

#fit the model:  apply back propagation and grident descient to update the weights
#scale the features
model.fit(predictors,target, epochs=10)

print("Loss Function: " + model.loss)

