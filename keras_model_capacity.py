# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 11:49:00 2018

@author: junch
"""

#Over-fitting and under-fitting
#Higher capicty --> on overfitting side of graph
#work flow:
#1.Start with small network
#2.Get validation score
#3.Keep increasing capacity until validations core is no longer improving

from keras.optimizers import SGD
import pandas as pd
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense



mist = pd.read_csv('mnist.csv', header=None)

y = to_categorical(mist[0])

X = mist.drop(columns=[0], axis=1).as_matrix()

model = Sequential()

model.add(Dense(50, activation='relu', input_shape=(784,)))

model.add(Dense(50, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, validation_split=0.3, epochs=10)