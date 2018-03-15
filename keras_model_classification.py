# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:34:27 2018

@author: junch
"""

#for classification, set the loss function to 'categorical_crossentropy'
#there are other loss functions
#1. Add metrics = ['accuracy'] to print out accuracy score at end of each epoch
#2. Modify output layer to have separate node for each possible outcome
#   using 'softmax' activation


#import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense


data = pd.read_csv('titanic_all_numeric.csv')

predictors = data.drop(['survived'], axis=1).as_matrix()
#as_matrix() turns dataframe into an ndarrary object

n_cols = predictors.shape[1]

target = to_categorical(data.survived)

model = Sequential()

model.add(Dense(100, activation='relu', input_shape=(n_cols,)))

model.add(Dense(100, activation='relu'))

model.add(Dense(100, activation='relu'))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',\
              metrics=['accuracy'])

model.fit(predictors, target, epochs=10)

# =============================================================================
# #save a model
# =============================================================================
from keras.models import load_model

model.save('my_model_file.h5')

# =============================================================================
# #reload model
# =============================================================================
my_model = load_model('my_model_file.h5')

predictions = my_model.predict(predictors)

probability_true = predictions[:,1]

#to check the model structure use
my_model.summary()

print("\nPrediction Probabilities")
print(probability_true)

# =============================================================================
# #Using a different optimizaer: 'sgd' Stochastic Gradient Descent
# =============================================================================
model = Sequential()

model.add(Dense(100, activation='relu', input_shape=(n_cols,)))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='sgd', loss='categorical_crossentropy',\
              metrics=['accuracy'])

model.fit(predictors, target, epochs=10, validation_split=0.3)
#split 30 to use as validation

predictions = model.predict(predictors)

predicted_prob_true = predictions[:,1]

print(predicted_prob_true)