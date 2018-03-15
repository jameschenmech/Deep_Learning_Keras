# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 10:26:48 2018

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


def get_new_model(input_shape):
    
    model = Sequential()
    
    model.add(Dense(100, activation='relu', input_shape=(input_shape,)))
    
    model.add(Dense(100, activation='relu'))
    
    model.add(Dense(2, activation='softmax'))
    
    return model


#Create list of learning rates: lr_to_test
lr_to_test = [0.000001, 0.01, 1]

#Loop over learning rates

for lr in lr_to_test:

    print('\n\nTesting model with learning rate: %f\n'%lr)
    
    #Build new model to test, unaffected by previous models
    model=get_new_model(n_cols)
    
    #Create SGD optimizer with specified learning rate:  my_optimizer
    my_optimizer = SGD(lr=lr)
    
    #Compile the model
    model.compile(optimizer=my_optimizer, loss='categorical_crossentropy')
    
    #Fit the model
    model.fit(predictors, target, epochs=10)
    
#k-fold cross validation typically not run on deep learning models
#data sets are too large
#Usually trust the score from a single validation run because of the
#large size of the validation

# =============================================================================
# #Early stopping
#    patience = how many epochs the model can go without improving before
#               stopping training, in practice usually 2 or 3
# =============================================================================
data = pd.read_csv('titanic_all_numeric.csv')

predictors = data.drop(['survived'], axis=1).as_matrix()
#as_matrix() turns dataframe into an ndarrary object

n_cols = predictors.shape[1]

target = to_categorical(data.survived)


model = get_new_model(n_cols)

#Create SGD optimizer with specified learning rate:  my_optimizer
my_optimizer = SGD(lr=0.0001)

#Compile the model
model.compile(optimizer=my_optimizer, loss='categorical_crossentropy')

from keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(patience=2)

model.fit(predictors, target, validation_split=0.3, epochs=20,
          callbacks = [early_stopping_monitor])
#monitors the val_loss


# =============================================================================
# #Print the loss
# =============================================================================
import matplotlib.pyplot as plt

early_stopping_monitor = EarlyStopping(patience=10)

model2 = model.fit(predictors, target, validation_split=0.3, epochs=50,
          callbacks = [early_stopping_monitor])

plt.plot(model2.history['val_loss'],'r')
plt.xlabel('Epochs')
plt.ylabel('Validation Score')
plt.show()
#Consider changing activation function if the below happens:
#dying neuron problem:  once turns negative node continues being negative
#Vanishing gradients:  layers have very small slopes