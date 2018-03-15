# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:47:08 2018

@author: junch
"""

# =============================================================================
# #Error = prediction - actual value
# #probably use mean square error
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

def predict_with_network(input_data_row, weights):
    
    #activation function applied to the node input
    node_0_input = (input_data_row*weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    #activation function applied to the node input
    node_1_input = (input_data_row*weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    #new hidden layer values
    hidden_layer_outputs = np.array([node_0_output, node_1_output])

 #   print("\nLearning model with activation RELU function")
 #   print("hidden layer outputs: ", hidden_layer_outputs)

    input_to_final_layer = (hidden_layer_outputs*weights['output']).sum()
    model_output = relu(input_to_final_layer)
  
#    print("output: ",model_output)
    
    return model_output


input_data = np.array([0,3])

#Sample Weights
weights_0 = {'node_0':[2,1],
             'node_1':[1,2],
             'output':[1,1]}

#target value
target_actual = 3

#make prediction
model_output_0 = predict_with_network(input_data, weights_0)

#Calculate error: error_0
error_0 = model_output_0 - target_actual


#Revised Weights
weights_1 = {'node_0':[2,1],
             'node_1':[1,2],
             'output':[-1,1]}

#make prediction
model_output_1 = predict_with_network(input_data, weights_1)

#Calculate error: error_0
error_1 = model_output_1 - target_actual


print("\nerrors:")
print(error_0)
print(error_1)


# =============================================================================
# #Scaling up to multiple data points
# =============================================================================
from sklearn.metrics import mean_squared_error

#create model_output_0
model_output_0 =[]
#create model_output_1
model_output_1 =[]

weights_0 = {'node_0':np.array([2,1]),
             'node_1':np.array([1,2]),
             'output':np.array([1,1])}

weights_1 = {'node_0':np.array([2,1]),
             'node_1':np.array([1.,1.5]),
             'output':np.array([1.,1.5])}

input_data = [np.array([0,3]), np.array([1,2]), np.array([-1,-2]),\
              np.array([4,0])]

target_actuals = [1,3,5,7]

#loop over input_data
for row in input_data:
    model_output_0.append(predict_with_network(row, weights_0))
    model_output_1.append(predict_with_network(row, weights_1))
    
#calculate msr
msr_0 = mean_squared_error(target_actuals, model_output_0)
msr_1 = mean_squared_error(target_actuals, model_output_1)    

print("\nMean Squared errors")
print("MSE 0: ",msr_0)
print("MSE 1: ",msr_1)

# =============================================================================
# #Gradient Descent 
#slope for a weight is the product of:
#1. slope of the loss function wrt value at the node we feed into: 2*error
#2. value of the node that feeds into the weight
#3. slope of the activation function wrt value we feed into
#Update the weight:  old_weight - Learning_rate*(slope of weight)
# =============================================================================

weights = np.array([1,2])

input_data = np.array([3,4])

target = 6

learning_rate = 0.01

preds = (weights * input_data).sum()

error = preds - target

print("\nerror: ", error)

gradient = 2* input_data* error

weights_updated = weights - learning_rate*gradient

preds_updated = (weights_updated*input_data).sum()

error_updated = preds_updated -  target

print("\nerror updated", error_updated)

# =============================================================================
# #Making multiple updates to weights
# =============================================================================
n_updates = 20
mse_hist =[]

input_data = np.array([1,2,3])

target = 0

weights = np.array([0,2,1])

def get_slope(input_data, target, weights):
    preds = input_data*weights
    error = preds - target
    return 2*input_data*error

def get_mse(input_data, target, weights):
    model_output = input_data*weights
    return np.square(target - model_output).sum()  

    
for i in range(n_updates):
    
    slope = get_slope(input_data, target, weights)
    
    weights = weights - 0.1*slope
 
    mse = get_mse(input_data, target, weights)
    
    mse_hist.append(mse)
    
plt.plot(mse_hist)
plt.xlabel('Iteractions')
plt.ylabel('Mean Squared Error')
plt.show()


# =============================================================================
# #Back propagation
#Do forward propagation first
# =============================================================================
