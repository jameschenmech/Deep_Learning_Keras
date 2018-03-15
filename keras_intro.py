# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 09:36:08 2018

@author: junch
"""
# =============================================================================
# #Forward propagation with one hidden layer
# =============================================================================
import numpy as np

input_data = np.array([2,3])

weights = {'node_0':np.array([1,1]),
            'node_1':np.array([-1,1]),
            'output':np.array([2,-1])}

node_0_value = (input_data*weights['node_0']).sum()
node_1_value = (input_data*weights['node_1']).sum()

hidden_layer_values = np.array([node_0_value, node_1_value])

print("\nFirst model deep learning")
print("hidden layer: ", hidden_layer_values)

output = (hidden_layer_values*weights['output']).sum()


print("output: ",output)

# =============================================================================
# #Another example of one hidden layer
# =============================================================================
input_data = np.array([3, 5])

weights = {'node_0':np.array([2,4]),
            'node_1':np.array([4,-5]),
            'output':np.array([2,7])}

node_0_value = (input_data*weights['node_0']).sum()
node_1_value = (input_data*weights['node_1']).sum()

hidden_layer_values = np.array([node_0_value, node_1_value])

print("\nSecond model deep learning")
print("hidden layer: ", hidden_layer_values)

output = (hidden_layer_values*weights['output']).sum()


print("output: ",output)

# =============================================================================
# #Activation functions
#captures non-linearities inthe hidden layers
#applied coming into the node
#relu function
#tanh function was popular previously
# =============================================================================
input_data = np.array([-1,2])

weights = {'node_0':np.array([3,3]),
            'node_1':np.array([1,5]),
            'output':np.array([2,-1])}

#activation function applied to the node input
node_0_input = (input_data*weights['node_0']).sum()
node_0_output = np.tanh(node_0_input)

#activation function applied to the node input
node_1_input = (input_data*weights['node_1']).sum()
node_1_output = np.tanh(node_1_input)

#new hidden layer values
hidden_layer_outputs = np.array([node_0_output, node_1_output])

print("\nLearning model with activation tanh function")
print("hidden layer outputs: ", hidden_layer_outputs)

output = (hidden_layer_outputs*weights['output']).sum()

print("output: ",output)

# =============================================================================
# #Using the RELU activation function
# =============================================================================
def relu(input):
    '''Define relu activation function'''
    output = max(input,0)
    return output

input_data = np.array([3,5])

weights = {'node_0':np.array([2,4]),
            'node_1':np.array([4,-5]),
            'output':np.array([2,7])}

#activation function applied to the node input
node_0_input = (input_data*weights['node_0']).sum()
node_0_output = relu(node_0_input)

#activation function applied to the node input
node_1_input = (input_data*weights['node_1']).sum()
node_1_output = relu(node_1_input)

#new hidden layer values
hidden_layer_outputs = np.array([node_0_output, node_1_output])

print("\nLearning model with activation RELU function")
print("hidden layer outputs: ", hidden_layer_outputs)

final_input_layer = (hidden_layer_outputs*weights['output']).sum()
output = relu(final_input_layer)

print("output: ",output)

# =============================================================================
# #Apply to many observations/rows of data
# =============================================================================

input_data = [np.array([3,5]), np.array([1,-1]),np.array([0,0]),\
              np.array([8,4])]

weights = {'node_0':np.array([2,4]),
            'node_1':np.array([4,-5]),
            'output':np.array([2,7])}

#define predict with network

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

#Create empty list to store prediction results
results=[]

for input_data_row in input_data:
    results.append(predict_with_network(input_data_row, weights))
    
print("\nresults for list of different inputs:")
print(results)

# =============================================================================
# #Deeper Networks
# =============================================================================
def predict_with_network_2d(input_data):
    
    #activation function applied to the node input
    node_0_0_input = (input_data*weights['node_0_0']).sum()
    node_0_0_output = relu(node_0_0_input)

    #activation function applied to the node input
    node_0_1_input = (input_data*weights['node_0_1']).sum()
    node_0_1_output = relu(node_0_1_input)

    #new hidden layer values
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])

    #activation function applied to the node input
    node_1_0_input = (hidden_0_outputs*weights['node_1_0']).sum()
    node_1_0_output = relu(node_1_0_input)

    #activation function applied to the node input
    node_1_1_input = (hidden_0_outputs*weights['node_1_1']).sum()
    node_1_1_output = relu(node_1_1_input)

    #new hidden layer values
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])

    input_to_final_layer = np.array(hidden_1_outputs*weights['output']).sum()
    model_output = relu(input_to_final_layer)
        
    return model_output

weights = {'node_0_0':np.array([2,4]),
           'node_0_1':np.array([4,-5]),
           'node_1_0':np.array([-1,2]),
           'node_1_1':np.array([1,2]),
           'output':np.array([2,7])}


input_data = np.array([3,5])

output = predict_with_network_2d(input_data)
print("\n2  layer networkd output:")
print(output)