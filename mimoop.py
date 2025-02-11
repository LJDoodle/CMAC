import math
import numpy as np
import random
import pandas as pd
import torch
import time

class CMAC:
    def __init__(self,weights,input_density,generalization,epoch,train_percent,data,learning_rate=0.01,in_dim=2): # STEP 0: INITIALIZE    
        self.weights = weights #the size of the weight table
        self.lr = learning_rate #How much the weights are adjusted for each training example
        self.rho = input_density #quantization resolution of input vector (number of bins)
        self.g = generalization #generalization parameter, how much each training datapoint effects its neighbors
        self.epoch = epoch #number of generations/epochs
        self.train_percent = train_percent #The percentage of datapoints reserved for training
        self.in_dim = in_dim #number of input dimensions, must be greater than one (FOR FUTURE REFERENCE, TRY TO GET in_dim=1 TO WORK)
        self.data = torch.from_numpy(data)

        self.num_data = self.data.size(0) #number of datapoints
        self.out_dim = self.data.size(1) - self.in_dim #number of output dimensions (total dimensions - input dimensions)

        self.xmax = self.data.max(dim=0) #takes the maximum datapoint for each input dimension (i.e. each column)
        self.xmin = self.data.min(dim=0) # ...  ... minimum ...

    def quantize(self): #STEP 1: QUANTIZE
        # Eqs. (1–2) are applied to quantize the components of the input patterns.
        # Eq 1.) Sij = [(N_i/(x_i^max - x_i^min))x(x_ij - x_i^min)] - 1
        # Eq 2.) If Sij < 0, --> Sij = 0
        self.index = torch.zeros((self.num_data,self.in_dim,))
        for i in range(self.in_dim):
            print('i: ',i)
            for j in range(self.num_data):
                #print(f'j: {j}/{D_total-1}')
                #print(data[j][i])
                self.index[j][i] = math.max((math.ceil(((self.rho/(self.xmax[i] - self.xmin[i]))*(self.data[j][i] - self.xmin[i])) - 1)),0) #Eq. 1
                #print(index)
        print('Final Index: ',self.index)
        




# FUNCT DEFINITIONS
'''
    learn
    inputs:
    - w_table: the weight table, an 2-D array of weights
    - a_table: the adress table, a 1-D array
    - dif: the differences between the predected and actual outputs, a 1-D array
    - rate: the learning rate, defaults to 0.1
    - g: the generalization factor, defaults to 10
    function: applies 'w(h+1) = w(h) + (beta*(y_real - y_predicted)/g)' to update the weights
'''
def learn(w_table, a_table, dif, rate=.1, g=10):
    weights = w_table.shape[0]
    for d in enumerate(w_table):
        for dim in range(w_table.shape[1]):
            if (a_table[d[0] % weights] == 1.0):
                w_table[d[0] % weights, dim] += ((rate * dif[dim]) / g)     

'''
    temp_matrix
    inputs:
    - dim: the number of input dimensions
    - g: the generalization factor
    - index: a matrix
    - rho: the input density
    - rand_table: the random table
    function: creates a temporary matrix of the wokring data and shifts it,
    meant for before passing to the association matrix
'''
def temp_matrix(dim,g,index,rho,rand_table):
  k = np.zeros((dim,g))
  for i in range(dim): #i is the current dimension index
    new_index = int(index[i][j])
    if new_index >= rho:
      new_index = rho - 1
      #print('New Index: ',new_index)
    k[i] = rand_table[i,new_index:new_index+g].astype(int)
    #print(k)
    shift_factor = new_index % g
    #print(shift_factor)
    k[i, :] = np.roll(k[i, :], shift_factor)
    #print('k, ',k)
  #print('k, ',k)
  return k

def address_matrix(weights, g):
  address_table = np.zeros((weights))
  #print('Address Table: ', address_table)
  for i in range(g):
      temp_address = k[:,i]
      #print('Temp Address: ',temp_address)
      last = temp_address[0]
      for item in temp_address[1:]:
        #print('item - ',item,', last - ',last)
        xor = int(item) ^ int(last)
        last = int(item)
        #print('xor - ',xor)
      address_table[xor]=1
      #print('Final Address Table: ',address_table)
  return address_table



#--Import Data--
df = pd.read_csv('tvdata.csv') #Replace 'tvdata.csv' with the address of the csv file containing your training and validation data
df = df.sample(frac = 1)
data = df.values

#--Based on the Data--
D_train = 200 #D_train is the number of training datapoints
D_total = df.shape[0]
dim_tot = df.shape[1] #total number of dimensions
out_dim = dim_tot - in_dim #number of output dimensions

# STEP 2: CREATE RANDOM TABLES

#The size of each random table i is given by Eq. 3
# Eq 3.) Ci (size of the random table of input vector x_i) = rho + g − 1, i = 1,2,...,n
rand_table_size = rho + g - 1

#each random table i consists of uniform random numbers that are generated from the interval [0,k/2]
rand_table = np.random.uniform(0,weights/2,[in_dim,rand_table_size])
rand_table = np.ceil(rand_table)
print(f"Random Tables: {rand_table}")
weight_table = np.full((weights,out_dim),w0)
print('Weight Table: ', weight_table.T)
a_matrix = weight_table
print('Association Matrix: ', a_matrix.T)

start = time.time()

for h in range(epoch):
    for j in range(D_train-1):
        k = temp_matrix(in_dim,g,index,rho,rand_table)
        address_table = address_matrix(weights,g)

        #STEP 4: CALCULATE ACTUAL OUTPUT
        output = weight_table.T @ address_table
        difference = data[j,in_dim:] - output

        #STEP 5: LEARNING ALGORITHM
        learn(weight_table, address_table, difference, learning_rate, g)
    #STEP 6: EVALUATE CLASSIFICATION ERROR
#print('Weight Table: ', weight_table.T)
end = time.time()

# STEP 7: MEASURE CLASSIFICATION ERROR

print('-TESTING-')

neterror = 0

for j in range(D_train,D_total):
    #print("Mass: ", data[0,j],"\nRadius: ",data[1,j],"\nAngular Acceleration: ",data[2,j],"\nExpected Result: ",data[3,j])
    k = np.zeros((in_dim,g))
    for i in range(in_dim):
        new_index = int(index[i][j])
        if new_index >= rho:
            new_index = rho-1
        k[i] = rand_table[i,new_index:new_index+g].astype(int)
        shift_factor = new_index % g
        k[i, :] = np.roll(k[i, :], shift_factor)
    address_table = np.zeros(weights)
    #print('Address Table: ', address_table)
    for i in range(g):
        temp_address = k[:,i]
        #print(temp_address)
        last = temp_address[0]
        for item in temp_address[1:]:
            xor = int(item) ^ int(last)
            last = int(item)
        #print(xor)
        address_table[xor]=1
        #print('Final Address Table: ',address_table)

    #Step 4: calculate actual output
    output = weight_table.T @ address_table
    difference = data[j,in_dim:] - output

    #Update these print statements to show the data you are interested in observing
    #print(data[j,:],'\t',output) #input, expected output(s), prediction(s)
    print(data[j,0],'\t',data[j,1],'\t',data[j, in_dim],'\t',data[j, in_dim+1],'\t',output[0],'\t',output[1])
    neterror += np.sqrt(np.sum((np.square(difference))))

avg_error = neterror / (D_total - D_train)
print(f"rho:\t{rho}\nweights:\t{weights}")
print('Average Error:\t',avg_error)
print("Training Time:\t", end - start)
