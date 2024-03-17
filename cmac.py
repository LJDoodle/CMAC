import math
import numpy as np
import random
import pandas as pd

#STEP 0: INITIALIZE

# Parameters
weights = 160 #the size of the weight table
initial_weights = 0.0 #The initial value of each each
learning_rate = 0.1
rho = 10 #quantization resolution of input vector x_i = number of bins (N_i)
g = 3 #generalization size

in_dim = 2 #number of input dimensions
out_dim = 1
dim_tot = in_dim + out_dim #total number of dimensions

fill_value = 1000 #This number is used to set the initial minimum value for the indexing function.
epoch = 50
D_train = 100         #D_train is the number of training datapoints

'''def generate_data():
    x = np.random.rand(in_dim)
    y = x[0]/x[1]
    y = math.sin(y)
    x = np.append(x,y)
    return x'''

#Data
df = pd.read_csv('/content/CMAC sinx.csv')
df = df.sample(frac = 1)
data = df.values

D_total = df.shape[0]

#STEP 1: QUANTIZE

# Determines the maximum and minimum values for each dimension
xmax=np.zeros(in_dim)
xmin=np.full(in_dim, fill_value)
for datum in data:
    for dim in range(in_dim):
        if datum[dim] > xmax[dim]:
            xmax[dim] = datum[dim]
        if datum[dim] < xmin[dim]:
            xmin[dim] = datum[dim]
print(xmax)
print(xmin)

# Eqs. (1–2) are applied to quantize the components of the input patterns.
# Eq 1.) Sij = [(N_i/(x_i^max - x_i^min))x(x_ij - x_i^min)] - 1
# Eq 2.) If Sij < 0, --> Sij = 0

index = np.zeros((in_dim, D_total))
for i in range(in_dim):
    print('i: ',i)
    for j in range(D_total):
        #print(f'j: {j}/{D_total-1}')
        #print(data[j][i])
        index[i][j] = math.ceil(((rho/(xmax[i] - xmin[i]))*(data[j][i] - xmin[i])) - 1) #Eq. 1
        if index[i][j] < 0 or index[i][j] == 'NaN':
            index[i][j] = 0 #Eq. 2
        #print(index)
print('Final Index: ',index)

#STEP 2: CREATE RANDOM TABLES

#The size of each random table i is given by Eq. 3
# Eq 3.) Ci (size of the random table of input vector x_i) = rho + g − 1, i = 1,2,...,n
# edere
rand_table_size = rho + g - 1

#each random table i consists of uniform random numbers that are generated from the interval [0,k/2]

rand_table = np.random.uniform(0,weights/2,[in_dim,rand_table_size]) # A 2x3 array
rand_table = np.ceil(rand_table)
print(f"Random Tables: {rand_table}")
weight_table = np.full((weights,out_dim),initial_weights)
print('Weight Table: ', weight_table.T)
a_matrix = weight_table
print('Association Matrix: ', a_matrix.T)

#STEPS 3-5

h = 0
while h < epoch:
    for j in range(D_train):
        k = np.zeros((in_dim,g))
        for i in range(in_dim): #i is the current dimension index
            new_index = int(index[i][j])
            if new_index >= 10:
                new_index = 9
            #print('New Index: ',new_index)
            k[i] = rand_table[i,new_index:new_index+g].astype(int)
            #print(k)
            shift_factor = new_index % g
            #print(shift_factor)
            k[i, :] = np.roll(k[i, :], shift_factor)
            #print('k, ',k)
        #print('k, ',k)
        address_table = np.zeros((weights,out_dim))
        #print('Address Table: ', address_table)
        for i in range(g):
            temp_address = k[:,i]
            #print('Temp Address: ',temp_address)
            last = temp_address[0]
            if in_dim == 1:
                xor = int(last)
            else:
                for item in temp_address[1:]:
                    #print('item - ',item,', last - ',last)
                    xor = int(item) ^ int(last)
                    last = int(item)
                    #print('xor - ',xor)
            address_table[xor]=1
            #print('Final Address Table: ',address_table)

            #Step 4: calculate actual output
            output = weight_table.T @ address_table
            #print('Output: ',output)
            difference = data[j,in_dim] - output
            #print('Difference: ',difference)

            #Step 5: learning algorithm
            # w(h+1) = w(h) + (beta*(y_real - y_predicted)/g)
            for d in enumerate(weight_table):
                if (address_table[d[0] % weights] == 1.0):
                    weight_table[d[0] % weights, math.floor(d[0]/weights)] += ((learning_rate * difference) / g)
        #print('Weight Table (',i,',',j,'): ', weight_table)
    #endFor
    #Step 6: evaluate classification error
    h += 1
#endWhile

print('Weight Table: ', weight_table.T)

#STEP 7: MEASURE CLASSIFICATION ERROR

print('-TESTING-')

neterror = 0

for j in range(D_train,D_total):
    #print("Mass: ", data[0,j],"\nRadius: ",data[1,j],"\nAngular Acceleration: ",data[2,j],"\nExpected Result: ",data[3,j])
    k = np.zeros((in_dim,g))
    for i in range(in_dim):
        new_index = int(index[i][j])
        if new_index >= 10:
            new_index = 9
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
    print(data[j,0],'\t',data[j, in_dim:],'\t',output[0]) #input, expected output(s), prediction(s)
    neterror += np.sum(math.sqrt(difference[0] ** 2))

avg_error = neterror / (D_total - D_train)
print(f"rho:\t{rho}\nweights:\t{weights}")
print('Average Error:\t',avg_error)
#print('Test Outputs (',j,'): ',output, ', ',data[dim_tot-1][j],', ',difference)
