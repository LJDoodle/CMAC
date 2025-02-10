import math
import numpy as np
import random
import pandas as pd
import tensorflow as tf
import time

class CMAC:
    def __init__(self,weights,input_density,generalization,epoch,train_percent,learning_rate=0.01,in_dim=2):
        self.weights = weights #the size of the weight table
        self.lr = learning_rate #How much the weights are adjusted for each training example
        self.rho = input_density #quantization resolution of input vector (number of bins)
        self.g = generalization #generalization parameter, how much each training datapoint effects its neighbors
        self.epoch = epoch #number of generations/epochs
        self.train_percent = train_percent #The percentage of datapoints reserved for training
        self.in_dim = in_dim #number of input dimensions, must be greater than one (FOR FUTURE REFERENCE, TRY TO GET in_dim=1 TO WORK)