import numpy as np


def relu(z):
    if z < 0 :
        return 0
    else:
        return z

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def leaky_relu(z):
    if z < 0:
        return z * 0.01
    else:        
        return z

        