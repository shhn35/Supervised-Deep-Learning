import numpy as np

### Define activation functions
def relu(z):
    A = np.maximum(0,z)
    assert(A.shape == z.shape)
    
    return A

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def leaky_relu(z):
    return np.where(z > 0, z, 0.01)

### Define derivatives of above functions

def d_relu(z):
    dz = np.where(z > 0, 1, 0)
    return dz

def d_sigmoid(z):
    a = sigmoid(z)
    dz = np.multiply(a,(1-a))
    return dz

def d_tanh(z):
    a = tanh(z)
    dz = 1 - np.multiply(a,a)
    return dz

def d_leaky_relu(z):
    dz = np.where(z < 0, 0.01, 1)

        