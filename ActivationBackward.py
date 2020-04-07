import numpy as np
import ActivationFunctions as af

def activation_backward(W,dA,forward_cache,activation_func_name):
    (Z,A_prev) = forward_cache
    m = A_prev.shape[1]

    derivative_Z = call_d_function(activation_func_name,Z)
    
    dZ = np.multiply(dA,derivative_Z)


    dW = np.dot(dZ,A_prev.T) / m
    db = np.sum(dZ,axis=1,keepdims=True) / m

    dA_prev = np.dot(W.T,dZ)

    return dW, db, dA_prev



# calls the derivative of the related function to the activation_function and
# returns the value (dA)
def call_d_function(function_name,z):
    d_activation_functions = {
        "relu" : af.d_relu,
        "sigmoid" : af.d_sigmoid,
        "tanh" : af.d_tanh,
        "leaky_relu" : af.d_leaky_relu
    }

    func = d_activation_functions.get(function_name,lambda : "Invalid function name")

    return func(z)

