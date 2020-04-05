import numpy as np
import ActivationFunctions as af


# This function produces the output of the current layer and 
# returns the output and the chach values of (Z,A_prev)
def activation_forward(A_prev,W,b,activation_function):
    Z = np.dot(W,A_prev) + b
    assert (Z.shape == (W.shape[0], A_prev.shape[1]))

    A = call_function(activation_function)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    forward_cache = (Z, A_prev)

    # forward_cache is a list of (Z , A_prev)
    # A is the output vector of current layer
    return A, forward_cache

# calls the related function to the activation_function and
# returns the value
def call_function(function_name,z):
    activation_functions = {
        "relu" : af.relu,
        "sigmoid" : af.sigmoid,
        "tanh" : af.tanh,
        "leaky_relu" : af.leaky_relu
    }

    func = activation_functions.get(function_name,lambda : "Invalid function name")

    return func(z)