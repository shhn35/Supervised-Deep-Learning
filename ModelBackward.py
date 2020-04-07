import numpy as np
import ActivationBackward as ab
import ComputeCostFunction as ccf

def L_model_backward(AL,Y,parameters,forward_caches,activation_functions,cost_func_name):
    """
    # Arguments:
        Parameters is a dictionary of W[l] , b[l]

        forward_caches is a list of (Z, A_prev) corresponding to each layer 

        Activation_Functions is np array of shape (1,L), 
        which indicates the activation function for each hidden layer

    Output:
        A dictionary of dWs and dbs for all layers
    """

    L = activation_functions.shape[1]
    grades = {}

    # calcutlate the derivative of dL over dAL
    cost, dAL = ccf.compute_cost(Al, Y, cost_func_name)
    dA = dAL

    for l in reversed(range(1,L+1)):
        W = parameters["W"+str(l)]
        current_forward_cache = forward_caches[l-1]

        dW, db, dA_prev = ab.activation_backward(W,dA,current_forward_cache,activation_functions[1,l-1])

        dA = dA_prev

        grades["dW" + str(l)] = dW
        grades["db" + str(l)] = db

    return grades



