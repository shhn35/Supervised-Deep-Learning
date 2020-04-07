import numpy as np
import ActivataionFeedForward as aff

def L_model_FeedForward(X,Parameters,Actionation_Functions):
    """
    # inputs:
        X is a matrix of (n,m), where m is the number of training samples, 
        whereas n is the number of features

        Parameters is a dictionary of W[l] , b[l]

        Activation_Functions is np array of shape (1,L), 
        which indicates the activation function for each hidden layer
    """
    L = Actionation_Functions.shape[1]
    
    # To save all Z[l] and A[l-1]
    forward_caches = []

    A_prev = X
    AL = 0
    # calculate outputs and caches for all layers 
    for l in range(1,L+1):
        W = Parameters["W" + str(l)]
        b = Parameters["b" + str(l)]
        activation_func = Actionation_Functions[1,l-1]

        A,forward_cache = aff.activation_forward(A_prev,W,b,activation_func)
        
        A_prev = A
        forward_caches.append(forward_cache)

        if l == L:
            AL=A_prev
    
    assert(AL.shape == (1,X.shape[1]))

    return AL, forward_caches            
    




    