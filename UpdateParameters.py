import numpy as np

def update_params(Parameters,grades,learning_rate):
    """
    This function updats the parameters based on derivatives of parameters
    
    Arguments:
        Parameters, is the current parameters incuding W and b for all layers

        grades, is the corresponding derivatives for all parameters incuding dW and db for all layers

        learning_rate, is the learing rate, in which how far changes should be applied to parameters

    Output:
        updated_parameters, which indicates new values for all parameters based on Gradient Decent algorithm
    """

    L = len(Parameters) // 2

    for l in reversed(range(1,L+1)):
        W = Parameters["W" + str(l)]
        b = Parameters["b" + str(l)]

        dW = grades["dW" + str(l)]
        db = grades["db" + str(l)]

        # updating parameters' value based on their own derivatives
        W = W - learning_rate * dW
        b = b - learning_rate * db

        W = np.where(W>1 , 1 , W)
        W = np.where(W<-1 , -1 , W)
        
        Parameters["W" + str(l)] = W
        Parameters["b" + str(l)] = b

    return Parameters
        