import numpy as np


"""
Implement the all cost functions and corresponding derevitives

Arguments:
AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
Y -- true "label" vector 

Returns:
cost -- cross-entropy cost
dAL -- derevative of L(Al,Y) over AL 
"""
def cost_function_1(AL,Y):
    # cost_func_1: L(O,Y) = - (Y.log(O) + (1-Y).log(1-O))

    m = Y.shape[1]

    # Compute loss from aL and y for all training examples
    cost =(np.dot(Y,np.log(AL).T)+np.dot((1-Y),np.log(1-AL).T)) / (-m)
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())

    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    return cost,dAL    