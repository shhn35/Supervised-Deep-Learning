import numpy as np
import CostFunctions as cf

def compute_cost(AL,Y,cost_function_name):
    """
    Calculates the cost for all training eaxamples

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector 

    Returns:
    cost -- cross-entropy cost
    dAL -- derevative of L(Al,Y) over AL 
    """
    cost_functions = {
        "cost_func_1": cf.cost_function_1
    }    

    activ_func = cost_functions.get(cost_function_name,lambda : "Invalid Cost Function Name !")

    cost,dAL = activ_func(AL,Y)

    return cost, dAL
