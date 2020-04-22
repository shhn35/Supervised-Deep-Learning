import numpy as np

def hyper_params_init():

    cost_func_name = "cost_func_1"
    """
        Choose one of the following cost functions
        cost_func_1: L(O,Y) = - (Y.log(O) + (1-Y).log(1-O))
    """

    learning_rate = 0.0075
    """
        Specify the learning rate for learning algorithm between [0 1]
    """

    max_epoch = 3000
    """
        Determine the max of learning iteration over all training samples
    """

    L = 4    
    """
        Indicate number of active layers in your DNN
    """

    layer_units = np.array([[20, 7, 5, 1]])
    assert(layer_units.shape == (1,L))
    """
        Define the number of active units (neurons) in each layer of DNN
        layer_units = [n1,n2,n3,...,nL]

        layer_units[0] indicates the number of neurons in the first active layer of DNN
    """

    activation_functions = np.array([["relu","relu","relu","sigmoid"]])
    assert(activation_functions.shape == (1,L))
    """
        Define the activation function of each layers of DNN
        layer_units = [af1,af2,af3,...,afL]

        activation_functions[0] indicates the activation function of the first layer of DNN, 
        in this case, it is Relu functions

        Available functions:
        1- 'relu': relu(z) = max(0,z)       if z>0
                             0              o.w
        2- 'sigmoid': sigmoid(z) = 1 / (1+exp(-z))
        3- 'tanh': tahh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
        4- 'leaky_relu': leaky_relu(z) = max(0.01*z , z)
    """

    return cost_func_name,learning_rate,max_epoch,L,layer_units,activation_functions
    