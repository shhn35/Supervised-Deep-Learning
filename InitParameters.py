import numpy as np

def init_parametsr(layer_units,L):
    parameters = {}

    for l in range(1,L+1):
        w_l = np.random.randn(layer_units[0,l],layer_units[0,l-1]) * 0.01
        b_l = np.zeros((layer_units[0,l],1))

        parameters["W" + str(l)] = w_l
        parameters["b" + str(l)] = b_l

    return parameters        
        
    