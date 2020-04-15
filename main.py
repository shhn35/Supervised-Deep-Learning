import numpy as np
import matplotlib.pyplot as plt

import HyperParamsInitSetting as hpis
import LoadDataset as lds
import InitParameters as ip
import ModelFeedForward as ff
import ModelBackward as bp
import UpdateParameters as up


def supervised_DNN():
    # Get hyper paremeters settings for both DNN structures as well as learning settings
    cost_func_name,learning_rate,max_epoch,L,layer_units,activation_functions = hpis.hyper_params_init()

    # Load the whole data set 
    train_X,train_Y,test_X,test_Y = lds.load_dataset("catvnoncat")

    # insert the number of features of X into layer_units
    layer_units = np.insert(layer_units,0,train_X.shape[0],axis=1) 


    # Initialize network's parameters for all layers
    parameters = ip.init_parametsr(layer_units,L)

    training_cost = np.zeros((1,max_epoch))
    # Start learning process
    for e in range(max_epoch):
        AL, forward_caches = ff.L_model_FeedForward(train_X, parameters,activation_functions)

        grades, cost = bp.L_model_backward(AL, train_Y, parameters, forward_caches, activation_functions, cost_func_name)
        training_cost[0,l] = cost

        parameters = up.update_params(parameters, grades, learning_rate)

        if (e % 10 == 0):
            print("epoch:" + str(e) + "-> cost:" + str(cost))
        

    print("Finished")


if __name__ == "__main__":
    supervised_DNN()