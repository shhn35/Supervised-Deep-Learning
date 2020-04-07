import numpy as np
import HyperParamsInitSetting as hpis
import LoadDataset as lds
import InitParameters as ip

def supervised_DNN():
    # Get hyper paremeters settings for both DNN structures as well as learning settings
    cost_func_name,learning_rate,max_epoch,L,layer_units,activation_functions = hpis.hyper_params_init()

    # Load the whole data set 
    X,Y = lds.load_dataset()

    layer_units = np.insert(layer_units,0,X.shape[0],axis=1) # insert th number of features of X into layer_units


    # Initialize network's parameters for all layers
    parameters = ip.init_parametsr(layer_units,L)

    print("Finished")


if __name__ == "__main__":
    supervised_DNN()