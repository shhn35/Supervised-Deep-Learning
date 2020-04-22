import numpy as np
import matplotlib.pyplot as plt

import HyperParamsInitSetting as hpis
import LoadDataset as lds
import InitParameters as ip
import ModelFeedForward as ff
import ModelBackward as bp
import UpdateParameters as up

def startDNN():

    # Get hyper paremeters settings for both DNN structures as well as learning settings
    cost_func_name,learning_rate,max_epoch,L,layer_units,activation_functions = hpis.hyper_params_init()

    # Load the whole data set 
    train_X,train_Y,test_X,test_Y = lds.load_dataset("catvnoncat")


    # Start leraning process
    parameters = supervised_DNN(cost_func_name,learning_rate,max_epoch,L,layer_units,activation_functions,train_X,train_Y,test_X,test_Y)


    # predict the train dataset in order to get train accuracy
    train_accuracy = predict(train_X,train_Y,parameters,activation_functions)
    print("Final Train accuracy: %s" % str(train_accuracy))

    # predict the test dataset in order to get train accuracy
    test_accuracy = predict(test_X,test_Y,parameters,activation_functions)
    print("Final Test accuracy: %s" % str(test_accuracy))

    
    print("Finished")


def supervised_DNN(cost_func_name,learning_rate,max_epoch,L,layer_units,activation_functions,train_X,train_Y,test_X,test_Y ):
    np.random.seed(1)

    # insert the number of features of X into layer_units
    layer_units = np.insert(layer_units,0,train_X.shape[0],axis=1) 


    # Initialize network's parameters for all layers
    parameters = ip.init_parametsr(layer_units,L)

    training_cost = []
    epoch_counter = []
    train_accuracy = []
    test_accuracy = []
    # Start learning process
    for e in range(1,max_epoch):
        AL, forward_caches = ff.L_model_FeedForward(train_X, parameters,activation_functions)

        grades, cost = bp.L_model_backward(AL, train_Y, parameters, forward_caches, activation_functions, cost_func_name)

        parameters = up.update_params(parameters, grades, learning_rate)

        if (e % 100 == 0):
            training_cost.append(cost)
            epoch_counter.append(e)

            print("epoch:" + str(e))
            print("-> cost:" + str(cost))
            # predict the train dataset in order to get train accuracy
            train_acc = predict(train_X,train_Y,parameters,activation_functions)
            train_accuracy.append(train_acc)
            print("Train accuracy: %s" % str(train_acc))


            # predict the test dataset in order to get train accuracy
            test_acc = predict(test_X,test_Y,parameters,activation_functions)
            test_accuracy.append(test_acc)
            print("Test accuracy: %s" % str(test_acc))
            print("_______________________")

    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.plot(np.array(epoch_counter),np.array(training_cost),'r')
    ax1.legend(['Learning cost'])
    ax1.set_title('Cost')
    ax1.set(xlabel='epoch',ylabel='cost')
    ax1.label_outer()

    ax2.plot(np.array(epoch_counter),np.array(train_accuracy),'b',np.array(epoch_counter),np.array(test_accuracy),'-g')
    ax2.legend(['Train accuracy (%)','Test accuracy (%)'])
    ax2.set_title('Accuracy')
    ax2.set(xlabel='epoch',ylabel='Accuracy in percent')
    # plt.plot(np.array(epoch_counter),np.array(training_cost),'r',np.array(epoch_counter),np.array(train_accuracy),'b',np.array(epoch_counter),np.array(test_accuracy),'-g')
    # plt.legend(['Learning cost','Train accuracy (%)','Test accuracy (%)'])
    plt.show()
    return parameters    





def predict(X,Y,parameters,activation_functions):
    # This function predict a binary clasification
    output, forward_caches = ff.L_model_FeedForward(X, parameters,activation_functions)
    output_binary = np.where(output >= 0.5, 1, 0)

    predict_binary_result = np.where(output_binary == Y, 1, 0)
    accuracy = np.sum(predict_binary_result) / Y.shape[1]

    return accuracy


if __name__ == "__main__":
    startDNN()