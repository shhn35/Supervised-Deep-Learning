import numpy as np
import h5py



def load_dataset(dataset_name,flatten = True):
    """
    Loads dataset based on h5 file, which need to be located in 'Datasets' directory. 
    Note: For your own dataset, devide it into 'train_' and 'test_' followed by your dataset name.
    
    Arguments:
    dataset_filename contains the filename of your custom dataset without 'train_' nor 'test_' nor '.h5'
    flatten indicates that the output matrixes be in whether flatten form or original form
    Outputs:
    train_X,train_Y,test_X,test_Y
    """
    dataset_directory_path = "Datasets\\" 
    test_filename = dataset_directory_path + "test_" + dataset_name + ".h5"
    train_filename =  dataset_directory_path + "train_" + dataset_name + ".h5"

    # Load train dataset
    print("Load Train dataset:")
    train_classes , train_X, train_Y = read_h5_file(train_filename)
    print("Train Dataset Classess %s" % train_classes)
    print("Train dataset original size (shape): %s" % str(train_X.shape))
    print("Train of test samples %s" % str(train_X.shape[0]))
    print("_________________")

    # Load test dataset
    print("Load Test dataset:")
    test_classes , test_X, test_Y = read_h5_file(test_filename)
    print("Test Dataset Classess %s" % test_classes)
    print("Test dataset original size (shape): %s" % str(test_X.shape))
    print("Number of test samples %s" % str(test_X.shape[0]))
    print("___________")

    if flatten:
        train_X_flatten = train_X.reshape(train_X.shape[0],-1).T
        test_X_flatten = test_X.reshape(test_X.shape[0],-1).T

        # standardize values between 0 and 1 for image

        train_X = train_X_flatten / 255
        test_X = test_X_flatten / 255 
        
        print("Final Train dataset size: %s " % str(train_X.shape))
        print("Final Test dataset size: %s " % str(test_X.shape))
        print("Number of features in X (input layer's nodes) %s " % str(train_X.shape[0]))

    return train_X,train_Y,test_X,test_Y
    
def read_h5_file(file_name):
    classes = list()
    X = list()
    Y = list()
    with (h5py.File(file_name,"r")) as f :
        # print("keys: %s :" % f.keys())

        classes_key = list(f.keys())[0]
        X_key = list(f.keys())[1]
        Y_key = list(f.keys())[2]

        classes = list(f[classes_key])
        X = list(f[X_key])
        Y = list(f[Y_key])

    return classes, np.array(X), np.array(Y).reshape((1,len(Y)))

### for testing the madule itself
if __name__ == "__main__":
    load_dataset("catvnoncat")