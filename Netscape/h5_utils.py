from data_utils import convert_to_one_hot
import h5py
import numpy as np

def load_dataset(relative_directory_path, train_data='train_signs.h5', test_data='test_signs.h5'):
    """
    Created By: Jacob Taylor Cassady
    Objective: Load in dataset.
    
    Arguments: 
    ...
    
    Returns: 
    train_set_x_orig -- A NumPy array of (currently) 1080 training images of shape (64,64,3).  Total nparray shape of (1080,64,64,3)
    train_set_y_orig -- A NumPy array of (currently) 1080 training targets.  Total nparray shape of (1, 1080) [After reshape]
    test_set_x_orig -- A NumPy array of (currently) 120 test images of shape (64,64,3).  Total nparray shape of (120,64,64,3)
    test_set_y_orig -- A NumPy array of (currently) 120 test targets.  Total nparray shape of (1,120) [After reshape]
    classes -- A NumPy array of (currently) 6 classes. (0-5)
    """
    train_dataset = h5py.File(relative_directory_path + train_data, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(relative_directory_path + test_data, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def load_and_process_data_set(relative_directory_path, number_of_targets):
    print("\nLoading data from relative directory path:", relative_directory_path)
    
    # load data set
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset(relative_directory_path)

    # process data set
    # Normalize image vectors
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.

    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, number_of_targets).T
    Y_test = convert_to_one_hot(Y_test_orig, number_of_targets).T

    print ("\tnumber of training examples = " + str(X_train.shape[0]))
    print ("\tnumber of test examples = " + str(X_test.shape[0]))
    print ("\tX_train shape: " + str(X_train.shape))
    print ("\tY_train shape: " + str(Y_train.shape))
    print ("\tX_test shape: " + str(X_test.shape))
    print ("\tY_test shape: " + str(Y_test.shape))

    return X_train, Y_train, X_test, Y_test, classes
