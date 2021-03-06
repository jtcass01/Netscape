import h5py
import math
import numpy as np

from FileSystem import FileSystem


def process_x(x):
    return x/255.

def process_y(y, number_of_targets):
    return convert_to_one_hot(y, number_of_targets).T

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Created By: Jacob Taylor Cassady
    Last Updated: 2/8/2018
    Objective: Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def log_model(model, accuracy, loss, model_type, model_alias = "test_model"):
    # Save the model to JSON
    json_model = model.to_json()
    with open(os.getcwd() + os.path.sep + "models" + os.path.sep + model_type + os.path.sep + model_alias + ".json", "w") as json_file:
        json_file.write(json_model)

    # Save weights
    model.save_weights(os.getcwd() + os.path.sep + "models" + os.path.sep + model_type + os.path.sep + model_alias + ".h5")
    print("Saved model " + model_alias + " to disk")

    # Save loss and accuracy
    FileSystem.start_log(str(loss), os.getcwd() + os.path.sep + "models" + os.path.sep + model_type + os.path.sep + model_alias + "_evaluation.txt")
    FileSystem.log(str(accuracy), os.getcwd() + os.path.sep + "models" + os.path.sep + model_type + os.path.sep + model_alias + "_evaluation.txt")

    # Save graphical model summary and print summary to console.
    print(model.summary())
#        plot_model(self.model, to_file= os.getcwd() + os.path.sep + "models" + os.path.sep + "ResNet50" + os.path.sep + model + ".png", show_shapes=True, show_layer_names=True)