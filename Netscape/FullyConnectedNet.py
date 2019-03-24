import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pydot, graphviz
from IPython.display import SVG
import scipy.misc
import os

from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model, model_from_json
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot, plot_model
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

# User create modules
from h5_utils import load_dataset
from data_utils import convert_to_one_hot, process_x, process_y
from FileSystem import FileSystem

import sys

class FullyConnectedNet(object):
    def __init__(self, classes, input_shape = (64, 64, 3), inner_layer_layout=[25, 12]):
        self.input_shape = input_shape
        self.inner_layer_layout=[25, 12]
        self.classes = classes.reshape((classes.shape[0],1))
        tf.reset_default_graph()

    def __del__(self):
        del self.input_shape
        del self.inner_layer_layout
        del self.classes
        del self

    def __str__(self):
        return "FullyConnectedNet"

    def train_model(self, x_train, y_train, epochs = 2, batch_size = 32, optimizer = 'adam', loss_func = 'categorical_crossentropy', metric='accuracy'):
        # Initialize variables
        print("\nBuilding Fully Connected model with input_shape:", str(self.input_shape), "and classes", str(self.classes))
        
        # Build model
        model = FullyConnectedNet.build_model(input_shape=self.input_shape, number_of_classes=self.classes.shape[0], inner_layer_layout=self.inner_layer_layout)

        # Compile model
        print("\tCompiling model with the following parameters:")
        print("\t\tOptimizer [Flavor of Gradient Descent] : " + optimizer)
        print("\t\tLoss Function : " + loss_func)
        print("\t\tMetrics : " + metric)
        model.compile(optimizer=optimizer, loss=loss_func, metrics=[metric])

        # Process data
        x = process_x(x_train)
        y = process_y(y_train, classes.shape[0])

        # train model
        print("\nTraining model... for ", epochs, "epochs with a batch size of", batch_size)
        print("\tx_train shape: ", str(x.shape))
        print("\ty_train shape: ", str(y.shape))
        model.fit(x, y, epochs = epochs, batch_size = batch_size)

        # evaluate model
        accuracy, loss = model.evaluate(x_train, y_train)

        # Save model
        log_model(model, "FullyConnectedNet", "most_recent_model")


    def evaluate(self, x_test, y_test):
        x = process_x(x_test)
        y = process_y(y_test, self.classes.shape[0])
        print("\nEvaluating Model...")
        print("x_test shape: ", str(x.shape))
        print("y_test shape: ", str(y.shape))
        preds = self.model.evaluate(x, y, verbose=1)
        print ("\tLoss = " + str(preds[0]))
        print ("\tTest Accuracy = " + str(preds[1]))
        return preds[0], preds[1]

    def predict_image(self, image_path, target_size=(64, 64)):
        print("preparing to predict image:", image_path)
        img = image.load_img(image_path, target_size=target_size)

        if img is None:
            print("Unable to open image", image_path)
            return None

        pixels = image.img_to_array(img)
        pixels = np.expand_dims(pixels, axis=0)
        pixels = preprocess_input(pixels)

        print('Input image shape:', pixels.shape)

        for index, response in enumerate(self.model.predict(pixels)[0]):
            if response == 1:
                return index

    def save_model(self, model = 'test_model'):
        # Save the model to JSON
        json_model = self.model.to_json()
        with open(os.getcwd() + os.path.sep +  "models" + os.path.sep + "Fully_Connected_Network" + os.path.sep + model + ".json", "w") as json_file:
            json_file.write(json_model)

        # Save weights
        self.model.save_weights(os.getcwd() + os.path.sep + "models" + os.path.sep + "Fully_Connected_Network" + os.path.sep + model + ".h5")
        print("Saved model " + model + " to disk")

        # Save loss and accuracy
#        FileSystem.start_log(str(loss), os.getcwd() + os.path.sep +  "models" + os.path.sep + "Fully_Connected_Network" + os.path.sep + model + "_evaluation.txt")
#        FileSystem.log(str(accuracy), os.getcwd() + os.path.sep + "models" + os.path.sep + "Fully_Connected_Network" + os.path.sep + model + "_evaluation.txt")

        # Save graphical model summary and print summary to console.
        print(self.model.summary())
#        plot_model(self.model, to_file= os.getcwd() + os.path.sep + "models" + os.path.sep + "Fully_Connected_Network" + os.path.sep + model + ".png", show_shapes=True, show_layer_names=True)

    def load_model(self, model = "test_model"):
        print("Attemping to load the model: " + model + " from disk.")

        # read in the model from json
        print("Building model graph using initial specifications")
        self.model = self.build_model(self.input_shape, self.classes)
        print("Successfully built model.")

        #load weights into new model
        print("Attempting to load model from disk..")
        self.model.load_weights("models" + os.path.sep + "Fully_Connected_Network" + os.path.sep + model + ".h5")
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Successfully loaded model weights for: " + model + " from disk.")


    @staticmethod
    def build_model(input_shape, number_of_classes, inner_layer_layout = None):
        """
        Implementation of Fully Connected Network with the following architecture:
        input_shape -> [inner_layer_output] -> softmax(number_of_classes)

        Arguments:
        input_shape -- shape of the images of the dataset
        number_of_classes -- integer, number of classes
        inner_layer_layout -- array, # of neurons per layer.

        Returns:
        model -- a Model() instance in Keras
        """
        X_input = Input(input_shape)

        X = Flatten()(X_input)

        if inner_layer_layout is not None:
            for layer in inner_layer_layout:
                X = Dense(layer, activation='relu', name='fc' + str(layer), kernel_initializer = glorot_uniform())(X)

        X = Dense(number_of_classes, activation="softmax", name='fc' + "output", kernel_initializer = glorot_uniform())(X)

        model = Model(inputs = X_input, outputs = X, name = "Fully Connected Network")

        return model


def train_FCNN(dataset_relative_directory_path, epochs, batch_size):
    x_train, y_train, x_test, y_test, classes = load_dataset(relative_directory_path=dataset_relative_directory_path)
    test_model = FullyConnectedNet(classes=classes)
    test_model.train_model(x_train, y_train, epochs, batch_size)


if __name__ == "__main__":
    """
    "   Main Function
    "   Description: Calls a test function when ran without arguments.  When provided arguments, the script will act as a work thread for training a FullyConnectedNet.
    """
    if len(sys.argv) == 1:
        train_FCNN("practice_data/", 2, 32)
    else:
        # Relative path of dataset
        dataset_relative_directory_path = sys.argv[1]

        # Training Epochs
        epochs = int(sys.argv[2])

        # Batch Size
        batch_size = int(sys.argv[3])

        train_FCNN(dataset_relative_directory_path, epochs, batch_size)
