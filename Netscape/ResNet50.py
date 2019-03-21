import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pydot
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
from data_utils import convert_to_one_hot
from FileSystem import FileSystem

class ResNet50(object):
    """
	Author: Jacob Taylor Cassady

    Description -- Notes Taken from Professor Ng's Convolutoinal Neural Network Course on Coursera:
	Implementation of the popular ResNet50 the following architecture:
	CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
	-> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

	Arguments:
	input_shape -- shape of the images of the dataset
	classes -- integer, number of classes
    """
    def __init__(self, classes, input_shape = (64, 64, 3)):
        tf.reset_default_graph()

        print("\nBuilding ResNet50 model with input_shape:", str(input_shape), "and classes", str(classes))
        self.classes = classes.reshape((classes.shape[0],1))
        self.input_shape = input_shape
        self.model = self.build_model(input_shape, number_of_classes=self.classes.shape[0])

        print("\tCompiling model with the following parameters:")
        print("\t\tOptimizer [Flavor of Gradient Descent] : Adam")
        print("\t\tLoss Function : Categorical Cross Entropy")
        print("\t\tMetrics : Accuracy")
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print("\tResNet50 model now ready to load data.")

    def __del__(self):
        del self.classes
        del self.input_shape
        del self.model

    def build_model(self, input_shape, number_of_classes):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

        Arguments:
        input_shape -- shape of the images of the dataset
        number_of_classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """
        # Define the input as a tensor with shape input_shape
        print(number_of_classes)
        X_input = Input(input_shape)


        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)

        # Stage 1
        X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = ResNet50.convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
        X = ResNet50.identity_block(X, 3, [64, 64, 256], stage=2, block='b')
        X = ResNet50.identity_block(X, 3, [64, 64, 256], stage=2, block='c')

        # Stage 3
        X = ResNet50.convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
        X = ResNet50.identity_block(X, 3, [128, 128, 512], stage=3, block='b')
        X = ResNet50.identity_block(X, 3, [128, 128, 512], stage=3, block='c')
        X = ResNet50.identity_block(X, 3, [128, 128, 512], stage=3, block='d')

        # Stage 4
        X = ResNet50.convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
        X = ResNet50.identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
        X = ResNet50.identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
        X = ResNet50.identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
        X = ResNet50.identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
        X = ResNet50.identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

        # Stage 5
        X = ResNet50.convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
        X = ResNet50.identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
        X = ResNet50.identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

        # AVGPOOL
        X = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

        # output layer
        X = Flatten()(X)
        X = Dense(number_of_classes, activation='softmax', name='fc' + str(number_of_classes), kernel_initializer = glorot_uniform(seed=0))(X)

        # Create model
        model = Model(inputs = X_input, outputs = X, name='ResNet50')

        return model

    def save_model(self, model = 'best_model'):
        # Save the model to JSON
        json_model = self.model.to_json()
        with open(".." + os.path.sep + ".." + os.path.sep + "models" + os.path.sep + model + ".json", "w") as json_file:
            json_file.write(json_model)

        # Save weights
        self.model.save_weights(".." + os.path.sep + ".." + os.path.sep + "models" + os.path.sep + model + ".h5")
        print("Saved model " + model + " to disk")

        # Save loss and accuracy
        loss, accuracy = self.evaluate_model()
        FileSystem.start_log(str(loss), os.getcwd() + os.path.sep + ".." + os.path.sep + ".." + os.path.sep + "models" + os.path.sep + model + "_evaluation.txt")
        FileSystem.log(str(accuracy), os.getcwd() + os.path.sep + ".." + os.path.sep + ".." + os.path.sep + "models" + os.path.sep + model + "_evaluation.txt")

        # Save graphical model summary and print summary to console.
        print(self.model.summary())
        plot_model(self.model, to_file= os.getcwd() + os.path.sep + ".." + os.path.sep + ".." + os.path.sep + "models" + os.path.sep + model + ".png", show_shapes=True, show_layer_names=True)

    def load_model(self, model = "best_model"):
        print("Attemping to load the model: " + model + " from disk.")

        # read in the model from json
        print("Building model graph using initial specifications")
        self.model = self.build_model(self.input_shape, self.classes)
        print("Successfully built model.")

        #load weights into new model
        print("Attempting to load model from disk..")
        self.model.load_weights(".." + os.path.sep + ".." + os.path.sep + "models" + os.path.sep + model + ".h5")
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Successfully loaded model weights for: " + model + " from disk.")


    def train_model(self, x_train, y_train, epochs = 2, batch_size = 32):
        x = ResNet50.process_x(x_train)
        y = ResNet50.process_y(y_train, self.classes.shape[0])
        print("\nTraining model... for ", epochs, "epochs with a batch size of", batch_size)
        print("x_train shape: ", str(x.shape))
        print("y_train shape: ", str(y.shape))
        self.model.fit(x, y, epochs = epochs, batch_size = batch_size)

    def evaluate(self, x_test, y_test):
        x = ResNet50.process_x(x_test)
        y = ResNet50.process_y(y_test, self.classes.shape[0])
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

    @staticmethod
    def process_x(x):
        return x/255.

    @staticmethod
    def process_y(y, number_of_targets):
        return convert_to_one_hot(y, number_of_targets).T

    @staticmethod
    def identity_block(X, f, filters, stage, block):
        """
        Implementation of the identity block as defined in 1.2/reference_images/identity_block.png

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network

        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value. You'll need this later to add back to the main path.
        X_shortcut = X

        # First component of main path
        X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path
        X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path
        X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X = Add()([X_shortcut, X])
        X = Activation('relu')(X)

        return X

    @staticmethod
    def convolutional_block(X, f, filters, stage, block, s = 2):
        """
        Implementation of the convolutional block as defined in 1.2/reference_images/convolutional_block.png

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        s -- Integer, specifying the stride to be used

        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value
        X_shortcut = X


        ##### MAIN PATH #####
        # First component of main path
        X = Conv2D(F1, (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path
        X = Conv2D(F2, (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path
        X = Conv2D(F3, (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

        ##### SHORTCUT PATH ####
        X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

def test_ResNet50(epochs = 2, batch_size = 32):
    x_train, y_train, x_test, y_test, classes = load_dataset(relative_directory_path="practice_data/")
    test_model = ResNet50(input_shape = (64, 64, 3), classes = classes)
    test_model.train_model(x_train, y_train, epochs, batch_size)
    test_model.evaluate(x_test, y_test)


if __name__ == "__main__":
    test_ResNet50(2, 32)
