import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy

from scipy import ndimage
from tensorflow.python.framework import ops

from data_utils import convert_to_one_hot, random_mini_batches
from h5_utils import load_dataset

class FullyConnectedNet(object):
    def __init__(self, classes):
        tf.reset_default_graph()
        self.classes = classes.reshape((classes.shape[0],1))
        self.model = None
        self.accuracy = None

    def train_model(self, x_train, y_train, epochs = 1500, batch_size = 32, learning_rate = 0.0001, print_cost = True):
        """
		Created By: Jacob Taylor Cassady
		Last Updated: 2/7/2018
	    Objective: Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

		Arguments:
		learning_rate -- learning rate of the optimization
		epochs -- number of epochs of the optimization loop
		batch_size -- size of a minibatch
		print_cost -- True to print the cost every 100 epochs

		Returns:
		parameters -- parameters learnt by the model. They can then be used to predict.
		"""
        x_train = FullyConnectedNet.process_x(x_train)
        y_train = FullyConnectedNet.process_y(y_train, self.classes.shape[0])
        print("Entering train....")
        ops.reset_default_graph()
        (n_x, m) = x_train.shape              # (n_x: input size, m : number of examples in the train set)
        n_y = y_train.shape[0]		          # n_y : output size.
        costs = []							  # Used to keep track of varying costs.

        # Create placeholders for TensorFlow graph of shape (n_x, n_y)
        print("Creating placeholders for TensorFlow graph...")
        X, Y = FullyConnectedNet.create_placeholders(n_x, n_y)
        print("Complete.\n")

        # Initialize Parameters
        print("Initailizing parameters for TensorFlow graph...")
        parameters = FullyConnectedNet.initialize_parameters(input_size=x_train.shape[0], output_size=self.classes.shape[0])
        print("Complete.\n")

        # Build the forward propagation in the TensorFlow Graph
        print("Building the forward propagation in the TensorFlow Graph...")
        Z3 = FullyConnectedNet.forward_propagation(X, parameters)
        print("Complete.\n")

        # Add the cost function to the Tensorflow Graph
        print("Adding cost function to the TensorFlow Graph")
        cost = FullyConnectedNet.compute_cost(Z3, Y)
        print("Complete.\n")

        # Define the TensorFlow Optimizer.. We are using an AdamOptimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Initialize all the variables with our newly made TensorFlow Graph
        graph = tf.global_variables_initializer()

        # Use the TensorFlow Graph to train the parameters.
        with tf.Session() as session:
            # Run the initialization
            session.run(graph)

            # Perform Training
            for epoch in range(epochs):
                epoch_cost = 0.								# Defines a cost related to the current epoch
                num_minibatches = int(m / batch_size)	# Calculates the number of minibatches in the trainset given a minibatch size
                minibatches = random_mini_batches(x_train, y_train, batch_size)

                for minibatch in minibatches:
                    # Retrieve train_matrix and train_targets from minibatch
                    mini_matrix, mini_targets = minibatch

                    # Run the session to execute the "optimizer" and the "cost",
                    _, minibatch_cost = session.run([optimizer, cost], feed_dict={X:mini_matrix, Y:mini_targets})

                    # Sum epoch cost
                    epoch_cost += minibatch_cost / num_minibatches

                # Done training.  Print the cost of every 100 epochs
                if print_cost == True and epoch % 100 == 0:
                    print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                # Keep track of the cost of every 5 epochs for plotting later
                if print_cost == True and epoch % 5 == 0:
                    costs.append(epoch_cost)

            # Plot the costs for analysis
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iteration ( per 5 )')
            plt.title('Learning rate = ' + str(learning_rate))
            if print_cost == True:
                #plt.show()
                pass

            # Save the parameters as a varaible for prediction and evaluation of fit to test set.
            parameters = session.run(parameters)

            # Develop TensorFlow prediction standards for testing accuracy  of test and train sets
            correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

            # Develop accuracy identifier using TensorFlow
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

            # Display accuracy of train and test predictions.
            print("Train Accuracy: ", self.accuracy.eval({X: x_train, Y: y_train}))

            # Evaluate training accuracy
            training_accuracy = self.accuracy.eval({X: x_train, Y: y_train})

            # Update model
            self.model = {"parameters": parameters, "training_accuracy": training_accuracy, "graph": graph}

    def evaluate(self, x_test, y_test):
        print("\nEvaluating Model...")
        x = FullyConnectedNet.process_x(x_test)
        y = FullyConnectedNet.process_y(y_test, self.classes.shape[0])
        (n_x, m) = x.shape                # (n_x: input size, m : number of examples in the test set)
        n_y = y.shape[0]		          # n_y : output size.
        print("x_test shape: ", str(x.shape))
        print("y_test shape: ", str(y.shape))
        X, Y = FullyConnectedNet.create_placeholders(n_x, n_y)

        # Use the TensorFlow Graph to test the parameters.
        with tf.Session() as session:
            # Run the initialization
            session.run(self.model["graph"])
            test_accuracy = self.accuracy.eval({X: x, Y: y})
            print ("\tTest Accuracy = " + str(test_accuracy))

            return test_accuracy

    def predict_image(self, image_path, target_size=(64,64)):
        image = np.array(ndimage.imread(ismage_path, flatten = False))
        pixels = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
        return self.predict(pixels)

    def predict(self, X):
        # Retrieve parameters from model
        parameters = self.model["parameters"]

        # Convert parameters to tensors
        W1 = tf.convert_to_tensor(parameters["W1"])
        b1 = tf.convert_to_tensor(parameters["b1"])
        W2 = tf.convert_to_tensor(parameters["W2"])
        b2 = tf.convert_to_tensor(parameters["b2"])
        W3 = tf.convert_to_tensor(parameters["W3"])
        b3 = tf.convert_to_tensor(parameters["b3"])

        params = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2,
                    "W3": W3,
                    "b3": b3}

        x = tf.placeholder("float", [12288, 1])

        z3 = FullyConnectedNet.forward_propagation(x, params)
        p = tf.argmax(z3)

        sess = tf.Session()
        prediction = sess.run(p, feed_dict = {x: X})

        return prediction[0]

    @staticmethod
    def create_placeholders(n_x, n_y):
        """
        Created By: Jacob Taylor Cassady
        Last Updated: 2/8/2018
        Objective: Creates the placeholders for the tensorflow session.  These are used in the Tensorflow Graph.

        Arguments:
        n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
        n_y -- scalar, number of classes (from 0 to 5, so -> 6)

        Returns:
        X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
        Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

        Note:
        - We will use None because it let's us be flexible on the number of examples you will for the placeholders.
	        In fact, the number of examples during test/train is different.
        """

        X = tf.placeholder(shape=[n_x, None], dtype=tf.float32, name='X')
        Y = tf.placeholder(shape=[n_y, None], dtype=tf.float32, name='Y')

        return X, Y

    @staticmethod
    def initialize_parameters(input_size, output_size, N1 = 25, N2 = 12):
        """
        Initializes parameters to build a neural network with tensorflow. The shapes are:
					        W1 : [N1, X_train.shape[0]]
					        b1 : [N1, 1]
					        W2 : [N2, N1]
					        b2 : [N2, 1]
					        W3 : [classes.shape[0], N2]
					        b3 : [classes.shape[0], 1]

        Returns:
        parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
        """
        W1 = tf.get_variable('W1', [N1, input_size], initializer = tf.contrib.layers.xavier_initializer(seed=1))
        b1 = tf.get_variable('b1', [N1, 1], initializer = tf.zeros_initializer())
        W2 = tf.get_variable('W2', [N2,N1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
        b2 = tf.get_variable('b2', [N2, 1], initializer = tf.zeros_initializer())
        W3 = tf.get_variable('W3', [output_size,N2], initializer = tf.contrib.layers.xavier_initializer(seed=1))
        b3 = tf.get_variable('b3', [output_size, 1], initializer = tf.zeros_initializer())

        parameters = {
	        "W1" : W1,
	        "b1" : b1,
	        "W2" : W2,
	        "b2" : b2,
	        "W3" : W3,
	        "b3" : b3,
	        }

        return parameters

    @staticmethod
    def forward_propagation(input_matrix, parameters):
        """
        Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

        Arguments:
        input_matrix -- input dataset placeholder, of shape (input size, number of examples)
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
				        the shapes are given in initialize_parameters

        Returns:
        Z3 -- the output of the last LINEAR unit
        """
        # Retrieve the parameters from the dictionary "parameters"
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        W3 = parameters['W3']
        b3 = parameters['b3']

        Z1 = tf.add(tf.matmul(W1, input_matrix), b1)                      # Z1 = np.dot(W1, X) + b1
        A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
        A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3

        return Z3

    @staticmethod
    def compute_cost(Z3, targets):
        """
        Computes the cost

        Arguments:
        Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
        Y -- "true" labels vector placeholder, same shape as Z3
	        Returns:
        cost - Tensor of the cost function
        """

        # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
        logits = tf.transpose(Z3)
        labels = tf.transpose(targets)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        return cost

    @staticmethod
    def process_x(x):
        x_flatten = x.reshape(x.shape[0], -1).T
        return x_flatten/255.

    @staticmethod
    def process_y(y, number_of_targets):
        return convert_to_one_hot(y, number_of_targets)

def test_FCNN(epochs = 2, batch_size = 32):
    x_train, y_train, x_test, y_test, classes = load_dataset(relative_directory_path="practice_data/")
    test_model = FullyConnectedNet(classes=classes)
    test_model.train_model(x_train, y_train, epochs, batch_size)
    test_model.evaluate(x_test, y_test)

if __name__ == "__main__":
    test_FCNN(2, 32)