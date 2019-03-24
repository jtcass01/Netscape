import enum
import os

from ResNet50 import ResNet50
from FullyConnectedNet import FullyConnectedNet
from h5_utils import load_dataset

PYTHON_INTERPRETER_PATH = "\"C:\\Program Files (x86)\\Microsoft Visual Studio\\Shared\\Python36_64\\python.exe\""

class Classifier(object):
    """Abstraction class for handling a variety of classifiers."""
    def __init__(self, model_name):
        self.model_name = model_name
        self.train_results = None
        self.test_results = None

    def __del__(self):
        del self.model_name
        del self

    """ Overloading of comparison operators """
    def __eq__(self, other):
        return self.train_results['accuracy'] == other.train_results['accuracy'] and self.test_results['accuracy'] == other.test_results['accuracy']
    
    def __le__(self, other):
        return self.train_results['accuracy'] <= other.train_results['accuracy'] and self.test_results['accuracy'] <= other.test_results['accuracy']

    def __lt__(self,other):
        return self.train_results['accuracy'] > other.train_results['accuracy'] and self.test_results['accuracy'] > other.test_results['accuracy']

    def __ge__(self, other):
        return self.train_results['accuracy'] >= other.train_results['accuracy'] and self.test_results['accuracy'] >= other.test_results['accuracy']

    def __gt__(self,other):
        return self.train_results['accuracy'] > other.train_results['accuracy'] and self.test_results['accuracy'] > other.test_results['accuracy']

    def train_model(self, dataset_relative_directory_path, epochs, batch_size):
        os.system(PYTHON_INTERPRETER_PATH + " " + os.getcwd() + os.path.sep + str(self.model_name) + ".py " + dataset_relative_directory_path + " " + str(epochs) + " " + str(batch_size))

    def evaluate(self, x, y):
        accuracy, loss = self.model.evaluate(x, y)
        self.test_results = {
            "accuracy" : accuracy,
            "loss" : loss
            }
        return accuracy, loss

    def save_model(self, model="test_model"):
        self.model.save_model(model)

    def load_model(self, model="test_model"):
        self.model.load_model(model)

    class NeuralNetworkModels():
        pass

def test_classifier_model(model_name):
    test_classifier = Classifier(model_name)

    # Train model
    epochs = int(input("How many epochs would you like to train the model for: "))
    batch_size = int(input("Minibatch size: "))
    dataset_relative_directory_path = input("Relative directory path for data: ")
    test_classifier.train_model(dataset_relative_directory_path, epochs, batch_size)

    # Evaluate model
#    loss, accuracy = test_classifier.evaluate(x_test, y_test)


def menu():
    print("\n==== Classifier Test ====")
    print(" (1) ResNet50")
    print(" (2) FullyConnectedNet")
    print(" (0) exit")

def handle_menu_response(response):
    if response == "1":
        test_classifier_model("ResNet50")
    elif response == "2":
        test_classifier_model("FullyConnectedNet")
    elif response == "0":
        pass
    else:
        print("Invalid response.  Please lookover the menu and try again.")

if __name__ == "__main__":
    response = "1"

    while response != "0":
        menu()
        response = input("Choose a model to test: ")
        handle_menu_response(response)