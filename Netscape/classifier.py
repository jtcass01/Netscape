from ResNet50 import ResNet50
from FullyConnectedNet import FullyConnectedNet
from h5_utils import load_dataset

class Classifier(object):
    """Abstraction class for handling a variety of classifiers."""
    def __init__(self, model):
        self.model = model
        self.train_results = None
        self.test_results = None

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

    def train_model(self, x, y, epochs, batch_size):
        self.model.train_model(x, y, epochs, batch_size)
        accuracy, loss = self.model.evaluate(x, y)
        self.train_results = {
            "accuracy" : accuracy,
            "loss" : loss
            }
        return accuracy, loss

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

def test_classifier_model_class(model_class):
    # load data
    x_train, y_train, x_test, y_test, classes = load_dataset(relative_directory_path="practice_data/")
    
    # Initialize objects
    test_model = model_class(classes=classes)
    test_classifier = Classifier(model=test_model)

    # Train model
    epochs = int(input("How many epochs would you like to train the model for: "))
    batch_size = int(input("Minibatch size: "))
    test_classifier.train_model(x_train, y_train, epochs, batch_size)

    # Evaluate model
    loss, accuracy = test_classifier.evaluate(x_test, y_test)

    # Save model
    test_classifier.save_model(model="test_model")

def menu():
    print("\n==== Classifier Test ====")
    print(" (1) ResNet50")
    print(" (2) FullyConnectedNet")
    print(" (0) exit")

def handle_menu_response(response):
    if response == "1":
        test_classifier_model_class(ResNet50)
    elif response == "2":
        test_classifier_model_class(FullyConnectedNet)
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