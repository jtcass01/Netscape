class classifier(object):
    """description of class"""
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