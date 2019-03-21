class classifier(object):
    """description of class"""
    def __init__(self):
        self.DNN_Model = None
        self.test_accuracy = 0.0
        self.train_accuracy = 0.0



class DNNPredictionModel(object):
	def __init__(self, parameters, accuracies):
		self.parameters = parameters
		self.accuracies = accuracies

	def evaluate_model(self):
		print(self)


	""" Overloading of string operator """
	def __str__(self):
		return "\t === CURRENT PREDICTION MODEL ===\n" + '\tTrain Accuracy: ' + str(self.accuracies['train_accuracy']) + '\n\tTest Accuracy: ' + str(self.accuracies['test_accuracy']) + '\n\tParameters: ' + str(self.parameters)

	""" Overloading of comparison operators """
	def __eq__(self, other):
		return self.accuracies['train_accuracy'] == other.accuracies['train_accuracy'] and self.accuracies['test_accuracy'] == other.accuracies['test_accuracy']

	def __le__(self, other):
		return self.accuracies['train_accuracy'] <= other.accuracies['train_accuracy'] and self.accuracies['test_accuracy'] <= other.accuracies['test_accuracy']

	def __lt__(self,other):
		return self.accuracies['train_accuracy'] > other.accuracies['train_accuracy'] and self.accuracies['test_accuracy'] > other.accuracies['test_accuracy']

	def __ge__(self, other):
		return self.accuracies['train_accuracy'] >= other.accuracies['train_accuracy'] and self.accuracies['test_accuracy'] >= other.accuracies['test_accuracy']

	def __gt__(self,other):
		return self.accuracies['train_accuracy'] > other.accuracies['train_accuracy'] and self.accuracies['test_accuracy'] > other.accuracies['test_accuracy']

	def load_model(self, model="dnn_best"):
		self.parameters = {
		    'W1' : np.load('../../models/' + model + '/paramW1.npy'),
		    'b1' : np.load('../../models/' + model + '/paramb1.npy'),
		    'W2' : np.load('../../models/' + model + '/paramW2.npy'),
		    'b2' : np.load('../../models/' + model + '/paramb2.npy'),
		    'W3' : np.load('../../models/' + model + '/paramW3.npy'),
		    'b3' : np.load('../../models/' + model + '/paramb3.npy')
		}

		self.accuracies = {
		    'train_accuracy' : np.load('../../models/' + model + '/trainaccuracy.npy'),
		    'test_accuracy' : np.load('../../models/' + model + '/testaccuracy.npy')
		}


	def improve_prediction_model(self, epochs = 5):
		# Load Data Set
		print("Loading data set.")

		X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_practice_dataset()

		test_model = DeepNeuralNetwork(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes)

		for i in range(epochs):
			parameters, accuracies = test_model.train(num_epochs = 1500, print_cost = True)
			new_model = PredictionModel(parameters, accuracies)

			if  new_model > self.prediction_model:
				print("\n\tNew model is better... Displaying accuracies and updating files.. ")
				self.prediction_model = new_model
				print(self.prediction_model)
				self.save_model()
			else:
				print("Previous model is superior or equivalent.")

		print(self)

	def save_model(self):
		parameters = self.parameters
		accuracies = self.accuracies

		W1 = parameters['W1']
		np.save('../../prior_best/paramW1.npy',W1)

		b1 = parameters['b1']
		np.save('../../prior_best/paramb1.npy',b1)

		W2 = parameters['W2']
		np.save('../../prior_best/paramW2.npy',W2)

		b2 = parameters['b2']
		np.save('../../prior_best/paramb2.npy',b2)

		W3 = parameters['W3']
		np.save('../../prior_best/paramW3.npy',W3)

		b3 = parameters['b3']
		np.save('../../prior_best/paramb3.npy',b3)

		train_accuracy = accuracies['train_accuracy']
		np.save('../../prior_best/trainaccuracy.npy',train_accuracy)

		test_accuracy = accuracies['test_accuracy']
		np.save('../../prior_best/testaccuracy.npy',test_accuracy)

