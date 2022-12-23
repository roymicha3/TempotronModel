import numpy as np
from scipy.special import softmax

from tools.Tools import make_directory
from tools.Optimizers import AdamOptimizer
from data_loader.DataLoader import DataLoader
from encoders.FlattenEncoder import FlattenEncoder

class Perceptron:
    
    def __init__(self, input_size, output_size):
        self.weights = np.random.uniform(0, 1, (input_size, output_size))
        self.biases = np.random.uniform(0, 1, output_size)
        self.epoch_size = 512
        
        self.data_loader = DataLoader(self.epoch_size, shuffle=True)
        self.encoder = FlattenEncoder()
        self.optimizer = AdamOptimizer((input_size, output_size))
        
        self.val_accuracies = []
        self.accuracies = []

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases

    def cross_entropy_loss(self, outputs, targets):
        return -np.sum(targets * np.log(softmax(outputs, axis=1)))

    def backward(self, inputs, outputs, targets):
        error = softmax(outputs, axis=1) - targets
        gradient_weights = np.dot(inputs.T, error)
        gradient_biases = np.sum(error, axis=0)
        return gradient_weights, gradient_biases

    def update_weights(self, gradient_weights, gradient_biases, learning_rate):
        self.weights -= learning_rate * gradient_weights
        self.biases -= learning_rate * gradient_biases

    def accuracy(self, outputs, targets):
        outputs = np.argmax(outputs, axis=1)
        targets = np.argmax(targets, axis=1)
        return np.mean(outputs == targets)

    def train(self, max_iterations = 200):
        progress = 0
        accuracy = 0
        
        val_inputs, val_targets = self.data_loader.load_test()
        val_inputs, val_targets = self.encoder.encode(val_inputs, val_targets)
        
        train_data_generator = self.data_loader.load_batch()
        
        for epoch in range(max_iterations):
            
            progress = int((epoch / max_iterations) * 100)
            
            inputs, targets = next(train_data_generator)
            inputs, targets = self.encoder.encode(inputs, targets)
                
            outputs = self.forward(inputs)
            loss = self.cross_entropy_loss(outputs, targets)
            gradient_weights, gradient_biases = self.backward(inputs, outputs, targets)
            
            # step = self.optimizer.step(gradient_weights)
            step = gradient_weights * 0.01
            self.update_weights(step, gradient_biases, 1)
                
            outputs = self.forward(inputs)
            val_outputs = self.forward(val_inputs)
            
            accuracy = self.accuracy(outputs, targets)
            val_accuracy = self.accuracy(val_outputs, val_targets)
            
            self.accuracies.append(accuracy)
            self.val_accuracies.append(val_accuracy)

            print(f'Epoch {epoch}. loss: {loss}. test accuracy {accuracy}. val accuracy: {val_accuracy}')
            print('[' + '#' * progress + '.' * (100 - progress) + ']')
            
    def summerize(self):
        summary = {}
        
        summary["weights"] = self.weights
        summary["num_of_iterations"] = len(self.val_accuracies)
        summary["validation_accuracies"] = self.val_accuracies
        summary["test_accuracies"] = self.accuracies
        summary["epoch_size"] = self.epoch_size
        
        dir_path = make_directory(prefix="Perceptron-summary")
        dict_path = dir_path + "\\output_summary" + ".npy"
        np.save(dict_path, summary)
