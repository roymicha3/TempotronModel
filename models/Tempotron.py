import imp
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.special import softmax
import sys
 
# adding tools folder to the system path
from tools.Tools import heaviside, numeric_integral, make_directory
from data_loader.DataLoader import DataLoader
from tools.Optimizers import AdamOptimizer
from data_loader.DataLoader import DataLoader
from encoders.RateEncoder import RateEncoder
from layers.InputLayer import InputLayer
from layers.HiddenLayer import HiddenLayer
from loss.CrossEntropy import CrossEntropy
from layers.OutputLayer import OutputLayer

class SimpleTempotron:

    def __init__(
        self,
        number_of_presynaptic,
        v_threshold = 1,
        v_rest = 0,
        v0 = 2.12,
        tau = 15,
        T = 500,
        time_step = 0.1,
        beta = 1,
        epoch_size = 128,
        k = 10,
        hidden_layer_size = 100,
        firing_rate = 50):

        self.n              =           number_of_presynaptic
        self.k              =           k
        self.v_threshold    =           v_threshold
        self.v_rest         =           v_rest
        self.v0             =           v0
        self.tau            =           tau
        self.tau_s          =           tau / 4
        self.T              =           int(T)
        self.time_step      =           time_step
        self.beta           =           beta
        self.epoch_size     =           epoch_size
        self.accuracies     =           []
        self.val_accuracies =           []
        
        self.data_loader    =           DataLoader(epoch_size, shuffle=True)
        self.encoder        =           RateEncoder(T, time_step, number_of_presynaptic, firing_rate=firing_rate)
        
        self.input_layer    =           InputLayer(T, time_step, number_of_presynaptic, v0, tau)
        
        # hidden_weights = np.random.rand(number_of_presynaptic, hidden_layer_size)
        # self.hidden_layer   =           HiddenLayer(T, time_step, input_size=number_of_presynaptic, output_size = hidden_layer_size, weights = hidden_weights)
        
        self.output_layer   =           OutputLayer(T = T, time_step = time_step, input_size = number_of_presynaptic, output_size = k, weights = None, loss=CrossEntropy((number_of_presynaptic, k), beta = beta))

    def k_function(self, ti):
        n = self.T
        t = np.arange(n) * self.time_step

        res = heaviside(self.time_step, ti, self.T) * (t - ti)
        res  = self.v0 * ( np.exp(-res / self.tau) - np.exp(-res / self.tau_s))

        return res

    def get_accuracy(self, samples, labels):

        predicted_labels = np.argmax(self.output_layer(samples), axis=1)

        num_of_correct_predictions = np.sum(predicted_labels == labels)
        return num_of_correct_predictions / len(labels)

    def train_and_validate(self, max_iterations = 500, save_progress = False):
        progress = 0
        accuracy = 0
        val_accuracy = 0
        validation_size = 250
        
        x_val, y_val = self.data_loader.load_test_batch(validation_size)
        x_val = self.encoder.encode(x_val)
        x_val = self.input_layer(x_val)
        
        train_data_generator = self.data_loader.load_batch()

        for epoch in range(max_iterations):
            
            x_train, y_train = next(train_data_generator)
            x_train = self.encoder.encode(x_train)
            x_train = self.input_layer(x_train)

            progress = int((epoch / max_iterations) * 100)

            epoch_loss = 0

            for sample, label in zip(x_train, y_train):
                # feed forward:
                # sample = self.hidden_layer(sample)
                
                # back propagate:
                grad, loss = self.output_layer.gradient(sample, label)
                self.output_layer.step(grad)
                
                # grad = self.hidden_layer.gradient(grad, label)
                # self.hidden_layer.step(grad)
                
                epoch_loss += loss
                    
            # delta = self.optimizer.step(epoch_grad / self.epoch_size)
            # self.weights += delta

            accuracy = self.get_accuracy(x_train, y_train)
            val_accuracy = self.get_accuracy(x_val, y_val)

            self.accuracies.append(accuracy)
            self.val_accuracies.append(val_accuracy)

            print('Epoch {}. loss: {}. train accuracy: {}. val accuracy: {}'.format(epoch, epoch_loss, accuracy, val_accuracy))
            print('[' + '#' * progress + '.' * (100 - progress) + ']')

            # TODO: might cause trouble, dont erase just yet
            # if np.isclose(accuracy, 1, rtol=1e-05, atol=1e-08, equal_nan=False):
            #     break

        print("the training accuracy is: " + str(accuracy))
        print("the validation accuracy is: " + str(val_accuracy))
        
        if save_progress:
            self.save()
        
    def get_output(self, data : np.array):
        encoded_data = self.encoder.encode(data)
        encoded_data = self.input_layer(encoded_data)
        return self.output_layer.get_output(encoded_data)

    def classify(self, data):

        num_of_samples = len(data)
        time_samples = int((self.T - self.time_step) / self.time_step)
        output = self.output_layer(data)
        # TODO: fix this function
        # predicted_labels = np.max(samples @ self.weights, axis=1) >= self.v_threshold
        # labels = np.array([1 if label else -1 for label in predicted_labels])
        return output
    
    def save(self):
        dir_path = make_directory()
        
        self.output_layer.save(dir_path, 0)
        
    def load(self, dir_path):
        self.output_layer.load(dir_path, 0)
        
    def summerize(self):
        summary = {}
        
        summary["weights"] = self.output_layer.weights
        summary["num_of_iterations"] = len(self.val_accuracies)
        summary["validation_accuracies"] = self.val_accuracies
        summary["test_accuracies"] = self.accuracies
        summary["tau"] = self.tau
        summary["T"] = self.T
        summary["time_step"] = self.time_step
        summary["beta"] = self.beta
        summary["epoch_size"] = self.epoch_size
        
        dir_path = make_directory(prefix="Tempotron-summary")
        dict_path = dir_path + "\\output_summary" + ".npy"
        np.save(dict_path, summary)
        