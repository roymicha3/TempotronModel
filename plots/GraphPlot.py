import matplotlib.pyplot as plt
import numpy as np

from data_loader.DataLoader import DataLoader
from encoders.RateEncoder import RateEncoder
from layers.InputLayer import InputLayer
from layers.Kernel import Kernel
from models.Tempotron import SimpleTempotron
from models.Perceptron import Perceptron
from tools.Tracker import Tracker

class GraphPlot:
    
    def __init__(self, T = 500, time_step= 0.1, epoch_size = 64, num_of_input_neurons = 784) -> None:
        self.T                      =   T
        self.time_step              =   time_step
        self.epoch_size             =   epoch_size
        self.num_of_input_neurons   =   num_of_input_neurons
        self.data_loader            =   DataLoader(self.epoch_size, shuffle=True)
        self.encoder                =   RateEncoder(self.T, self.time_step, self.num_of_input_neurons)

    def plot_rate_encoding(self):
        
        data, _ = self.data_loader.load_batch() 
        
        spike_timings = self.encoder.encode(data)[0]
        
        # Get the number of neurons
        num_neurons = len(spike_timings)

        # Initialize figure
        fig, ax = plt.subplots()

        # Iterate over the neurons
        for i in range(num_neurons):
            # Get the spike timings for the neuron
            neuron_spike_timings = spike_timings[i]
            # Iterate over the spike timings
            for t in neuron_spike_timings:
                # Plot a point at the spike timing
                ax.scatter(t, i, color='black', s=5)

        # Set the title and axis labels
        ax.set_title('Raster Plot')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron')

        # Show the plot
        plt.show()
        
    def plot_input_layer_output(self):
        tau                 = 2.164
        v0                  = 1
        presynaptic_num     = 784
        firing_rate         = 50
        
        loader = self.data_loader.load_batch()
        data, _ = next(loader)
        
        kernel = Kernel(self.T, self.time_step, input_size = presynaptic_num, v0 = v0, tau = tau)
        
        spike_timings = self.encoder.encode_new(data)
        
        output_voltage = kernel.get_voltage(spike_timings[0])
        
        indices = np.argwhere(np.max(output_voltage, axis=1) > 0)[:, 0]
        np.random.shuffle(indices)
        
        cols = 3
        rows = 3
        
        figure, axis = plt.subplots(rows, cols)
        
        t = np.arange(0, int(self.T / self.time_step), 1)
        
        for i in range(rows):
            for j in range(cols):
                voltage = output_voltage[indices[i * cols + j]]
                axis[i, j].plot(t, voltage)
                axis[i, j].set_title("input voltage - random neuron #" + str(i * cols + j))

        plt.show()
        
    def plot_output_layer_output(self):
        tau     = 2.164
        presynaptic_num = 784
        firing_rate = 50
        
        loader = self.data_loader.load_batch()
        data, y = next(loader)

        model = SimpleTempotron(presynaptic_num, tau = tau, T= self.T, firing_rate=firing_rate)

        model.load("C:\\Users\\roymi\\Projects\\TempotronModel\\weights\\Tempotron-weights2022-12-17_21-07-03.587629")

        output_voltage = model.get_output(data)
        
        # Initialise the subplot function using number of rows and columns
        figure, axis = plt.subplots(2, 5, sharex=True, sharey=True)
        
        plt.suptitle("output voltage for label " + str(y[0]))
        
        t = np.arange(0, int(self.T / self.time_step), 1)
        
        for i in range(2):
            for j in range(5):
                index = i * 5 + j
                voltage = output_voltage[0, :, index]

                axis[i, j].plot(t, voltage)
                title = "output voltage - neuron #" + str(index)
                axis[i, j].set_title(title)
        
        plt.show()
        
        
    def plot_model_validation(self):
        tau = 2.164
        max_iterations = 1000
        presynaptic_num = 784

        model = SimpleTempotron(presynaptic_num ,tau = tau, T= self.T)
        
        # model.train(x_train, y_train, learning_rate, max_iterations=max_iterations)
        model.load("C:\\Users\\roymi\\Projects\\TempotronModel\\weights\\Tempotron-weights2022-12-17_21-07-03.587629")

        model.train_and_validate(max_iterations = max_iterations, save_progress=True)

        plt.figure("num of iterations validation")
        plt.xlabel("num of iterations")
        plt.ylabel("accuracy")
        plt.plot(range(len(model.accuracies)), model.accuracies, label="training accuracy")
        plt.plot(range(len(model.val_accuracies)), model.val_accuracies, label="validation accuracy")

        plt.legend()
        plt.show()
        
    def plot_perceptron_tempotron_comparison(self):
        
        tau = 2.164
        max_iterations = 500
        presynaptic_num = 784

        tempotron_model = SimpleTempotron(presynaptic_num ,tau = tau, T= self.T)
        perceptron_model = Perceptron(784, 10)
        
        tempotron_model.train_and_validate(max_iterations = max_iterations, save_progress=True)
        
        perceptron_model.train(max_iterations = max_iterations)
        
        plt.figure("num of iterations validation - Perceptron vs Tempotron")
        plt.xlabel("num of iterations")
        plt.ylabel("accuracy")
        plt.plot(range(len(perceptron_model.val_accuracies)), perceptron_model.val_accuracies, label="perceptron accuracy")
        plt.plot(range(len(tempotron_model.val_accuracies)), tempotron_model.val_accuracies, label="temportron accuracy")

        plt.legend()
        plt.show()
        
    def train_in_parts(self):
        tau = 2.164
        max_iterations = 2
        presynaptic_num = 784

        tempotron_model = SimpleTempotron(presynaptic_num ,tau = tau, T= self.T)
        
        tempotron_model.train_and_validate(max_iterations = max_iterations, save_progress=True)
        tempotron_model.track()
        
        tracker = Tracker(dir_path = tempotron_model.tracker.directory_path())
        
        tempotron_model = SimpleTempotron(presynaptic_num ,tracker = tracker)
        tempotron_model.load()
        
        max_iterations = 5
        
        tempotron_model.train_and_validate(max_iterations = max_iterations, save_progress=True)