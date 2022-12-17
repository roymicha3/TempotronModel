import matplotlib.pyplot as plt
import numpy as np

from data_loader.DataLoader import DataLoader
from encoders.RateEncoder import RateEncoder
from layers.InputLayer import InputLayer
from models.Tempotron import SimpleTempotron

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
        tau     = 2.164
        v0 = 1
        presynaptic_num = 784
        firing_rate = 50
        
        loader = self.data_loader.load_batch()
        data, _ = next(loader)
        
        layer = InputLayer(self.T, self.time_step, input_size = 784, v0 = v0, tau = tau)
        
        spike_timings = self.encoder.encode(data)
        
        output_voltage = layer(spike_timings)
        
        indices = np.argwhere(np.max(output_voltage[0, :, :], axis=0) > 0)[:, 0]
        np.random.shuffle(indices)
        
        # Initialise the subplot function using number of rows and columns
        figure, axis = plt.subplots(2, 2)
        
        t = np.arange(0, int(self.T / self.time_step), 1)

        voltage = output_voltage[0, :, indices[0]]
        axis[0, 0].plot(t, voltage)
        axis[0, 0].set_title("input voltage - random neuron #1")

        voltage = output_voltage[0, :, indices[1]]
        axis[0, 1].plot(t, voltage)
        axis[0, 1].set_title("input voltage - random neuron #2")

        voltage = output_voltage[0, :, indices[2]]
        axis[1, 0].plot(t, voltage)
        axis[1, 0].set_title("input voltage - random neuron #3")

        voltage = output_voltage[0, :, indices[3]]
        axis[1, 1].plot(t, voltage)
        axis[1, 1].set_title("input voltage - random neuron #4")
        
        # Combine all the operations and display
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
        max_iterations = 2
        presynaptic_num = 784
        weights = np.random.rand(presynaptic_num, 10)

        model = SimpleTempotron(presynaptic_num ,weights.copy(), tau = tau, T= self.T)
        
        # model.train(x_train, y_train, learning_rate, max_iterations=max_iterations)
        model.load("C:\\Users\\roymi\\Projects\\TempotronModel\\weights\\Tempotron-weights2022-12-17_21-07-03.587629")

        model.train_and_validate(max_iterations = max_iterations)

        plt.figure("num of iterations validation")
        plt.xlabel("num of iterations")
        plt.ylabel("accuracy")
        plt.plot(range(len(model.accuracies)), model.accuracies, label="training accuracy")
        plt.plot(range(len(model.val_accuracies)), model.val_accuracies, label="validation accuracy")

        plt.legend()
        plt.show()
        