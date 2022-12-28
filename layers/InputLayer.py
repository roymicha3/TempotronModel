import numpy as np

from tools.Tools import heaviside, numeric_integral
from layers.Kernel import Kernel

class InputLayer:
    
    def __init__(self, T, time_step, input_size, v0, tau) -> None:
        self.T          =    T
        self.time_step  =    time_step
        self.input_size =    input_size
        self.v0         =    v0
        self.tau        =    tau
        self.tau_s      =    tau / 4
        self.kernel     =    Kernel(T, time_step=time_step, input_size=input_size, v0=v0, tau=tau)
    
    def __call__(self, data : list) -> np.array:
        num_of_samples = len(data)
        time_samples = int(self.T / self.time_step)
        res = np.zeros((num_of_samples, time_samples, self.input_size))

        # create full x_val matrix (all the input presynaptic neurons voltages)
        for sample_index in range(num_of_samples):
            res[sample_index, :, :] = self.kernel.get_voltage(data[sample_index]).T
            
        return res
    
    def k_function(self, ti):
        n = int(self.T / self.time_step)
        t = np.arange(n) * self.time_step

        res = heaviside(self.time_step, ti, self.T) * (t - ti)
        res  = self.v0 * ( np.exp(-res / self.tau) - np.exp(-res / self.tau_s))

        return res
    
    
    def get_voltage_from_spike_timings(self, data : list):

        num_of_input_neurons = len(data)
        time_samples = int(self.T / self.time_step)

        presynaptic_input = np.zeros((num_of_input_neurons, time_samples)).T
        for afferent in range(len(data)):
            spike_times = data[afferent]
            for t in spike_times:
                Ti = t * self.time_step
                presynaptic_input[:, afferent] += self.k_function(Ti)

        return presynaptic_input
