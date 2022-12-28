import numpy as np
from scipy.signal import convolve2d

class Kernel:
    
    def __init__(self, T, time_step, input_size, v0, tau) -> None:
        self.T          =    T
        self.time_step  =    time_step
        self.input_size =    input_size
        self.v0         =    v0
        self.tau        =    tau
        self.tau_s      =    tau / 4
        
        self.kernel = self.k_function()
    
    def k_function(self):
        n = int(self.T / self.time_step)
        t = np.arange(n) * self.time_step
        res  = self.v0 * ( np.exp(-t / self.tau) - np.exp(-t / self.tau_s))

        return res
    
    """ This function is responsible for computing the voltage out of the spike timing
    """
    def get_voltage(self, data : list):

        time_samples = int(self.T / self.time_step)
        
        output = np.zeros((self.input_size, time_samples))
        
        for i in range(len(data[0])):
            output[data[0, i], data[1, i] :] += self.kernel[: time_samples - data[1, i]]
        
        return output