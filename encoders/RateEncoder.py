import numpy as np

class RateEncoder:
    
    def __init__(self, T, time_step, num_of_neurons, firing_rate = 50) -> None:
        self.T              = T
        self.time_step      = time_step
        self.num_of_neurons = num_of_neurons
        self.firing_rate    = firing_rate
    
    def encode(self, data: np.array) -> list:

        assert(len(data.shape) == 3)
        assert(self.num_of_neurons == data.shape[1] * data.shape[2])
        
        max_spikes_in_trial = self.firing_rate * (self.T / 1000)
        num_of_samples = len(data)
        time_samples = int(self.T / self.time_step)
        
        flatten_data = data.copy().reshape((num_of_samples, self.num_of_neurons)) / 255
        
        spike_timings = []
        for sample in flatten_data:
            spike_timing = []
            for neuron in sample:
                num_of_spikes = int(neuron * max_spikes_in_trial)
                spike_timing.append(poisson_events(num_of_spikes, time_samples))
                
            spike_timings.append(spike_timing)

        return spike_timings
    
def poisson_events(k : int, T : int) -> np.array:

    # Generate a random integer between 0 and T-1 (inclusive)
    event_times = np.random.randint(0, T, size=k)
    
    return event_times