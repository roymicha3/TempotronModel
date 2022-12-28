import numpy as np

class RateEncoder:
    
    def __init__(self, T, time_step, num_of_neurons, firing_rate = 20) -> None:
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
    
    """This function computes the spike timings and returns two arrays - the neurons indexes and its corresponding spike timings
    """
    def encode_new(self, data: np.array) -> list:

        assert(len(data.shape) == 3)
        assert(self.num_of_neurons == data.shape[1] * data.shape[2])
        
        max_spikes_in_trial = self.firing_rate * (self.T / 1000)
        num_of_samples = len(data)
        time_samples = int(self.T / self.time_step)
        
        flatten_data = data.reshape((num_of_samples, self.num_of_neurons)) / 255
        
        res = []
        
        for sample in flatten_data:
            spike_timings = []
            spike_indexes = []
            
            for neuron_idx, neuron in zip(range(len(sample)), sample):
                num_of_spikes = int(neuron * max_spikes_in_trial)
                
                spike_timings = np.append(spike_timings, poisson_events(num_of_spikes, time_samples))
                spike_indexes = np.append(spike_indexes, np.ones(num_of_spikes, dtype=int) * neuron_idx)

            res.append(np.array([spike_indexes, spike_timings], dtype=int))
            
        return res
    
def poisson_events(k : int, T : int) -> np.array:

    # Generate a random integer between 0 and T-1 (inclusive)
    event_times = np.random.randint(0, T, size=k)
    
    return event_times