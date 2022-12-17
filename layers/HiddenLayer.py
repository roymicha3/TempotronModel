import numpy as np
from tools.Optimizers import AdamOptimizer

class HiddenLayer:
    
    def __init__(self, T, time_step, input_size, output_size, weights) -> None:
        self.T              =    T
        self.time_step      =    time_step
        self.input_size     =    input_size
        self.output_size    =    output_size
        self.weights        =    weights
        self.optimizer      =    AdamOptimizer(self.weights.shape)
        
        # TODO: not finished, see how to implement multilayer 
    
    def get_output(self, data : np.array) -> np.array:
        num_of_samples = len(data)
        time_samples = int(self.T / self.time_step)
        res = np.zeros((num_of_samples, time_samples, self.output_size))

        # TODO: add support for other cost functions 
        for sample_index in range(num_of_samples):
            res[sample_index, :, :] = data @ self.weights
            
        return res
    
    def __call__(self, data : np.array) -> np.array:
        return data @ self.weights
    
    
    def gradient(self, data : np.array, label : int):
        return self.weights.T @ data
    
    def step(self, delta: np.array):
        self.weights += self.optimizer.step(delta)
        
    def save(self, dir_path : str, index : int):
        weights_path = dir_path + "\\hidden_weights" + str(index) +".npy"
        np.save(weights_path, self.weights)
        
    def load(self, dir_path : str, index : int):
        weights_path = dir_path + "\\hidden_weights" + str(index) +".npy"
        self.weights = np.load(weights_path)