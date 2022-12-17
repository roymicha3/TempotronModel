import numpy as np
from tools.Optimizers import AdamOptimizer
from loss.CrossEntropy import CrossEntropy

class OutputLayer:
    
    def __init__(self, T, time_step, input_size, output_size, weights, loss) -> None:
        self.T              =    T
        self.time_step      =    time_step
        self.input_size     =    input_size
        self.output_size    =    output_size
        self.weights        =    weights
        
        if self.weights is None:
                    self.weights = np.random.rand(input_size, output_size)
                    
        self.loss           =    loss
        self.beta = 1
        self.optimizer      =    AdamOptimizer(self.weights.shape)
    
    def get_output(self, data : np.array) -> np.array:
        num_of_samples = len(data)
        time_samples = int(self.T / self.time_step)
        res = np.zeros((num_of_samples, time_samples, self.output_size))

        return data @ self.weights

    
    def __call__(self, data : np.array) -> np.array:
        num_of_samples = len(data)
        res = np.zeros((num_of_samples, self.output_size))

        # TODO: add support for other cost functions 
        for sample_index in range(num_of_samples):
            res[sample_index, :] = self.loss(data[sample_index] @ self.weights)
            
        return res
    
    
    def gradient(self, data : np.array, label : int):
        
        total_grad = np.zeros(self.weights.shape)
        loss = 0

        output = data @ self.weights
        t_max = np.argmax(output, axis=0)
        
        grad, loss = self.loss.gradient(output, label)
        
        for k in range(self.output_size):
            total_grad[:, k] = grad[k] * data[t_max[k], :]

        return total_grad, loss
        # grad = np.zeros(self.weights.shape)
        # loss = 0

        # output = data @ self.weights
        # t_max = np.argmax(output, axis=0)
        # v_max = np.max(output, axis=0)

        # sigma = softmax(self.beta * v_max)
        # for k in range(self.output_size):
        #     label_k = int(label == k)
        #     grad_k = - self.beta * (sigma[k] - label_k) * data[t_max[k], :]

        #     grad[:, k] += grad_k
        #     loss += - label_k * np.log(sigma[k])

        # return grad, loss
    
    def step(self, delta: np.array):
        self.weights += self.optimizer.step(delta)
        
    def save(self, dir_path : str, index : int):
        weights_path = dir_path + "\\output_weights" + str(index) +".npy"
        np.save(weights_path, self.weights)
        
    def load(self, dir_path : str, index : int):
        weights_path = dir_path + "\\output_weights" + str(index) +".npy"
        self.weights = np.load(weights_path)