import numpy as np
from scipy.special import softmax

class CrossEntropy:
    
    def __init__(self, shape, beta = 1, label_types = 10) -> None:
        self.shape = shape
        self.beta = beta
        self.label_types = label_types
    
    def __call__(self, data : np.array) -> np.array:
        v_max = np.max(data, axis=0)
        sigma = softmax(self.beta * v_max)
        # actually softmax is an activation function, should be implemented in the future
        return sigma
    
    def gradient(self, data: np.array, label : int) -> np.array:
        loss = 0
        
        v_max = np.max(data, axis=0)
        y = self.encode_one_hot(label)

        sigma = softmax(self.beta * v_max)
        grad = -self.beta * (sigma - y)

        for k in range(self.label_types):
            loss += - y[k] * np.log(sigma[k])

        return grad, loss
    
    def loss(self, data: np.array, label: int):
        output_size = self.shape[1]
        loss = 0
        v_max = np.max(data, axis=0)

        sigma = softmax(self.beta * v_max)
        for k in range(output_size):
            label_k = int(label == k)
            loss += - label_k * np.log(sigma[k])

        return loss
    
    def encode_one_hot(self, y):
        return np.eye(self.label_types)[y]