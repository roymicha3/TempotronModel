import numpy as np

class FlattenEncoder:
    
    def __init__(self) -> None:
        pass

    def encode(self, data: np.array, labels) -> list:

        assert(len(data.shape) == 3)

        num_of_samples = len(data)

        flatten_data = data.copy().reshape((num_of_samples, data.shape[1] * data.shape[2])) / 255

        return flatten_data, self.one_hot(labels)
    
    def one_hot(self, labels, max_val=10):
        encoded = np.zeros((len(labels), max_val), dtype=int)
        encoded[np.arange(len(labels)), labels] = 1 
        return encoded
    