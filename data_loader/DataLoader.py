import random
import numpy as np
from tensorflow import keras

class DataLoader:
    
    def __init__(self, batch_size, shuffle=True) -> None:
        self.batch_size     =    batch_size
        self.shuffle        =    shuffle
        
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
        
    def load_batch(self):
        
        train_size = len(self.x_train)
        
        #shuffle data
        indices = np.arange(train_size)
        
        while True:
            batch_indices = np.random.choice(indices, self.batch_size)
            yield self.x_train[batch_indices], self.y_train[batch_indices]
            
        # old function:
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, train_size - self.batch_size, self.batch_size):
            yield self.x_train[indices[i: i + self.batch_size]], self.y_train[indices[i: i + self.batch_size]]
            
    def load_test_batch(self, batch_size):
        test_size = len(self.x_test)
        
        #shuffle data
        indices = np.arange(test_size)
        if self.shuffle:
            np.random.shuffle(indices)
        
        return self.x_test[indices[: batch_size]], self.y_test[indices[: batch_size]]
    
    def load_test(self):
        
        return self.x_test, self.y_test