import numpy as np

class AdamOptimizer:

    def __init__(self, grad_shape):
        self.p = 0
        self.q = 0
        self.t = 0
        self.grad_shape = grad_shape

    def gradient(self, sample, label):
        grad = np.zeros(self.grad_shape)
        loss = 0
        membrane_voltage = sample @ self.weights

        t_max = np.argmax(membrane_voltage, axis=0)
        v_max = np.max(membrane_voltage, axis=0)

        sigma = softmax(self.beta * v_max)
        for k in range(self.k):
            label_k = int(label == k)
            grad_k = -self.beta * (sigma[k] - label_k) * sample[t_max[k], :]

            grad[:, k] += grad_k
            loss += - label_k * np.log(sigma[k])

        return grad, loss

    def step(self, gradient, alpha=1e-3, m1=0.9, m2=0.999, epsilon=1e-8):

        self.t += 1
        self.p = m1 * self.p + (1 - m1) * gradient
        self.q = m2 * self.q + (1 - m2) * gradient ** 2

        p_hat = self.p / (1 - m1 ** self.t)
        q_hat = self.q / (1 - m2 ** self.t)

        return alpha * p_hat / (np.sqrt(q_hat) + epsilon)


class SGD:
    
    # TODO: implement SGD optimizer!
    def sgd(gradient, x, y, start, learn_rate=0.1, batch_size=1,
    tolerance=1e-06, random_state=None):

        n_obs = x.shape[0]
        if n_obs != y.shape[0]:
            raise ValueError("'x' and 'y' lengths do not match")
        xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]

        # Initializing the random number generator
        seed = None if random_state is None else int(random_state)
        rng = np.random.default_rng(seed=seed)

        # Initializing the values of the variables
        vector = np.array(start)

        # Setting up and checking the learning rate
        learn_rate = np.array(learn_rate)
        if np.any(learn_rate <= 0):
            raise ValueError("'learn_rate' must be greater than zero")

        # Setting up and checking the size of minibatches
        batch_size = int(batch_size)
        if not 0 < batch_size <= n_obs:
            raise ValueError("'batch_size' must be greater than zero and less than "
                "or equal to the number of observations")

        # Shuffle x and y
        rng.shuffle(xy)

        # Performing minibatch moves
        for start in range(0, n_obs, batch_size):
            stop = start + batch_size
            x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]

            # Recalculating the difference
            grad = np.array(gradient(x_batch, y_batch, vector))
            diff = -learn_rate * grad

            # Checking if the absolute difference is small enough
            if np.all(np.abs(diff) <= tolerance):
                break

            # Updating the values of the variables
            vector += diff

        return vector if vector.shape else vector.item()
