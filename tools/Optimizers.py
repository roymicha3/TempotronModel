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

