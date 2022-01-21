from abc import ABC, abstractmethod

import numpy as np

class Activation(ABC):
    @abstractmethod
    def __call__(self, z):
        raise NotImplementedError('Subclass must implement __call__()')

    @abstractmethod
    def gradient(self, z):
        raise NotImplementedError('Subclass must implement gradient()')
    
class Linear(Activation):
    def __call__(self, z):
        return z

    def gradient(self, z):
        return np.ones(z.shape)

class Relu(Activation):
    def __call__(self, z):
        return np.maximum(0, z)

    def gradient(self, z):
        return (z > 0) * 1

class Sigmoid(Activation):
    def __call__(self, z):
        return 1 / (1 + np.exp(-z))

    def gradient(self, z):
        f = self(z)
        return f * (1 - f)

class Tanh(Activation):
    def __call__(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def gradient(self, z):
        f = self(z)
        return 1 - f**2


if __name__ == '__main__':
    # Small test suite
    z = np.array([-1, -0.5, 0, 0.5, 1])

    linear = Linear()
    relu = Relu()
    sigmoid = Sigmoid()
    tanh = Tanh()

    print('Linear:', linear(z))
    print('Linear derivative:', linear.gradient(z))

    print('ReLU:', relu(z))
    print('ReLU derivative:', relu.gradient(z))

    print('Sigmoid:', sigmoid(z))
    print('Sigmoid derivative:', sigmoid.gradient(z))

    print('tanh:', tanh(z))
    print('tanh derivative:', tanh.gradient(z))

