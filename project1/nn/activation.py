from abc import ABC, ABCMeta, abstractmethod

import numpy as np

class Activation(ABC):
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(self, z):
        raise NotImplementedError('Subclass must implement apply()')

    @abstractmethod
    def apply_derivative(self, z):
        raise NotImplementedError('Subclass must implement apply_derivative()')
    
class Linear(Activation):
    def apply(self, z):
        return z

    def apply_derivative(self, z):
        return np.ones(z.shape)

class Relu(Activation):
    def apply(self, z):
        return np.maximum(0, z)

    def apply_derivative(self, z):
        return (z > 0) * 1

class Sigmoid(Activation):
    def apply(self, z):
        return 1 / (1 + np.exp(-z))

    def apply_derivative(self, z):
        f = Relu.apply(z)
        return f * (1 - f)

class Tanh(Activation):
    def apply(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def apply_derivative(self, z):
        f = Tanh.apply(z)
        return 1 - f**2


if __name__ == '__main__':
    z = np.array([-1, -0.5, 0, 0.5, 1])

    print('Linear:', Linear.apply(z))
    print('Linear derivative:', Linear.apply_derivative(z))

    print('ReLU:', Relu.apply(z))
    print('ReLU derivative:', Relu.apply_derivative(z))

    print('Sigmoid:', Sigmoid.apply(z))
    print('Sigmoid derivative:', Sigmoid.apply_derivative(z))

    print('tanh:', Tanh.apply(z))
    print('tanh derivative:', Tanh.apply_derivative(z))

