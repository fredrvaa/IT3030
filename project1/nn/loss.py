from abc import ABC, abstractmethod

import numpy as np

class Loss(ABC):
    @abstractmethod
    def __call__(self, y_hat, y_true):
        raise NotImplementedError('Subclass must implement __call__()')

    @abstractmethod
    def gradient(self, z):
        raise NotImplementedError('Subclass must implement gradient()')

    def __str__(self):
        return self.__class__.__name__


class MSE(Loss):
    def __call__(self, y_hat, y_true):
        return 1/2 * np.mean((y_true - y_hat)**2)

    def gradient(self, y_hat, y_true):
        return - (y_true - y_hat)

class CrossEntropy(Loss):
    def __call__(self,  y_hat, y_true, epsilon=1e-7):
        y_hat = np.clip(y_hat, epsilon, 1. - epsilon)
        return - np.mean((1-y_true) * np.log(1-y_hat) + y_true * np.log(y_hat))

    def gradient(self, y_hat, y_true, epsilon=1e-7):
        y_hat = np.clip(y_hat, epsilon, 1. - epsilon)
        return - (1 - y_true) / (1 - y_hat) + y_true / y_hat

if __name__ == '__main__':
    # Small test suite
    y_hat = np.array([1,0,0])
    y_true = np.array([0,1,0])

    mse = MSE()
    ce = CrossEntropy()

    print("MSE loss: ", mse(y_hat, y_true))
    print("MSE gradient: ", mse.gradient(y_hat, y_true))
    print("Cross entropy loss: ", ce(y_hat, y_true))
    print("Cross entropy gradient: ", ce.gradient(y_hat, y_true))