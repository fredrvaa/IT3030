from abc import ABC, abstractmethod

import numpy as np

class Loss(ABC):
    @abstractmethod
    def apply(self, y_hat, y_true):
        raise NotImplementedError('Subclass must implement apply()')

    @abstractmethod
    def apply_derivative(self, z):
        raise NotImplementedError('Subclass must implement apply_derivative()')


class MSE(Loss):
    def apply(self, y_hat, y_true):
        return 1/2 * np.mean((y_true - y_hat)**2)

    def apply_derivative(self, y_hat, y_true):
        return - (y_true - y_hat)

class CrossEntropy(Loss):
    def apply(self,  y_hat, y_true, epsilon=1e-7):
        y_hat = np.clip(y_hat, epsilon, 1. - epsilon)
        return - np.mean((1-y_true) * np.log(1-y_hat) + y_true * np.log(y_hat))

    def apply_derivative(self, y_hat, y_true, epsilon=1e-7):
        y_hat = np.clip(y_hat, epsilon, 1. - epsilon)
        return - (1 - y_true) / (1 - y_hat) + y_true / y_hat

if __name__ == '__main__':
    y_hat = np.array([1,0,0])
    y_true = np.array([0,1,0])
    print("MSE loss: ", MSE().apply(y_hat, y_true))
    print("MSE gradient: ", MSE().apply_derivative(y_hat, y_true))
    print("Cross entropy loss: ", CrossEntropy().apply(y_hat, y_true))
    print("Cross entropy gradient: ", CrossEntropy().apply_derivative(y_hat, y_true))