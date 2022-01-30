from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    """Abstract loss class.

    Classes can inherit from this class to make sure __call__() and gradient() are implemented.
    """

    @abstractmethod
    def __call__(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """Calculates the loss using the implemented loss function.

        :param y_hat: The predicted output tensor.
        :param y: The ground truth output tensor.
        :return: The calculated loss between y_hat and y
        """

        raise NotImplementedError('Subclass must implement __call__()')

    @abstractmethod
    def gradient(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """Calculates the gradient of the loss function with respect to y_hat.

        :param y_hat: The predicted output tensor.
        :param y: The ground truth output tensor.
        :return: The calculated gradient of the loss between y_hat and y
        """

        raise NotImplementedError('Subclass must implement gradient()')

    def __str__(self):
        return self.__class__.__name__


class MSE(Loss):
    """Class implementing the mean squared error (MSE) loss function."""

    def __call__(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """Calculates the loss using the mean squared error loss function.

        :param y_hat: The predicted output tensor.
        :param y: The ground truth output tensor.
        :return: The mean squared error loss between y_hat and y
        """

        return 1/2 * np.mean((y_hat - y)**2)

    def gradient(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculates the gradient of the mean squared error loss function with respect to y_Hat.

        :param y_hat: The predicted output tensor.
        :param y: The ground truth output tensor.
        :return: The calculated gradient of the mean squared error loss between y_hat and y
        """

        return (y_hat - y) / y_hat.shape[0]


class CrossEntropy(Loss):
    """Class implementing the cross entropy loss function."""

    def __call__(self,  y_hat: np.ndarray, y: np.ndarray, epsilon=1e-7) -> float:
        """Calculates the loss using the cross entropy loss function.

        :param y_hat: The predicted output tensor.
        :param y: The ground truth output tensor.
        :param epsilon: A small term used to clip y_hat such that we don't take log(0).
        :return: The cross entropy loss between y_hat and y
        """
        y_hat = np.clip(y_hat, epsilon, 1. - epsilon)
        return - np.sum(y * np.log(y_hat))

    def gradient(self, y_hat: np.ndarray, y: np.ndarray, epsilon=1e-7) -> np.ndarray:
        """Calculates the gradient of the cross entropy loss function with respect to y_hat.

        :param y_hat: The predicted output tensor.
        :param y: The ground truth output tensor.
        :param epsilon: A small term used to clip y_hat such that we don't divide by 0.
        :return: The calculated gradient of the cross entropy loss between y_hat and y
        """
        y_hat = np.clip(y_hat, epsilon, 1. - epsilon)
        return - (y / y_hat)


if __name__ == '__main__':
    # Small test suite
    y_hat = np.array([1,1,0])
    y = np.array([1,1,1])

    mse = MSE()
    ce = CrossEntropy()

    print("MSE loss: ", mse(y_hat, y))
    print("MSE gradient: ", mse.gradient(y_hat, y))
    print("Cross entropy loss: ", ce(y_hat, y))
    print("Cross entropy gradient: ", ce.gradient(y_hat, y))