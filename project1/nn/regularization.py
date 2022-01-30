from abc import ABC, abstractmethod

import numpy as np


class Regularization(ABC):
    """Abstract regularization class.

     Classes can inherit from this class to make sure __call__() and gradient() are implemented.
     """

    @abstractmethod
    def __call__(self, W: np.ndarray) -> float:
        """Calculates and returns the regularization term given the weights.

        :param W: The weights.
        :return: The regularization term.
        """
        raise NotImplementedError('Subclass must implement __call__()')

    @abstractmethod
    def gradient(self, W: np.ndarray) -> np.ndarray:
        """Calculates and returns the gradient of the regularization term with respect to the weights.

        :param W: The weights.
        :return: The gradient of the regularization term with respect to the weights..
        """
        raise NotImplementedError('Subclass must implement gradient()')


class L1(Regularization):
    def __call__(self, W: np.ndarray) -> float:
        """Calculates and returns the L1 regularization (Lasso regression) term given weights.

        :param W: The weights.
        :return: The L1 regularization term.
        """
        return np.sum(np.abs(W))

    def gradient(self, W: np.ndarray) -> np.ndarray:
        """Calculates and returns the gradient of the L1 regularization term with respect to the weights.

       :param W: The weights.
       :return: The gradient of the L1 regularization term with respect to the weights..
       """
        return np.sign(W)


class L2(Regularization):
    def __call__(self, W: np.ndarray) -> float:
        """Calculates and returns the L2 regularization (Ridge regression) term given weights.

        :param W: The weights.
        :return: The L2 regularization term.
        """
        return 1/2 * np.sum(np.power(W, 2))

    def gradient(self, W: np.ndarray) -> np.ndarray:
        """Calculates and returns the gradient of the L2 regularization term with respect to the weights.

       :param W: The weights.
       :return: The gradient of the L2 regularization term with respect to the weights..
       """
        return W


if __name__ == '__main__':
    # Small test suite
    W = [-2, -1, 0, 1, 2]
    l1 = L1()
    l2 = L2()
    print('L1: ', l1(W))
    print('L1 gradient: ', l1.gradient(W))
    print('L2: ', l2(W))
    print('L2 gradient: ', l2.gradient(W))