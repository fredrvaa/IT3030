from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    """Abstract activation function class.

    Classes can inherit from this class to make sure __call__() and gradient() are implemented.
    """
    @abstractmethod
    def __call__(self, z: np.ndarray) -> np.ndarray:
        """Returns the output of the activation function applied to the input.

        :param z: The input tensor.
        :return: The output of the activation function applied to z.
        """
        raise NotImplementedError('Subclass must implement __call__()')

    @abstractmethod
    def gradient(self, z: np.ndarray) -> np.ndarray:
        """Computes the gradient of the activation function with respect to the input.

        :param z: The input tensor.
        :return: The gradient of the activation function with respect to z.
        """
        raise NotImplementedError('Subclass must implement gradient()')

    def __str__(self):
        return self.__class__.__name__


class Linear(Activation):
    """Class implementing the linear activation function."""

    def __call__(self, z: np.ndarray) -> np.ndarray:
        """Returns the output of the linear activation function applied to the input.

        :param z: The input tensor.
        :return: The output of the linear activation function applied to z.
        """

        return z

    def gradient(self, z: np.ndarray) -> np.ndarray:
        """Returns the gradient of the linear activation function with respect to the input.

        :param z: The input tensor.
        :return: The gradient of the linear activation function with respect to z.
        """

        return np.ones(z.shape)


class Relu(Activation):
    """Class implementing the ReLU activation function."""

    def __call__(self, z: np.ndarray) -> np.ndarray:
        """Returns the output of the ReLU activation function applied to the input.

        :param z: The input tensor.
        :return: The output of the ReLU activation function applied to z.
        """

        return np.maximum(0.0, z)

    def gradient(self, z: np.ndarray) -> np.ndarray:
        """Returns the gradient of the ReLU activation function with respect to the input.

        :param z: The input tensor.
        :return: The gradient of the ReLU activation function with respect to z.
        """

        return (z > 0.0) * 1


class Sigmoid(Activation):
    """Class implementing the sigmoid activation function."""

    def __call__(self, z: np.ndarray) -> np.ndarray:
        """Returns the output of the sigmoid activation function applied to the input.

        :param z: The input tensor.
        :return: The output of the sigmoid activation function applied to z.
        """

        return 1 / (1 + np.exp(-z))

    def gradient(self, z: np.ndarray) -> np.ndarray:
        """Returns the gradient of the sigmoid activation function with respect to the input.

        :param z: The input tensor.
        :return: The gradient of the sigmoid activation function with respect to z.
        """

        f = self(z)
        return f * (1 - f)


class Tanh(Activation):
    """Class implementing the tanh activation function."""

    def __call__(self, z: np.ndarray) -> np.ndarray:
        """Returns the output of the tanh activation function applied to the input.

        :param z: The input tensor.
        :return: The output of the tanh activation function applied to z.
        """

        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def gradient(self, z: np.ndarray) -> np.ndarray:
        """Returns the gradient of the tanh activation function with respect to the input.

        :param z: The input tensor.
        :return: The gradient of the tanh activation function with respect to z.
        """

        f = self(z)
        return 1 - f**2


class SoftMax(Activation):
    """Class implementing the Softmax activation function."""

    def __call__(self, z: np.ndarray) -> np.ndarray:
        """Returns the output of the Softmax activation function applied to the input.

        :param z: The input tensor.
        :return: The output of the Softmax activation function applied to z.
        """

        e = np.exp(z - np.max(z))
        return e / e.sum()

    def gradient(self, z: np.ndarray) -> np.ndarray:
        """Returns the gradient of the Softmax activation function with respect to the input.

        This is done in a rather manual fashion as seen in the lecture slides.

        :param z: The input tensor.
        :return: The gradient of the Softmax activation function with respect to z.
        """

        grad = np.diag(z)

        for i in range(len(grad)):
            for j in range(len(grad)):
                if i == j:
                    grad[i][j] = z[i] * (1 - z[j])
                else:
                    grad[i][j] = -z[i] * z[j]

        return grad


if __name__ == '__main__':
    # Small test suite
    z = np.array([-1, -0.5, 0, 0.001, 0.5, 1])

    linear = Linear()
    relu = Relu()
    sigmoid = Sigmoid()
    tanh = Tanh()

    print('Input: ', z)
    print('------------')
    print('Linear: ', linear(z))
    print('Linear gradient: ', linear.gradient(z))

    print('ReLU: ', relu(z))
    print('ReLU gradient: ', relu.gradient(z))

    print('Sigmoid: ', sigmoid(z))
    print('Sigmoid gradient: ', sigmoid.gradient(z))

    print('tanh: ', tanh(z))
    print('tanh gradient: ', tanh.gradient(z))

