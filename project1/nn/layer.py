from abc import ABC, abstractmethod

import numpy as np

from nn.activation import Activation, Relu


class Layer(ABC):
    @abstractmethod
    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Propagates the input through the layer by applying weights, biases, and activation function of the layer.

        :param X: The input tensor
        :return: The output tensor
        """

        raise NotImplementedError('Subclasses must implement forward_pass()')

    @abstractmethod
    def backward_pass(self, J_L_N: np.ndarray) -> np.ndarray:
        """Backpropagates the jacobian through the layer by finding the new jacobian using the previous one.

        :param J_L_N: The computed jacobian passed from the downstream layer.
        :return: The computed jacobian from this layer.
        """

        raise NotImplementedError('Subclasses must implement backward_pass()')

    def __str__(self):
        return f'{self.__class__.__name__} \t {self.size} neurons'


class InputLayer(Layer):
    """Class for the input layer.

    This layer passes tensors directly through, and is mainly used for keeping track of the input dimension
    of a network.
    """

    def __init__(self, input_size: int) -> None:
        """
        :param input_size: Size of the input layer.
        """
        self.size = input_size

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Propagates the input directly through the layer without modification.
        :param X: The input tensor.
        :return: The input tensor.
        """

        return X

    def backward_pass(self, J_L_N: np.ndarray) -> np.ndarray:
        """Propagates the jacobian directly through the layer without modification.
        :param J_L_N: The computed jacobian passed from the downstream layer.
        :return: The same jacobian passed from the downstream layer.
        """

        return J_L_N


class HiddenLayer(Layer):
    """Class for a hidden layer.

    This is the only type of layer that stores parameters (weights/biases). The parameters can be updated using
    update_parameters() after performing an arbitrary amount of forward_pass() and backward_pass().
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 learning_rate: float = 0.001,
                 activation: Activation = Relu,
                 weight_range: tuple = (-1, 1),
                 bias_range: tuple = (0, 1)
                 ):
        """
        :param input_size: Size of the input to the layer.
        :param output_size: Size of the output of the layer.
        :param learning_rate: Term used to weigh how much of the gradients are added when updating parameters.
        :param activation: The activation function used at the output of the layer.
        :param weight_range: Interval for the random initialization of weights.
        :param bias_range: Interval for the random initialization of biases.
        """

        self.size: int = output_size

        self.W: np.ndarray = np.random.uniform(weight_range[0], weight_range[1], (input_size, output_size))
        self.b: np.ndarray = np.random.uniform(bias_range[0], bias_range[1], output_size)

        self._learning_rate: int = learning_rate
        self._activation: Activation = activation()

        self.W_gradients = []
        self.b_gradients = []

        self.input = None
        self.output = None

    def update_parameters(self) -> None:
        """Updates all weights and biases using the accumulated gradients and clears the gradients."""

        self.W -= self._learning_rate * np.sum(self.W_gradients, axis=0)
        self.W_gradients = []
        self.b -= self._learning_rate * np.sum(self.b_gradients, axis=0)
        self.b_gradients = []

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Propagates the input through the layer by applying weights, biases, and activation function of the layer.

        :param X: The input tensor.
        :return: The output tensor.
        """

        self.input = X
        self.output = self._activation(np.dot(X, self.W) + self.b)
        return self.output

    def backward_pass(self, J_L_N: np.ndarray) -> np.ndarray:
        """Backpropagates the jacobian through the layer by finding the new jacobian using the previous one.

        :param J_L_N: The computed jacobian passed from the downstream layer.
        :return: The computed jacobian from this layer.
        """

        # Compute intermediate jacobians

        J_N_sum = self._activation.gradient(np.diag(J_L_N))
        J_N_N_prev = np.dot(J_N_sum, self.W.T)
        J_N_W_hat = np.outer(self.input, np.diag(J_N_sum))
        # Compute final jacobians
        J_L_W = J_L_N * J_N_W_hat
        J_L_b = np.diag(J_N_sum)
        J_L_N_prev = np.dot(J_L_N, J_N_N_prev)

        # Store J_L_W and J_L_b for future parameter update
        self.W_gradients.append(J_L_W)
        self.b_gradients.append(J_L_b)

        return J_L_N_prev


class SoftmaxLayer(Layer):
    """Class for softmax output layer.

    This layer can be added onto the output of a network to compute the softmax of the output. This is usually done
    for classification problems.
    """
    def __init__(self, input_size: int):
        self.size = input_size

        self.input = None
        self.output = None

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Propagates the input through the layer by computing the softmax of the input.

        :param X: The input tensor.
        :return: The softmaxed output tensor.
        """

        self.input = X
        e = np.exp(X - np.max(X))
        self.output = e / e.sum()
        return self.output

    def backward_pass(self, J_L_S: np.ndarray) -> np.ndarray:
        """Backpropagates the jacobian through the layer by finding the new jacobian using the previous one.

        :param J_L_S: The computed jacobian with respect to the output.
        :return: The computed jacobian from this layer.
        """

        J_S_N = np.diag(self.output)

        for i in range(len(J_S_N)):
            for j in range(len(J_S_N)):
                if i == j:
                    J_S_N[i][j] = self.output[i] - self.output[i] ** 2
                else:
                    J_S_N[i][j] = -self.output[i] * self.output[j]

        J_L_N = np.dot(J_L_S, J_S_N)
        return J_L_N


if __name__ == '__main__':
    # Small test suite
    X = [0.1, 0.2, 0.3]
    i = InputLayer(3)
    h1 = HiddenLayer(3, 3)
    h2 = HiddenLayer(3, 2)
    o = SoftmaxLayer(2)

    output = o.forward_pass(h2.forward_pass(h1.forward_pass(i.forward_pass(X))))

    print("Output of toy network:", output)
