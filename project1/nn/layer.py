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
                 learning_rate: float = 0.01,
                 activation: Activation = Relu,
                 weight_range: tuple[float] = (-0.1, 0.1),
                 bias_range: tuple[float] = (0, 0),
                 init_scheme: str = 'glorot_uniform',
                 ):
        """
        :param input_size: Size of the input to the layer.
        :param output_size: Size of the output of the layer.
        :param learning_rate: Term used to weigh how much of the gradients are added when updating parameters.
        :param activation: The activation function used at the output of the layer.
        :param weight_range: Interval for the random initialization of weights. Only used if init_scheme is 'uniform'.
        :param bias_range: Interval for the random initialization of biases.
        :param init_scheme: Weight initialization scheme. Can be 'glorot_uniform', 'glorot_normal', 'uniform'.
        """

        # Sizes
        self._input_size = input_size
        self._output_size = output_size
        self.size: int = output_size

        # Weight and bias initialization
        self._weight_range = weight_range
        self._bias_range = bias_range

        self.W: np.ndarray = self._initialize_weights(init_scheme)
        self.b: np.ndarray = np.random.uniform(bias_range[0], bias_range[1], self._output_size)

        # Hyperparameters
        self._activation: Activation = activation()
        self._learning_rate: int = learning_rate

        # Storing gradients
        self.W_gradients: list[np.ndarray] = []
        self.b_gradients: list[np.ndarray] = []

        # Storing input/output used in backward_pass()
        self._input: np.ndarray = None
        self._output: np.ndarray = None

    def _initialize_weights(self, init_scheme: str) -> np.ndarray:
        if init_scheme == 'glorot_uniform':
            sd: float = np.sqrt(6.0 / (self._input_size + self._output_size))
            return np.random.uniform(-sd, sd, (self._input_size, self._output_size))
        elif init_scheme == 'glorot_normal':
            sd: float = np.sqrt(2.0 / (self._input_size + self._output_size))
            return np.random.normal(0.0, sd, (self._input_size, self._output_size))
        elif init_scheme == 'uniform':
            return np.random.uniform(
                self._weight_range[0],
                self._weight_range[1],
                (self._input_size, self._output_size)
            )
        else:
            raise ValueError(f'{init_scheme} is not a supported initialization scheme')

    def update_parameters(self) -> None:
        """Updates all weights and biases using the accumulated gradients and clears the gradients."""

        self.W -= self._learning_rate * np.mean(self.W_gradients, axis=0)
        self.W_gradients = []
        self.b -= self._learning_rate * np.mean(self.b_gradients, axis=0)
        self.b_gradients = []

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Computes and returns the activation of the input using the weights,
        biases, and activation function of the layer.

        :param X: The input activation tensor.
        :return: The output activation tensor.
        """

        self._input = X
        self._output = self._activation(np.dot(X, self.W) + self.b)
        return self._output

    def backward_pass(self, J_L_N: np.ndarray) -> np.ndarray:
        """Computes and returns the jacobian of the loss with respect to the upstream layer M (J_L_M).

        Here, we assume the current layer to be layer N, and the upstream layer in a network to be layer M.
        A network might look something like this: Input -> M-1 -> M -> N -> N+1 -> Softmax.

        J_L_N will be passed from the downstream layer (N+1) during the iterative backpropagation algorithm.
        J_L_M is computed here along with the jacobians used to update the parameters (J_L_W and J_L_b).

        :param J_L_N: The jacobian of the loss with respect to the current layer (N).
        :return : The computed jacobian of the loss with respect to the upstream layer (M).
        """

        # Compute intermediate jacobians
        J_N_sum = self._activation.gradient(np.diag(self._output))
        J_N_M = np.dot(J_N_sum, self.W.T)
        J_N_W_hat = np.outer(self._input, np.diag(J_N_sum))

        # Compute final jacobians
        J_L_W = J_L_N * J_N_W_hat
        J_L_b = np.diag(J_N_sum)
        J_L_M = np.dot(J_L_N, J_N_M)

        # Store J_L_W and J_L_b for future parameter update
        self.W_gradients.append(J_L_W)
        self.b_gradients.append(J_L_b)

        return J_L_M


class SoftmaxLayer(Layer):
    """Class for softmax output layer.

    This layer can be added onto the output of a network to compute the softmax of the output. This is usually done
    for classification problems.
    """
    def __init__(self, input_size: int):
        self.size = input_size

        self._input = None
        self._output = None

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Computes and returns the softmax of the input.

        :param X: The input tensor.
        :return: The softmaxed output tensor.
        """

        self._input = X
        e = np.exp(X - np.max(X))
        self._output = e / e.sum()
        return self._output

    def backward_pass(self, J_L_S: np.ndarray) -> np.ndarray:
        """Computes and returns the jacobian of the loss with respect to the last hidden layer M (J_L_M).

        This layer is the output softmax layer, and we assume the upstream hidden layer to be layer M.
        A network might look something like this: Input -> M-1 -> M -> Softmax.

        J_L_S is the jacobian of the loss with respect to the output. Here we compute the jacobian of the loss
        with respect to the last layer (J_L_M). This is done in a rather manual fashion as seen in the lecture slides.

        :param J_L_S: The jacobian of the loss with respect to the softmaxed output.
        :return : The computed jacobian of the loss with respect to the upstream layer (M).
        """

        J_S_M = np.diag(self._output)

        for i in range(len(J_S_M)):
            for j in range(len(J_S_M)):
                if i == j:
                    J_S_M[i][j] = self._output[i] - self._output[i] ** 2
                else:
                    J_S_M[i][j] = -self._output[i] * self._output[j]

        J_L_M = np.dot(J_L_S, J_S_M)
        return J_L_M


if __name__ == '__main__':
    # Small test suite
    X = [0.1, 0.2, 0.3]
    i = InputLayer(3)
    h1 = HiddenLayer(3, 3)
    h2 = HiddenLayer(3, 2)
    o = SoftmaxLayer(2)

    output = o.forward_pass(h2.forward_pass(h1.forward_pass(i.forward_pass(X))))

    print("Output of toy network:", output)
