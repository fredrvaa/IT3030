from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from nn.activation import Activation, Relu, SoftMax
from nn.initialization import Initialization, GlorotUniform, Uniform
from nn.regularization import Regularization, L2


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
                 activation: Activation = Relu(),
                 weight_range: tuple[float] = (-0.1, 0.1),
                 bias_range: tuple[float] = (0, 0),
                 initialization: Initialization = GlorotUniform(),
                 wreg: float = 0.01,
                 wrt: Regularization = None,
                 **kwargs,
                 ):
        """
        :param input_size: Size of the input to the layer.
        :param output_size: Size of the output of the layer.
        :param learning_rate: Term used to weigh how much of the gradients are added when updating parameters.
        :param activation: The activation function used at the output of the layer.
        :param weight_range: Interval for the random initialization of weights. Only used if initialization is Uniform.
        :param bias_range: Interval for the random initialization of biases.
        :param initialization: Weight initialization scheme.
        :param wreg: Weight regularization constant.
        :param wrt: Weight regularization type.
        """

        # Sizes
        self._input_size = input_size
        self._output_size = output_size
        self.size: int = output_size

        # Weight and bias initialization
        self._weight_range = weight_range
        self._bias_range = bias_range

        self.W: np.ndarray = initialization(self._input_size, self._output_size, weight_range)
        self.b: np.ndarray = np.random.uniform(bias_range[0], bias_range[1], self._output_size)

        # Hyperparameters
        self.activation: Activation = activation
        self.learning_rate: int = learning_rate
        self.wreg: float = wreg
        self.wrt: Regularization = wrt

        # Storing gradients
        self.W_gradients: list[np.ndarray] = []
        self.b_gradients: list[np.ndarray] = []

        # Storing input/output used in backward_pass()
        self._input: Optional[np.ndarray] = None
        self._output: Optional[np.ndarray] = None

    def update_parameters(self) -> None:
        """Updates all weights and biases using the accumulated gradients and clears the gradients."""
        if self.wrt is not None:
            self.W += self.learning_rate * self.wreg * self.wrt.gradient(self.W)

        self.W -= self.learning_rate * np.sum(self.W_gradients, axis=0)
        self.W_gradients = []
        self.b -= self.learning_rate * np.sum(self.b_gradients, axis=0)
        self.b_gradients = []

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Computes and returns the activation of the input using the weights,
        biases, and activation function of the layer.

        :param X: The input activation tensor.
        :return: The output activation tensor.
        """

        self._input = X
        self._output = self.activation(np.dot(X, self.W) + self.b)
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
        J_N_sum = np.diag(self.activation.gradient(self._output))
        J_N_M = np.dot(J_N_sum, self.W.T)
        J_N_W_hat = np.outer(self._input, np.diag(J_N_sum))

        # Compute final jacobians
        J_L_W = J_L_N * J_N_W_hat
        J_L_b = J_L_N
        J_L_M = np.dot(J_L_N, J_N_M)

        # Store J_L_W and J_L_b for future parameter update
        self.W_gradients.append(J_L_W)
        self.b_gradients.append(J_L_b)

        return J_L_M


class OutputLayer(HiddenLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    """Class for output layer.

    This inherits everything from the HiddenLayer class. It is only used to signify the end of the network.
    """


class OutputActivationLayer(Layer):
    """Class for applying an activation to the output of a network.

    This layer can be added onto the network to alter the output of a network with a different activation.
    For classification problems, it is natural to have a SoftMax output layer at the end of the network.
    """

    def __init__(self, input_size: int, output_activation: Activation = SoftMax()):
        self.size: int = input_size
        self.activation: Activation = output_activation
        self._input: Optional[np.ndarray] = None
        self._output: Optional[np.ndarray] = None

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Computes and returns the activation of the input.

        :param X: The input tensor.
        :return: The softmaxed output tensor.
        """

        self._input = X
        self._output = self.activation(X)
        return self._output

    def backward_pass(self, J_L_O: np.ndarray) -> np.ndarray:
        """Computes and returns the jacobian of the loss with respect to the last hidden layer M (J_L_M).

        This layer is the output layer (often with SoftMax activation function), and we assume the upstream hidden
        layer to be layer M.

        A network might look something like this: Input -> M-1 -> M -> Output(Softmax).

        J_L_O is the jacobian of the loss with respect to the output. Here we compute the jacobian of the loss
        with respect to the last layer (J_L_M).

        :param J_L_O: The jacobian of the loss with respect to the output.
        :return : The computed jacobian of the loss with respect to the upstream layer (M).
        """

        J_O_M = self.activation.gradient(self._output)
        J_L_M = np.dot(J_L_O, J_O_M)
        return J_L_M


if __name__ == '__main__':
    # Small test suite
    X = [3.0, 1.0, 0.2]
    i = InputLayer(3)
    h1 = HiddenLayer(3, 3)
    h2 = HiddenLayer(3, 2)
    o = OutputLayer(2)

    output = o.forward_pass(h2.forward_pass(h1.forward_pass(i.forward_pass(X))))

    print("Output of toy network:", output)

    s = OutputLayer(2)
    print("Output of softmax:", s.forward_pass(np.array([0.41643299, 0.])))  # -> [0.60262938 0.39737062]
