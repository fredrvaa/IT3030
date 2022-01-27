import matplotlib.pyplot as plt
import numpy as np

from nn.layer import Layer, HiddenLayer
from nn.loss import Loss, CrossEntropy
from nn.regularization import Regularization


class Network:
    """Class used to create, train, and use a neural network.

    This class allows generation of a whole neural network in a few lines of code.
    This is the highest level of abstraction for the nn package implemented here.
    """

    def __init__(self,
                 loss_function: Loss = CrossEntropy,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 wreg: float = 0.01,
                 wrt: Regularization = None
                 ):
        """
        :param loss_function: The loss function used at the output of the network.
        :param learning_rate: The global learning rate. Single layers can override this locally.
        :param batch_size: How many samples are passed through the network before updating parameters in the layers.
        :param wreg: The weight regularization.
        :param wrt: The weight regularization type.
        """

        self.layers: list[Layer] = []
        self.loss_function: Loss = loss_function()
        self.learning_rate: int = learning_rate
        self.batch_size: int = batch_size
        self.wreg: float = wreg
        self.wrt: Regularization = wrt()

        self.training_loss = []
        self.validation_loss = []
    
    def _forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Propagates a single input sample through the whole network (input-, hidden-, and output layers).

        :param X: The input tensor (a single sample).
        :return: The output tensor (the predicted value based on the input).
        """

        output = X
        for layer in self.layers:
            output = layer.forward_pass(output)

        return output

    def _backward_pass(self, J_L_S: np.ndarray) -> None:
        """Propagates the computed loss gradient through the whole network.

        :param J_L_S: The gradient with respect to output.
        """

        J_L_N = J_L_S
        for layer in self.layers[::-1]:
            J_L_N = layer.backward_pass(J_L_N)

    def _update_parameters(self) -> None:
        """Updates the parameters in all hidden layers (only layers with parameters) of the network."""

        layer: HiddenLayer
        for layer in [layer for layer in self.layers if type(layer) == HiddenLayer]:
            # if layer.size == 5:
            #     print("W", layer.W, '\ndW', np.mean(layer.W_gradients, axis=0))
            layer.update_parameters()


    def add_layer(self, layer: Layer) -> None:
        """Appends a layer to the network.

        The layer is appended to a list of layers. Be sure to add layers in the correct order, with correct
        input/output dimensions.

        :param layer: The layer to be appended to the network.
        """

        self.layers.append(layer)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 1) -> None:
        """Fits the parameters of the network to the training data.

        :param X_train: The training data.
        :param y_train: The labels corresponding to the training data
        :param epochs: Number of times the whole training set is passed through the network.
        """

        self.training_loss = []
        self.validation_loss = []

        print('Fitting model to data...')
        for epoch in range(epochs):
            print(f'Epoch {epoch}')
            aggregated_training_loss = 0
            for i, X in enumerate(X_train):
                y_hat = self._forward_pass(X)
                #print(y_hat)
                aggregated_training_loss += self.loss_function(y_hat, y_train[i])

                J_L_S = self.loss_function.gradient(y_hat, y_train[i])
                self._backward_pass(J_L_S)

                if (epoch*len(X_train) + i + 1) % self.batch_size == 0:
                    self._update_parameters()
            mean_training_loss = aggregated_training_loss / (i + 1)
            self.training_loss.append([epoch, mean_training_loss])

        self.training_loss = np.array(self.training_loss)
        self.validation_loss = np.array(self.validation_loss)

    def visualize_fit(self) -> None:
        """Visualizes training and validation loss recorded during the previous fit of the network."""
        fig, ax = plt.subplots(figsize=(12, 12))
        train_x = self.training_loss[:, 0]
        train_y = self.training_loss[:, 1]
        #val_x = self.validation_loss[:, 0]
        #val_y = self.validation_loss[:, 1]

        ax.plot(train_x, train_y, label='Training loss')
        #ax.plot(val_x, val_y, label='Validation loss')
        ax.legend()

        ax.set_title(f'{self.loss_function} loss during training')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')

        plt.show()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the output label of a given input sample.

        :param X: The input tensor (sample).
        :return: The predicted output label.
        """

        y_hat = self._forward_pass(X)
        prediction = np.zeros(y_hat.shape)
        prediction[np.argmax(y_hat)] = 1
        return prediction

    def __str__(self):
        """Prints all layers and hyperparameters of the network."""

        out = '--------Layers--------\n'
        for layer in self.layers:
            out += layer.__str__() + '\n'
        
        out += '---Hyperparameters---\n'
        out += f'Loss function: \t {self.loss_function}\n'
        out += f'Learning rate: \t {self.learning_rate}\n'

        return out
