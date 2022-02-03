import pickle

import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

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
                 wrt: Regularization = None,
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
        self.wrt: Regularization = wrt() if wrt is not None else None

        self.train_loss = []
        self.val_loss = []
        self.train_accuracy = []
        self.val_accuracy = []

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
            layer.update_parameters()

    def add_layer(self, layer: Layer) -> None:
        """Appends a layer to the network.

        The layer is appended to a list of layers. Be sure to add layers in the correct order, with correct
        input/output dimensions.

        :param layer: The layer to be appended to the network.
        """

        self.layers.append(layer)

    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None,
            epochs: int = 1,
            verbose: bool = False,
            ) -> None:
        """Fits the parameters of the network to the training data.

        After fitting/training, loss and accuracy can be visualized using
        visualize_loss() and visualize_accuracy() respectively.

        :param X_train: The training data.
        :param y_train: The labels corresponding to the training data
        :param epochs: Number of times the whole training set is passed through the network.
        :param verbose: Prints additional information during fit such as loss and accuracy if set to True.
        """

        is_validating = X_val is not None and y_val is not None

        self.train_loss = []
        self.train_accuracy = []

        self.val_loss = []
        self.val_accuracy = []

        print('Fitting model to data...')
        for epoch in range(epochs):
            # Train
            aggregated_train_loss: int = 0
            num_train_correct: int = 0
            for i, (X, y) in enumerate(zip(X_train, y_train)):
                y_hat = self._forward_pass(X)

                aggregated_train_loss += self.loss_function(y_hat, y)
                if self.wrt is not None:
                    for layer in self.layers:
                        if type(layer) == HiddenLayer:
                            aggregated_train_loss += self.wreg * self.wrt(layer.W)

                # Create prediction and check if correct
                prediction = np.zeros(y_hat.shape)
                prediction[np.argmax(y_hat)] = 1
                if np.array_equal(prediction, y):
                    num_train_correct += 1

                # Perform backprop by propagating the jacobian of the loss with respect to the (softmax) output
                # through the network.
                J_L_S = self.loss_function.gradient(y_hat, y)
                self._backward_pass(J_L_S)

                # If batch size has been processed, update weights
                if (epoch*len(X_train) + i + 1) % self.batch_size == 0:
                    self._update_parameters()

            # Record train loss and accuracy
            train_loss = aggregated_train_loss / (i + 1)
            train_accuracy = num_train_correct / (i + 1)
            self.train_loss.append([epoch, train_loss])
            self.train_accuracy.append([epoch, train_accuracy])

            val_loss = None
            val_accuracy = None

            # Validation
            if is_validating:
                val_correct = {x: 0 for x in range(len(y_val[0]))}
                aggregated_val_loss: int = 0
                num_val_correct: int = 0
                for i, (X, y) in enumerate(zip(X_val, y_val)):
                    y_hat = self._forward_pass(X)
                    aggregated_val_loss += self.loss_function(y_hat, y)
                    if self.wrt is not None:
                        for layer in self.layers:
                            if type(layer) == HiddenLayer:
                                aggregated_val_loss += self.wreg * self.wrt(layer.W)

                    # Create prediction and check if correct
                    prediction = np.zeros(y_hat.shape)
                    prediction[np.argmax(y_hat)] = 1
                    if np.array_equal(prediction, y):
                        num_val_correct += 1
                        val_correct[np.argmax(y_hat)] += 1

                # Record val loss and accuracy
                val_loss = aggregated_val_loss / (i + 1)
                val_accuracy = num_val_correct / (i + 1)
                self.val_loss.append([epoch, val_loss])
                self.val_accuracy.append([epoch, val_accuracy])

            print(f'Finished epoch {epoch}')
            if verbose:
                loss_table = PrettyTable(['Dataset', 'Loss'], title=f'Loss')
                loss_table.add_row(['Train', round(train_loss, 2)])

                accuracy_table = PrettyTable(['Dataset', 'Accuracy'], title=f'Accuracy')
                accuracy_table.add_row(['Train Accuracy', f'{round(train_accuracy * 100, 2)}%'])

                classification_table = ''
                if is_validating:
                    loss_table.add_row(['Val', round(val_loss, 2)])
                    accuracy_table.add_row(['Val', f'{round(val_accuracy * 100, 2)}%'])
                    classification_table = PrettyTable(['Class', 'Num Correct'], title='Validation Classification')
                    total = 0
                    for key, value in val_correct.items():
                        classification_table.add_row([key, value])
                        total += value
                    classification_table.add_row(['Total', total])

                print(f'{loss_table}\n{accuracy_table}\n{classification_table}')

        # Convert to numpy arrays
        self.train_loss = np.array(self.train_loss)
        self.val_loss = np.array(self.val_loss)
        self.train_accuracy = np.array(self.train_accuracy)
        self.val_accuracy = np.array(self.val_accuracy)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the output label of a given input sample.

        :param X: The input tensor (sample).
        :return: The predicted output label.
        """

        y_hat = self._forward_pass(X)
        prediction = np.zeros(y_hat.shape)
        prediction[np.argmax(y_hat)] = 1
        return prediction

    def visualize_loss(self) -> None:
        """Visualizes training and validation loss recorded during the previous fit of the network."""
        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot train loss
        train_x = self.train_loss[:, 0]
        train_y = self.train_loss[:, 1]
        ax.plot(train_x, train_y, label='Train loss')

        # Plot val loss if it has been recorded
        if self.val_loss.shape[0] > 0:
            val_x = self.val_loss[:, 0]
            val_y = self.val_loss[:, 1]
            ax.plot(val_x, val_y, label='Validation loss')

        ax.legend()

        ax.set_title(f'{self.loss_function} loss during training')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')

        plt.show()

    def visualize_accuracy(self) -> None:
        """Visualizes training and validation accuracy recorded during the previous fit of the network."""
        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot train loss
        train_x = self.train_accuracy[:, 0]
        train_y = self.train_accuracy[:, 1]
        ax.plot(train_x, train_y, label='Train Accuracy')

        # Plot val loss if it has been recorded
        if self.val_accuracy.shape[0] > 0:
            val_x = self.val_accuracy[:, 0]
            val_y = self.val_accuracy[:, 1]
            ax.plot(val_x, val_y, label='Validation Accuracy')

        ax.legend()

        ax.set_title(f'Accuracy during training')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy[\%]')

        plt.show()

    def save(self, file_name: str) -> None:
        """Saves the Network object to a file.

        :param file_name: File name of where to save model. Expects that folder where file should be created exists.
        """

        with open(file_name, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file_name: str) -> 'Network':
        """Loads a Network object from a file.

        :param file_name: File name of saved network.
        :return: A Network object as specified by the file.
        """

        with open(file_name, 'rb') as file:
            network = pickle.load(file)
        return network

    def __str__(self):
        """Prints all layers and hyperparameters of the network."""

        layer_table = PrettyTable(['Type', 'Size'], title='Layers')
        layer: Layer
        for layer in self.layers:
            layer_table.add_row([layer.__class__.__name__, layer.size])

        parameter_table = PrettyTable(['Parameter', 'Value'], title='Hyperparameters')
        parameter_table.add_rows([
            ['Loss function', self.loss_function],
            ['Learning rate', self.learning_rate],
            ['Batch size', self.learning_rate],
            ['Weight regularization', self.wreg],
            ['Weight reg. type', self.wrt.__class__.__name__],
        ])

        return f'{layer_table.get_string()}\n{parameter_table.get_string()}'


if __name__ == '__main__':
    import os

    # Test save and load
    file_name = 'test_save_load.pkl'
    network: Network = Network(batch_size=999)
    network.save(file_name)
    loaded_network: Network = Network.load(file_name)
    assert (loaded_network.batch_size == 999)
    print('Correctly saved and loaded network.')
    os.remove(file_name)
