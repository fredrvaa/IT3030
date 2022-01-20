import numpy as np

from nn.layer import Layer
from nn.loss import mse


class Network():
    def __init__(self):
        self.layers = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def forward_pass(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward_pass(output)
        return output

    def fit(self, X_train, y_train):
        for i, X in enumerate(X_train):
            y_hat = self.forward_pass(X)
            print(mse(y_hat, y_train[i]))

    def __str__(self):
        out = '--- Network ---\n'
        for layer in self.layers:
            out += layer.__str__() + '\n'
        return out