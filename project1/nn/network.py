import numpy as np

from nn.layer import Layer
from nn.loss import mse


class Network():
    def __init__(self, layer_sizes=[3,2,1]):
        self.layers = [Layer(i,j) for i,j in zip(layer_sizes[:-1], layer_sizes[1:])]

    def forward_pass(self, X):
        for layer in self.layers:
            X = layer.forward_pass(X)
        return X

    def fit(self, X_train, y_train):
        for X in X_train:
            self.forward_pass(X)

if __name__ == '__main__':
    np.random.seed(0)
    n = Network()
    out = n.forward_pass(np.array([3,3,3]))
    print(out)