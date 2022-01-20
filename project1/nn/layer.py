from abc import ABC, abstractmethod

import numpy as np

from nn.activation import Activation, Relu

class Layer(ABC):
    name = None
    def __init__(self):
        if self.name == None:
            raise NotImplementedError('Subclasses must define name')

    @abstractmethod
    def forward_pass(self, X):
        raise NotImplementedError('Subclasses must implement forward_pass()')

    @abstractmethod
    def backward_pass(self, X):
        raise NotImplementedError('Subclasses must implement backward_pass()')

    def __str__(self):
        return f'{self.name} \t {self.size} neurons'

class HiddenLayer(Layer):
    name = 'Hidden layer'
    
    def __init__(self, input_size, output_size, learning_rate=0.001, activation:Activation=Relu):
        super().__init__()
        self.size = input_size

        self.W = np.random.rand(input_size, output_size) - 0.5
        self.b = np.random.rand(output_size) - 0.5

        self._learning_rate = learning_rate
        self._activation = activation()

    def forward_pass(self, X):
        self.input = X
        self.output = self._activation.apply(np.dot(X, self.W) + self.b)
        return self.output

    def backward_pass(self, J):
        pass
        

class SoftmaxLayer(Layer):
    name = 'Softmax layer'

    def __init__(self, input_size):
        self.size = input_size

    def forward_pass(self, X):
        self.input = X
        e = np.exp(X - np.max(X))
        self.output = e / e.sum()
        return self.output

    def backward_pass(self, J):
        pass

