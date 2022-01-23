from abc import ABC, abstractmethod

import numpy as np

from nn.activation import Activation, Relu

class Layer(ABC):
    @abstractmethod
    def forward_pass(self, X):
        raise NotImplementedError('Subclasses must implement forward_pass()')

    @abstractmethod
    def backward_pass(self, X):
        raise NotImplementedError('Subclasses must implement backward_pass()')

    def __str__(self):
        return f'{self.__class__.__name__} \t {self.size} neurons'

class InputLayer(Layer):
    def __init__(self, input_size):
        self.size = input_size

    def forward_pass(self, X):
        return X

    def backward_pass(self, X):
        return X

class HiddenLayer(Layer):    
    def __init__(self, 
                input_size:int, 
                output_size:int, 
                learning_rate:float=0.001, 
                activation:Activation=Relu, 
                weight_range:tuple=(-1,1), 
                bias_range:tuple=(0,1)
            ):
            
        self.size = output_size

        self.W = np.random.uniform(weight_range[0], weight_range[1], (input_size, output_size))
        self.b = np.random.uniform(bias_range[0], bias_range[1], output_size)

        self._learning_rate = learning_rate
        self._activation = activation()

        self.W_gradients = []
        self.b_gradients = []

    def update_parameters(self):
        #print(self.W, "\n", np.sum(self.W_gradients, axis=0))
        self.W -= self._learning_rate * np.sum(self.W_gradients, axis=0)
        self.W_gradients = []
        self.b -= self._learning_rate * np.sum(self.b_gradients, axis=0)
        self.b_gradients = []

    def forward_pass(self, X):
        self.input = X
        self.output = self._activation(np.dot(X, self.W) + self.b)
        return self.output

    def backward_pass(self, J_L_N):
        """
        J_L_W = J_L_N * J_N_W           | J_N_W_hat = np.outer(self.input, np.diag(J_N_sum))
        J_L_N_prev = J_L_N * J_L_N_prev | J_L_N_prev = np.dot(J_N_sum * self.W.T)
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
    def __init__(self, input_size):
        self.size = input_size

    def forward_pass(self, X):
        self.input = X
        e = np.exp(X - np.max(X))
        self.output = e / e.sum()
        return self.output

    def backward_pass(self, J_L_S):
        J_S_N = np.diag(self.output)

        for i in range(len(J_S_N)):
            for j in range(len(J_S_N)):
                if i==j:
                    J_S_N[i][j] = self.output[i] - self.output[i]**2
                else:
                    J_S_N[i][j] = -self.output[i] * self.output[j]


        J_L_N = np.dot(J_L_S, J_S_N)
        return J_L_N


if __name__ == '__main__':
    X = [0.1, 0.2, 0.3]
    sl = SoftmaxLayer(3)
    sl.forward_pass(X)
    print(sl.backward_pass(3))
