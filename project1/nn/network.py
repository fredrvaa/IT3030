import numpy as np

from nn.layer import Layer, HiddenLayer
from nn.loss import Loss, MSE, CrossEntropy

class Network():
    def __init__(self, loss_function:Loss=CrossEntropy, learning_rate:float=0.001, batch_size=32):
        self.layers = []
        self.loss_function = loss_function()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
    
    def _forward_pass(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward_pass(output)
        return output

    def _backward_pass(self, J_L_N):
        for layer in self.layers[::-1]:
            J_L_N = layer.backward_pass(J_L_N)

    def _update_parameters(self):
        for layer in [layer for layer in self.layers if type(layer) == HiddenLayer]:
            layer.update_parameters()

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def fit(self, X_train, y_train, epochs=1):
        print('Fitting model to data...')
        for epoch in range(epochs):
            print(f'Epoch {epoch}')
            aggregated_output_error = 0
            for i, X in enumerate(X_train):
                y_hat = self._forward_pass(X)
                
                aggregated_output_error += self.loss_function.apply(y_hat, y_train[i])

                J_L_S = self.loss_function.apply_derivative(y_hat, y_train[i])
                self._backward_pass(J_L_S)

                if (epoch*len(X_train) + i + 1) % self.batch_size == 0:
                    self._update_parameters()
            mean_output_error = aggregated_output_error / (i + 1)
            print('Mean output error:', mean_output_error)


        return y_hat

    def predict(self, X):
        y_hat = self._forward_pass(X)
        prediction = np.zeros(y_hat.shape)
        prediction[np.argmax(y_hat)] = 1
        return prediction

    def __str__(self):
        out = '--------Layers--------\n'
        for layer in self.layers:
            out += layer.__str__() + '\n'
        
        out += '---Hyperparameters---\n'
        out += f'Loss function: \t {self.loss_function}\n'
        out += f'Learning rate: \t {self.learning_rate}\n'

        return out