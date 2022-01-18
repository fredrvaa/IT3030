import numpy as np


class Layer():
    def __init__(self, input_size, output_size):
        self.W = np.random.rand(input_size, output_size)
        self.b = np.random.rand(output_size)

    def Relu(self, z):
        return max(0, z)

    def forward_pass(self, X):
        self.input = X
        self.output = np.dot(X, self.W) + self.b
        return self.output

if __name__ == '__main__':
    np.random.seed(0)
    l = Layer(3,2)
    print(l.forward_pass(np.array([1,1,1])))
