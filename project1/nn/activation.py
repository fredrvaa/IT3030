import numpy as np

class Activation:
    @classmethod
    def apply(cls, z):
        raise NotImplementedError

    @classmethod
    def apply_derivative(cls, z):
        raise NotImplementedError

class Linear(Activation):
    @classmethod
    def apply(cls, z):
        return z

    @classmethod
    def apply_derivative(cls, z):
        return np.ones(z.shape)

class Relu(Activation):
    @classmethod
    def apply(cls, z):
        return np.maximum(0, z)

    @classmethod
    def apply_derivative(cls, z):
        return (z > 0) * 1

class Sigmoid(Activation):
    @classmethod
    def apply(cls, z):
        return 1 / (1 + np.exp(-z))

    @classmethod
    def apply_derivative(cls, z):
        f = cls.apply(z)
        return f * (1 - f)

class Tanh(Activation):
    @classmethod
    def apply(cls, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    @classmethod
    def apply_derivative(cls, z):
        f = cls.apply(z)
        return 1 - f**2


if __name__ == '__main__':
    z = np.array([-1, -0.5, 0, 0.5, 1])

    print('Linear:', Linear.apply(z))
    print('Linear derivative:', Linear.apply_derivative(z))

    print('ReLU:', Relu.apply(z))
    print('ReLU derivative:', Relu.apply_derivative(z))

    print('Sigmoid:', Sigmoid.apply(z))
    print('Sigmoid derivative:', Sigmoid.apply_derivative(z))

    print('tanh:', Tanh.apply(z))
    print('tanh derivative:', Tanh.apply_derivative(z))

