import numpy as np


def mse(y_hat, y_true):
    return np.mean((y_hat - y_true)**2)