from data_utils.data_generator import DataGenerator, Shapes
from nn.network import Network
from nn.layer import HiddenLayer, SoftmaxLayer
from nn.activation import Linear, Relu, Sigmoid, Tanh

import matplotlib.pyplot as plt

dataset = DataGenerator(noise_level=0).generate_dataset()
(X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset.load_data(flatten=True)

input_size = X_train.shape[1]

network = Network()
network.add_layer(HiddenLayer(input_size, 20, activation=Relu))
network.add_layer(HiddenLayer(20, 20, activation=Relu))
network.add_layer(HiddenLayer(20, len(Shapes), activation=Relu))
network.add_layer(SoftmaxLayer(len(Shapes)))

print(network)

out = network.fit([dataset.train_flattened[0].image],[y_train[0]])

print(out)