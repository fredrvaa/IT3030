from data_utils.data_generator import DataGenerator, Shapes
from nn.network import Network
from nn.layer import InputLayer, HiddenLayer, SoftmaxLayer
from nn.activation import Linear, Relu, Sigmoid, Tanh
from nn.loss import CrossEntropy

import matplotlib.pyplot as plt

# Generate dataset
dataset = DataGenerator(n_samples=1000, noise_level=0, image_dim=5).generate_dataset()
(X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset.load_data(flatten=True)

# Construct network
input_size = X_train.shape[1]
network = Network(loss_function=CrossEntropy)
network.add_layer(InputLayer(input_size))
network.add_layer(HiddenLayer(input_size, 10, activation=Relu, learning_rate=0.1))
network.add_layer(HiddenLayer(10, len(Shapes), activation=Relu, learning_rate=0.1))
network.add_layer(SoftmaxLayer(len(Shapes)))

print(network)

# Train network
network.fit(X_train, y_train, epochs=10)


# Predict a case
y = y_train[0]
y_hat = network.predict(X_train[0])

print(f'y: {y} \ny_hat: {y_hat}')


