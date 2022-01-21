from data_utils.data_generator import DataGenerator
from utils.network_generator import NetworkGenerator

import matplotlib.pyplot as plt

import yaml

with open("config.yaml", "r") as stream:
    config = yaml.safe_load(stream)

# Generate network
network = NetworkGenerator(config).generate_network()

print(network)

# Generate dataset
dataset = DataGenerator(n_samples=1000, noise_level=0, image_dim=50).generate_dataset()
(X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset.load_data(flatten=True)

# Train network
network.fit(X_train, y_train, epochs=10)

# Predict a case
y = y_train[0]
y_hat = network.predict(X_train[0])

print(f'y: {y} \ny_hat: {y_hat}')


