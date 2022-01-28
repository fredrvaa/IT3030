import numpy as np

from data_utils.data_generator import DataGenerator
from data_utils.dataset import Dataset
from nn.network import Network
from utils.network_generator import NetworkGenerator

# Generate network
network: Network = NetworkGenerator("config.yaml").generate_network()

print(network)

# Generate dataset
dataset: Dataset = DataGenerator(n_samples=1000, noise_level=0, image_dim=50).generate_dataset()
(X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset.load_data(flatten=True)

# Train network
network.fit(X_train, y_train, X_val, y_val, epochs=10)

network.visualize_fit()

# Predict a case
idx = np.random.randint(1)
y = y_train[idx]
y_hat = network.predict(X_train[idx])

print(y, y_hat)



