import numpy as np

from data_utils.data_generator import DataGenerator
from data_utils.dataset import Dataset
from nn.network import Network
from utils.network_generator import NetworkGenerator
# Generate network
network: Network = NetworkGenerator("config.yaml").generate_network()

print(network)

# Generate dataset
dataset: Dataset = DataGenerator(
    n_samples=500,
    noise_level=0,
    image_dim=10,
    centered=False,
    split_ratios=[0.7, 0.3, 0],
    shape_ratio_range=(0.1, 0.9)).generate_dataset()
print(dataset)
dataset.visualize_data('train', 10)
(X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset.load_data(flatten=True)

# Train network

# X_train = np.array([[1,2,1]]) #, [1,0,1]
# y_train = np.array([[1,0]]) #, [0,1]

epochs = 100
network.fit(X_train, y_train, X_val, y_val, epochs=epochs, verbose=True)

network.visualize_loss()
network.visualize_accuracy()

network.save(f'models/L{len(network.layers)}E{epochs}.pkl')

#  Predict a case
idx = np.random.randint(1)
y = y_train[idx]
y_hat = network.predict(X_train[idx])
print(y, y_hat)



