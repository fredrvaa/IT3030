from data_utils.data_generator import DataGenerator
from nn.network import Network

import matplotlib.pyplot as plt

dataset = DataGenerator(noise_level=0).generate_dataset()
(X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset.load_data(flatten=True)

input_dim = X_train.shape[1]

network = Network([input_dim, 20, 20, 1])

out = network.forward_pass(dataset.train_flattened[0].image)

plt.imshow(dataset.train[0].image)
plt.show()

print(out)