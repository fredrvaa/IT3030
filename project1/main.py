from data_utils.data_generator import DataGenerator
from nn.network import Network

import matplotlib.pyplot as plt

dataset = DataGenerator(noise_level=0).generate_dataset()

input_dim = dataset.flattened_shape[0]

network = Network([input_dim, 20, 20, 1])

out = network.forward_pass(dataset.train_flattened[0].image)

plt.imshow(dataset.train[0].image)
plt.show()

print(out)