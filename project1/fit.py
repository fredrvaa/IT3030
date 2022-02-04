import argparse

from data_utils.data_generator import DataGenerator
from data_utils.dataset import Dataset
from nn.network import Network
from utils.config_parser import ConfigParser

from utils.network_generator import NetworkGenerator
from utils.yn import yes_or_no

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='path/to/config/file', type=str, required=True)
parser.add_argument('-s', '--save_folder', help='path/to/save/folder', type=str, default=None)
args = parser.parse_args()

# Parse config file
config = ConfigParser(args.config)

# Generate dataset and load dataset into train, val, test partitions
dataset: Dataset = DataGenerator(**config.get_data_parameters()).generate_dataset()
(X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset.load_data(flatten=True)

# Generate network
network: Network = NetworkGenerator(config.get_network_parameters()).generate_network()

# Print dataset and network
print(dataset)
print(network)

if not yes_or_no(input('Start fit? [y/n]')):
    exit()

# Fit network
network.fit(X_train, y_train, X_val, y_val, **config.get_fit_parameters())

if args.save_folder is not None:
    network.save(args.save_folder)
