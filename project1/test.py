"""SCRIPT USED TO TEST TRAINED MODEL ON NEW DATA CREATED THROUGH CONFIG FILE"""

import argparse

from data_utils.data_generator import DataGenerator
from data_utils.dataset import Dataset
from nn.network import Network
from utils.config_parser import ConfigParser

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, help='path/to/config/specifying/data', required=True)
parser.add_argument('-m', '--model', type=str, help='path/to/trained/model', required=True)
args = parser.parse_args()

config = ConfigParser(args.config)
dataset: Dataset = DataGenerator(**config.get_data_parameters()).generate_dataset()
(X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset.load_data(flatten=True)

network = Network.load(args.model)
test_accuracy = network.test(X_test, y_test)
print(f'Test accuracy on {X_test.shape[0]} samples: ', test_accuracy)