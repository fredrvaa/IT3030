"""SCRIPT USED TO VISUALIZE NEW DATA CREATED THROUGH CONFIG FILE"""

import argparse

from data_utils.data_generator import DataGenerator
from data_utils.dataset import Dataset
from utils.config_parser import ConfigParser

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, help='path/to/config', required=True)
parser.add_argument('-p', '--partition', type=str, help="Which partition to sample from. train/val/test", required=True)
parser.add_argument('-n', '--display_number',
                    help='Number of data samples from the training set to display before fit.',
                    type=int, required=False, default=1)
args = parser.parse_args()

config = ConfigParser(args.config)
dataset: Dataset = DataGenerator(**config.get_data_parameters()).generate_dataset()
dataset.visualize_data(args.partition, args.display_number)
