"""SCRIPT USED TO VISUALIZE LOSS AND ACCURACY DURING TRAINING OF A TRAINED MODEL"""

import argparse
from nn.network import Network

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help='path/to/trained/model', required=True)
args = parser.parse_args()

network = Network.load(args.model)
network.visualize_loss()
network.visualize_accuracy()
