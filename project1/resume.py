import argparse

# Parse CLI arguments
from nn.network import Network

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='path/to/saved/model', type=str, required=True)
parser.add_argument('-e', '--epochs', help='New number of epochs', type=int, default=None)
parser.add_argument('-s', '--save_folder', help='path/to/save/folder', type=str, default=None)

args = parser.parse_args()

network: Network = Network.load(args.model)
network.resume(args.epochs)

if args.save_folder is not None:
    network.save(args.save_folder)
