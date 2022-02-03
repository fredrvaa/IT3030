from nn.network import Network

network = Network.load('checkpoints/layers3_epoch50.pkl')
network.resume()