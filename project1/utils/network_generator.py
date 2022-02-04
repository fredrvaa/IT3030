from nn.layer import InputLayer, HiddenLayer, OutputLayer, OutputActivationLayer
from nn.network import Network
from utils.config_parser import ConfigParser


class NetworkGenerator:
    """A utility class used to parse a config file and generate a neural network using the nn package."""

    def __init__(self, config: dict):
        """
        :param config: Config object holding parameters for 'globals' and 'layers'.
        """

        self._config = config

    def generate_network(self) -> Network:
        """Generates a network based on the config that was provided when instantiating the generator.

        :return: A Network object with layers and hyperparameters as specified by the config file.
        """

        globals = self._config['globals']
        layers = self._config['layers']
        network = Network(**{k: v for k, v in globals.items() if v is not None})

        prev_size = layers['input']['size']

        # Add input layer
        network.add_layer(InputLayer(prev_size))

        # Add hidden layers
        for layer in layers['hidden']:
            # First load global kwargs
            kwargs = {k: v for k, v in globals.items() if v is not None}

            # Then override/add layer kwargs
            kwargs.update({k: v for k, v in layer.items() if v is not None})

            network.add_layer(HiddenLayer(input_size=prev_size, output_size=kwargs['size'], **kwargs))
            prev_size = layer['size']

        # Add output layer
        # First load global kwargs
        kwargs = {k: v for k, v in globals.items() if v is not None}
        # Then override/add layer kwargs
        kwargs.update({k: v for k, v in layers['output'].items() if v is not None})

        network.add_layer(OutputLayer(input_size=prev_size, output_size=kwargs['size'], **kwargs))
        prev_size = layers['output']['size']

        # Add output activation layer if specified
        if 'output_activation' in globals:
            network.add_layer(
                OutputActivationLayer(input_size=prev_size, output_activation=globals['output_activation']))

        return network


if __name__ == '__main__':
    config = ConfigParser('configs/config_base.yaml')
    network = NetworkGenerator(config.get_network_parameters()).generate_network()
    print(network)