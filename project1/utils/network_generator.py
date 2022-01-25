import yaml

import nn.loss
import nn.activation
import nn.layer
import nn.regularization
from nn.layer import InputLayer, HiddenLayer
from nn.network import Network


class NetworkGenerator:
    """A utility class used to parse a config file and generate a neural network using the nn package."""

    def __init__(self, config_file: str):
        """
        :param config_file: Path to a config file.

        The config file is loaded.
        """
        with open(config_file, "r") as stream:
            config = yaml.safe_load(stream)

        self._config = self._parse_config(config)

    def _str_to_class(self, module: str, classname: str):
        """ Gets class from a module using strings of the module name and classname.

        This is used to dynamically load classes from strings in the config file.

        :param module:
        :param classname:
        :return: The class as an attribute.
        """

        return getattr(module, classname)

    def _parse_config(self, config: dict) -> dict:
        """Parses a config by dynamically load objects from strings in the config file.

        Example:
            Given the dictionary: {layers:{output:{type:'SoftMax}}}. The string 'SoftMax' is converted to the
            SoftMax object from the nn.layer module. A softmax layer can then be instantiated by:
            layers['output']['type']().

        :param config: A config dictionary containing strings instead of objects.
        :return: A new config dictionary with objects instead of strings.
        """

        parsed_config = config.copy()
        if 'loss_function' in parsed_config['globals']:
            parsed_config['globals']['loss_function'] = self._str_to_class(nn.loss, parsed_config['globals']['loss_function'])
        if 'wrt' in parsed_config['globals']:
            parsed_config['globals']['wrt'] = self._str_to_class(nn.regularization, parsed_config['globals']['wrt'])

        for layer in parsed_config['layers']['hidden']:
            layer['output_size'] = layer.pop('size')
            if 'activation' in layer:
                layer['activation'] = self._str_to_class(nn.activation, layer['activation'])
            if 'weight_range' in layer:
                layer['weight_range'] = eval(layer['weight_range'])
            if 'bias_range' in layer:
                layer['bias_range'] = eval(layer['bias_range'])

        if 'output' in parsed_config['layers']:
            parsed_config['layers']['output']['type'] = self._str_to_class(nn.layer, parsed_config['layers']['output']['type'])

        return parsed_config

    def generate_network(self) -> Network:
        """Generates a network based on the config file that was provided when instantiating the generator.

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
            network.add_layer(HiddenLayer(input_size=prev_size, **{k: v for k, v in layer.items() if v is not None}))
            prev_size = layer['output_size']

        # Add output layer if specified
        if 'output' in layers:
            network.add_layer(layers['output']['type'](prev_size))

        return network
