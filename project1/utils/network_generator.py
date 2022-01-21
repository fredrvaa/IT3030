import nn.loss
import nn.activation
import nn.layer
from nn.layer import InputLayer, HiddenLayer, SoftmaxLayer
from nn.network import Network

class NetworkGenerator:
    def __init__(self, config):
        self._config = self._parse_config(config)
    

    def _str_to_class(self, module, classname):
        return getattr(module, classname)

    def _parse_config(self, config: dict) -> dict:
        parsed_config = config.copy()
        if 'loss_function' in parsed_config['globals']:
            parsed_config['globals']['loss_function'] = self._str_to_class(nn.loss, parsed_config['globals']['loss_function'])

        for layer in parsed_config['layers']['hidden']:
            layer['output_size'] = layer.pop('size')
            if 'activation' in layer:
                layer['activation']=self._str_to_class(nn.activation, layer['activation'])

        parsed_config['layers']['output']['type'] = self._str_to_class(nn.layer, parsed_config['layers']['output']['type'])
        return parsed_config

    def generate_network(self) -> Network:
        globals = self._config['globals']
        layers = self._config['layers']
        network = Network(**{k: v for k, v in globals.items() if v is not None})

        prev_size = layers['input']['size']
        network.add_layer(InputLayer(prev_size))
        for layer in layers['hidden']:
            network.add_layer(HiddenLayer(input_size=prev_size, **{k: v for k, v in layer.items() if v is not None}))
            prev_size = layer['output_size']

        if type(layers['output']['type']) is SoftmaxLayer:
            network.add_layer(SoftmaxLayer(prev_size))

        return network
