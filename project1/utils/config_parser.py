import inspect

import yaml

# Load all classes from these files such that eval() can access them without adding them to import statement
from nn.activation import *
from nn.initialization import *
from nn.layer import *
from nn.regularization import *
from nn.loss import *

# Keys that are not evaluated in the recursive dictionary parsing.
STRING_EXCEPTIONS = ['name', 'checkpoint_folder']


class ConfigParser:
    def __init__(self, config_file: str):
        with open(config_file, "r") as stream:
            self._config = yaml.safe_load(stream)

    def _parse_config(self, config) -> dict:
        parsed_config = {}
        for k, v in config.items():
            if v is not None:
                if type(v) is dict:
                    parsed_config[k] = self._parse_config(v)
                elif type(v) is list:
                    parsed_config[k] = [self._parse_config(x) for x in v]
                elif type(v) is str and k not in STRING_EXCEPTIONS:
                    parsed_config[k] = eval(v)
                    if inspect.isclass(parsed_config[k]):
                        parsed_config[k] = parsed_config[k]()
                else:
                    parsed_config[k] = v
                parsed_config
        return parsed_config

    def get_data_parameters(self) -> dict:
        return self._parse_config(self._config['data'])

    def get_network_parameters(self) -> dict:
        return self._parse_config(self._config['network'])

    def get_fit_parameters(self) -> dict:
        return self._parse_config(self._config['fit'])


if __name__ == '__main__':
    c: ConfigParser = ConfigParser('config_new.yaml')
    print('Data:\n', c.get_data_parameters())
    print('Network:\n', c.get_network_parameters())
    print('Fit:\n', c.get_fit_parameters())
