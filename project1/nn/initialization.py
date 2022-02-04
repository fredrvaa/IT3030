from abc import ABC, abstractmethod

import numpy as np


class Initialization(ABC):
    @abstractmethod
    def __call__(self, input_size: int, output_size: int, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError('Subclasses must implement __call__()')


class Uniform(Initialization):
    def __call__(self,
                 input_size: int,
                 output_size: int,
                 weight_range: tuple[float, float] = (-0.1, 0.1),
                 *args, **kwargs
                 ) -> np.ndarray:

        return np.random.uniform(
            weight_range[0],
            weight_range[1],
            (input_size, output_size)
        )


class GlorotUniform(Initialization):
    def __call__(self, input_size: int, output_size: int, *args, **kwargs) -> np.ndarray:
        sd: float = np.sqrt(6.0 / (input_size + output_size))
        return np.random.uniform(-sd, sd, (input_size, output_size))


class GlorotNormal(Initialization):
    def __call__(self, input_size: int, output_size: int, *args, **kwargs) -> np.ndarray:
        sd: float = np.sqrt(2.0 / (input_size + output_size))
        return np.random.normal(0.0, sd, (input_size, output_size))
