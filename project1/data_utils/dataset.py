from dataclasses import dataclass, field
import numpy as np


@dataclass
class DataPoint:
    image:np.ndarray
    label:int

@dataclass
class Dataset:
    train:list = field(default_factory=list)
    val:list = field(default_factory=list)
    test:list = field(default_factory=list)

    def _flatten_partition(self, partition_name):
        partition = getattr(self, partition_name)
        return [DataPoint(datapoint.image.flatten(), datapoint.label) for datapoint in partition]

    @property
    def train_flattened(self):
        return self._flatten_partition('train')
    
    @property
    def val_flattened(self):
        return self._flatten_partition('val')

    @property
    def test_flattened(self):
        return self._flatten_partition('test')

    def shuffle_partitions(self):
        np.random.shuffle(self.train)
        np.random.shuffle(self.val)
        np.random.shuffle(self.test)