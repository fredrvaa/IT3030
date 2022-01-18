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

    @property
    def shape(self):
        return self.train[0].image.shape

    @property
    def flattened_shape(self):
        return self.train_flattened[0].image.shape

    def shuffle_partitions(self):
        np.random.shuffle(self.train)
        np.random.shuffle(self.val)
        np.random.shuffle(self.test)

    def load_data(self, flatten=False):
        X_train = np.array([datapoint.image.flatten() if flatten else datapoint.image for datapoint in self.train])
        y_train = np.array([datapoint.label for datapoint in self.train])

        X_val = np.array([datapoint.image.flatten() if flatten else datapoint.image for datapoint in self.val])
        y_val = np.array([datapoint.label for datapoint in self.val])

        X_test = np.array([datapoint.image.flatten() if flatten else datapoint.image for datapoint in self.test])
        y_test = np.array([datapoint.label for datapoint in self.test])


        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
