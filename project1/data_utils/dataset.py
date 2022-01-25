from dataclasses import dataclass, field
import numpy as np


@dataclass
class DataPoint:
    """Dataclass used to store datapoints. The labels are not one hot encoded."""

    image: np.ndarray
    label: int

@dataclass
class Dataset:
    """Dataclass used to to store whole dataset of Datapoints.

    This class provides multiple methods that can be useful when interacting with the dataset.
    Examples:
        * Retrieving flattened partions
        * Getting the shape of the images stored in the dataset
        * Shuffling the dataset partitions
    """

    train: list = field(default_factory=list)
    val: list = field(default_factory=list)
    test: list = field(default_factory=list)

    def _flatten_partition(self, partition_name: str) -> list[DataPoint]:
        """Flattens and returns the partition given by a string.

        :param partition_name: Partition specified by a string.
        :return: The flattened partition(list).
        """

        partition = getattr(self, partition_name)
        return [DataPoint(datapoint.image.flatten(), datapoint.label) for datapoint in partition]

    @property
    def train_flattened(self) -> list[DataPoint]:
        """Returns the flattened train partition."""

        return self._flatten_partition('train')
    
    @property
    def val_flattened(self) -> list[DataPoint]:
        """Returns the flattened val partition."""

        return self._flatten_partition('val')

    @property
    def test_flattened(self) -> list[DataPoint]:
        """Returns the flattened test partition."""

        return self._flatten_partition('test')

    @property
    def image_shape(self) -> tuple(int):
        """Returns the shape of the images in the dataset. This assumes all images are of the same size."""

        return self.train[0].image.shape

    @property
    def flattened_shape(self) -> tuple(int):
        """Returns the flattened shape of the images in the dataset. This assumes all images are of the same size."""

        return self.train_flattened[0].image.shape

    def shuffle_partitions(self):
        """Shuffles the train, val, and test partitions internally.

        The partitions are not mixed with each other, only the internal order of each partition is shuffled.
        """

        np.random.shuffle(self.train)
        np.random.shuffle(self.val)
        np.random.shuffle(self.test)

    def load_data(self, flatten: bool = False) -> (tuple(np.ndarray), tuple(np.ndarray), tuple(np.ndarray)):
        """Loads and returns a tuple of images and labels for each of the train, val, test partitions.

        The returned tuples contains numpy arrays of the images and labels.

        :param flatten: Specifies if the images should be flattened.
        :return: Tuples (images, labels) for each of the partitions: train, val, test.
        """

        X_train = np.array([datapoint.image.flatten() if flatten else datapoint.image for datapoint in self.train])
        y_train = np.array([datapoint.label for datapoint in self.train])

        X_val = np.array([datapoint.image.flatten() if flatten else datapoint.image for datapoint in self.val])
        y_val = np.array([datapoint.label for datapoint in self.val])

        X_test = np.array([datapoint.image.flatten() if flatten else datapoint.image for datapoint in self.test])
        y_test = np.array([datapoint.label for datapoint in self.test])

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
