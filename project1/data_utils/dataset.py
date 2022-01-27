from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import itertools

from dataclasses import dataclass, field


@dataclass
class DataPoint:
    """Dataclass used to store datapoints. The labels are not one-hot encoded."""

    image: np.ndarray
    label: int

@dataclass
class Dataset:
    """Dataclass used to to store whole dataset of Datapoints.

    This class provides multiple methods that can be useful when interacting with the dataset.
    Examples:
        * Retrieving flattened partitions
        * Getting the shape of the images stored in the dataset
        * Shuffling the dataset partitions
        * Loading one hot encoded partitions

    :param labels: Enum containing the dataset labels.
    """
    labels: Enum
    train: list = field(default_factory=list)
    val: list = field(default_factory=list)
    test: list = field(default_factory=list)

    def _onehot_encode_label(self, label: int) -> np.ndarray:
        """One hot encodes an integer label.

        The one hot encoded list that is returned is the length of the Shapes enum where one element is set to 1
        and the rest is set to 0.

        :param label: Integer label to be one hot encoded.
        :return: The onehot encoded label in the form of a numpy array (list).
        """

        onehot_label = np.zeros(len(self.labels))
        onehot_label[label] = 1
        return onehot_label

    def flatten_partition(self, partition_name: str) -> list[DataPoint]:
        """Flattens and returns the partition given by a string.

        :param partition_name: Partition specified by a string (must be 'train', 'val', or 'test').
        :return: The flattened partition(list).
        """

        partition = getattr(self, partition_name)
        return [DataPoint(datapoint.image.flatten(), datapoint.label) for datapoint in partition]

    @property
    def image_shape(self) -> tuple[int]:
        """Returns the shape of the images in the dataset. This assumes all images are of the same size."""

        return self.train[0].image.shape

    @property
    def flattened_image_shape(self) -> tuple[int]:
        """Returns the flattened shape of the images in the dataset. This assumes all images are of the same size."""

        return self.flatten_partition('train')[0].image.shape

    def shuffle_partitions(self):
        """Shuffles the train, val, and test partitions internally.

        The partitions are not mixed with each other, only the internal order of each partition is shuffled.
        """

        np.random.shuffle(self.train)
        np.random.shuffle(self.val)
        np.random.shuffle(self.test)

    def _load_partition(self, partition_name: str, flatten: bool, onehot: bool) -> tuple[np.ndarray, np.ndarray]:
        """Loads and returns a a tuple of images and labels for a single partition.

        :param partition_name: Partition specified by a string (must be 'train', 'val', or 'test').
        :param flatten: Specifies if the images should be flattened.
        :param onehot: Specifies if the labels should be one-hot encoded.
        :return: Tuple (images, labels) for the partition
        """

        partition = getattr(self, partition_name)
        X = np.array([datapoint.image.flatten() if flatten else datapoint.image for datapoint in partition])
        y = np.array([self._onehot_encode_label(datapoint.label)
                      if onehot else datapoint.label for datapoint in partition])

        return X, y

    def load_data(self, flatten: bool = True, onehot: bool = True) \
            -> (tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]):
        """Loads and returns a tuple of images and labels for each of the train, val, test partitions.

        The returned tuples contains numpy arrays of the images and labels.

        :param flatten: Specifies if the images should be flattened.
        :param onehot: Specifies if the labels should be one-hot encoded.
        :return: Tuples (images, labels) for each of the partitions: train, val, test.
        """

        (X_train, y_train) = self._load_partition('train', flatten, onehot)
        (X_val, y_val) = self._load_partition('val', flatten, onehot)
        (X_test, y_test) = self._load_partition('test', flatten, onehot)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def _set_ax(self, ax, data: DataPoint) -> None:
        """Utility method for creating a plot of the image.

        Manipulates ax in place.

        :param ax: The axis to plot the data on.
        :param data: The Datapoint to be plotted
        """

        image_dim: int = data[0].image.shape[0] - 1
        ax.set_title(self.labels(data.label).name)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.imshow(data.image, cmap='cividis')
        ax.set_xlim(0, image_dim)
        ax.set_ylim(0, image_dim)

    def visualize_data(self, partition_name: str, n_samples: int = 10) -> None:
        """Visualizes n random samples of the chosen dataset/partition.

        :param partition_name: Partition specified by a string (must be 'train', 'val', or 'test').
        :param n_samples: Number of samples to be visualized
        """

        partition = getattr(self, partition_name)
        random_data: list[DataPoint] = np.random.choice(partition, n_samples, replace=False)


        grid_size: int = int(np.ceil(np.sqrt(n_samples)))
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))

        for data, ax in itertools.zip_longest(random_data, axs.flat):
            if data is not None:
                self._set_ax(ax, data)
            else:
                ax.remove()

        plt.show()

    def visualize_sample(self, partition_name: str, idx: int) -> None:
        """Visualizes a single sample.

        :param partition_name: Partition specified by a string (must be 'train', 'val', or 'test').
        :param idx: Index of the sample to be visualized
        """

        partition = getattr(self, partition_name)
        data: DataPoint = partition[idx]

        fig, ax = plt.subplots(figsize=(12,12))

        self._set_ax(ax, data)

        plt.show()
