from enum import Enum

import numpy as np

from data_utils.dataset import Dataset, DataPoint


class Shapes(Enum):
    """Enum class for dataset labels."""

    square = 0
    filled_square = 1
    cross = 2
    circle = 3
    filled_circle = 4
    # horizontal_line = 5
    # vertical_line = 6


class DataGenerator:
    """Class used for generating custom datasets for classification tasks.

    This class is strongly coupled with the Dataset class in this package (data_utils/dataset.py).
    """

    def __init__(self,
                 n_samples: int = 1000,
                 image_dim: int = 50,
                 noise_level: float = 0.0,
                 shape_ratio_range: tuple[float, float] = (0.1, 1.0),
                 split_ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
                 centered: bool = False,
                 ):
        """
        :param n_samples: Number of samples that should be generated.
        :param image_dim: Dimensions of the square image generated. E.g. 50x50 image.
        :param noise_level: Level of noise in image. 0.0: no noise, 1.0: pixels have 50% chance of being activated.
        :param shape_ratio_range: Range for how large the shape can be relative to the whole image.
        :param split_ratios: Split used for train, val, and test set.
        :param centered: If set to True, the shape is centered in the image.
        """

        self._image_dim: int = image_dim
        self._noise_level: float = noise_level
        self._shape_ratio_range: tuple[float, float] = shape_ratio_range
        self._split: tuple[float, float, float] = self._get_split(n_samples, split_ratios)
        self._centered: bool = centered

    def _get_split(self, n_samples: int, split_ratios: tuple[float, float, float]) -> tuple[float, float, float]:
        """Converts a tuple of split ratios to a tuple of samples in each split.

        :param n_samples: Total number of samples.
        :param split_ratios: Tuple of split ratios.
        :return: Tuple of samples in each split.
        """

        split = [int(x * n_samples) for x in split_ratios]  # Split n_samples into partitions weighted by split_ratios
        split[0] += n_samples - sum(split)  # Add remaining samples to training set (can be 1 leftover)
        return tuple(split)

    def _place_shape(self, image: np.ndarray, shape: Shapes) -> None:
        """Places shape in image.

        :param image: Image array where shape should be placed. The array is modified in place.
        :param shape: Shape to be placed.
        """

        # Randomly assign shape_size
        low: int = round(self._shape_ratio_range[0] * self._image_dim)
        high: int = round(self._shape_ratio_range[1] * self._image_dim)
        shape_size: int = np.random.randint(low, high) if low != high else high

        # Force shape_size to be at least 7 to make distinct shapes (needed for circle)
        if shape_size < 7:
            shape_size = 7

        # Select bounding box for shape to be in
        if self._image_dim == shape_size:
            x1 = y1 = 0
            x2 = y2 = shape_size - 1
        elif self._centered:
            x1 = y1 = round((self._image_dim - shape_size) / 2)
            x2 = y2 = round((self._image_dim + shape_size) / 2)
        else:
            x1 = np.random.randint(0, self._image_dim - shape_size)
            x2 = x1 + shape_size
            y1 = np.random.randint(0, self._image_dim - shape_size)
            y2 = y1 + shape_size

        if shape == Shapes.square:
            image[y1:y2 + 1, x1] = 1  # Upper line
            image[y1:y2 + 1, x2] = 1  # Lower line
            image[y1, x1:x2 + 1] = 1  # Leftmost line
            image[y2, x1:x2 + 1] = 1  # Rightmost line
        elif shape == Shapes.filled_square:
            image[y1:y2, x1:x2] = 1  # Whole Square
        elif shape == Shapes.cross:
            # Enforces odd number of pixels in each dimension of bounding box
            # This makes sure the cross is symmetric
            if (y2 - y1) % 2:
                y2 -= 1
            if (x2 - x1) % 2:
                x2 -= 1
            image[y1:y2 + 1, round(np.mean([x1, x2]))] = 1  # Vertical line
            image[round(np.mean([y1, y2])), x1:x2 + 1] = 1  # Horizontal line
        # elif shape == Shapes.vertical_line:
        #     image[y1:y2 + 1, round(np.mean([x1, x2]))] = 1
        # elif shape == Shapes.horizontal_line:
        #     image[round(np.mean([y1, y2])), x1:x2 + 1] = 1
        elif shape == Shapes.circle:
            self._generate_circle(image[y1:y2, x1:x2], filled=False)
        elif shape == Shapes.filled_circle:
            self._generate_circle(image[y1:y2, x1:x2], filled=True)

    def _generate_circle(self, bounding_box: np.ndarray, filled: bool) -> None:
        """Utility function for generating circle in bounding box.

        :param bounding_box: Bounding box to fill.
        :param filled: Whether or not the circle should be filled
        """

        radius = bounding_box.shape[0] // 2 - 1
        a = np.arange(-radius, radius + 1) ** 2
        dists = np.sqrt(a[:, None] + a)

        if filled:
            b = (dists - radius < 0.5).astype(int)
        else:
            b = (np.abs(dists - radius) < 0.5).astype(int)
        bounding_box[:b.shape[0], :b.shape[1]] = b

    def _apply_noise(self, image: np.ndarray):
        """Adds noise to the image.

        :param image: Image array that noise should be added to. The array is modified in place.
        """

        noise_arr = np.random.choice([0, 1], size=(self._image_dim, self._image_dim),
                                     p=[1 - (self._noise_level / 2), (self._noise_level / 2)])
        idx = np.where(noise_arr == 1)
        image[idx] = 1 - image[idx]

    def _generate_image(self, shape: Shapes = Shapes.square) -> np.ndarray:
        """Generates a new image containing a given shape.

        :param shape: Shape to be placed in the image.
        :return: The generated image in the form of a numpy array.
        """

        image = np.zeros((self._image_dim, self._image_dim))

        self._place_shape(image, shape)
        self._apply_noise(image)

        return image

    def populate_dataset(self, dataset: Dataset) -> None:
        """Populates an existing dataset with generated images.

        This can be used to add data from multiple generators into a single dataset. The images are appended to the
        end of the sets in the dataset. The dataset is modified in place.

        :param dataset: The dataset to be populated.
        """

        # Loop through partitions (train, val, test)
        for i, partition_name in enumerate(['train', 'val', 'test']):
            partition = getattr(dataset, partition_name)

            # Append images to partition
            for n in range(self._split[i]):
                shape = Shapes(n % len(Shapes))
                generated_image = self._generate_image(shape=shape)
                label = shape.value
                datapoint = DataPoint(generated_image, label)
                partition.append(datapoint)

    def generate_dataset(self) -> Dataset:
        """Generates and returns a new dataset.

        This can be used if no existing dataset exists.
        """

        dataset = Dataset(labels=Shapes)
        self.populate_dataset(dataset)
        dataset.shuffle_partitions()
        dataset.shuffle_partitions()
        return dataset


if __name__ == '__main__':
    dataset = DataGenerator(
        image_dim=20,
        noise_level=0,
        #shape_ratio_range=[0.5, 0.5],
        split_ratios=[1, 0, 0],
        n_samples=len(Shapes)
    ).generate_dataset()
    dataset.visualize_data('train', n_samples=len(Shapes))
