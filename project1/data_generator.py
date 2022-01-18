import numpy as np
from enum import Enum
from dataclasses import dataclass, field, fields


class Shapes(Enum):
    sqaure = 0
    filled_sqaure = 1
    cross = 2
    horizontal_line = 3
    vertical_line = 4

@dataclass
class DataPoint:
    image:np.ndarray
    label:int

@dataclass
class Dataset:
    train:list = field(default_factory=list)
    val:list = field(default_factory=list)
    test:list = field(default_factory=list)

class DataGenerator():
    def __init__(self, n_samples=1000, image_dim=50, noise_level=0.5, shape_ratio_range=[0.1,1.0], split_ratios=[0.7,0.2,0.1], centered=False):
        self._image_dim = image_dim
        self._noise_level = noise_level
        self._shape_ratio_range = shape_ratio_range
        self._split = self._get_split(n_samples, split_ratios)
        self._centered = centered

    def _get_split(self, n_samples, split_ratios):
        split = [int(x * n_samples) for x in split_ratios] # Split n_samples into partitions weighted by split_ratios
        split[0] += n_samples - sum(split) # Add remaining samples to training set (can be 1 leftover)
        return split

    def _place_shape(self, image, shape):
        # Randomly assign shape_size
        low = round(self._shape_ratio_range[0] * self._image_dim)
        high = round(self._shape_ratio_range[1] * self._image_dim)
        shape_size = np.random.randint(low, high) if low != high else high

        # Force shape_size to be at least 3 to make distinct shapes
        if shape_size < 3: shape_size = 3 

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
        
        if shape == Shapes.sqaure:
            image[y1:y2+1, x1] = 1 # Upper line
            image[y1:y2+1, x2] = 1 # Lower line
            image[y1, x1:x2+1] = 1 # Leftmost line
            image[y2, x1:x2+1] = 1 # Rightmost line   
        elif shape == Shapes.filled_sqaure:
            image[y1:y2, x1:x2] = 1 # Whole Square
        elif shape == Shapes.cross:
            image[y1:y2+1, round(np.mean([x1,x2]))] = 1 # Vertical line
            image[round(np.mean([y1,y2])), x1:x2+1] = 1 # Horizontal line
        elif shape == Shapes.vertical_line:
            image[y1:y2+1, round(np.mean([x1,x2]))] = 1
        elif shape == Shapes.horizontal_line:
            image[round(np.mean([y1,y2])), x1:x2+1] = 1
            

    def _apply_noise(self, image):
        noise_arr = np.random.choice([0,1], size=(self._image_dim, self._image_dim), p=[1-(self._noise_level/2), (self._noise_level/2)])
        idx = np.where(noise_arr == 1)
        image[idx] = 1 - image[idx]

    def _generate_image(self, shape=Shapes.sqaure):
        # Initialize image
        image = np.zeros((self._image_dim, self._image_dim))

        self._place_shape(image, shape)
        self._apply_noise(image)

        return image

    def populate_dataset(self, dataset):
        # Loop through partitions (train, val, test)
        for i, field in enumerate(fields(Dataset)):
            partition = getattr(dataset, field.name)

            # Construct partition
            for n in range(self._split[i]):
                shape = Shapes(n%len(Shapes))
                generated_image = self._generate_image(shape=shape)
                datapoint = DataPoint(generated_image, shape.value)
                partition.append(datapoint)

            # Shuffle partition
            np.random.shuffle(partition)


if __name__=='__main__':
    import matplotlib.pyplot as plt
    
    # Generate dataset
    dataset = Dataset()
    generator = DataGenerator(noise_level=0, shape_ratio_range=[0.005,1.0], n_samples=100)
    generator.populate_dataset(dataset)

    # Plot first 25 samples of train set
    n = 5
    fig, axs = plt.subplots(n,n,figsize=(9,9))
    for i in range(n):
        for j in range(n):
            image = dataset.train[i*n + j].image
            label = dataset.train[i*n + j].label
            axs[i][j].title.set_text(Shapes(label).name)
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
            axs[i][j].imshow(image)

    plt.show()
    