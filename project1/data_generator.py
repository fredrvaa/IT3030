import numpy as np
import random

class DataGenerator():
    def __init__(self, n_samples=16, image_dim=50, noise_level=0.5, split=[0.7,0.2,0.1], shape_ratio_range=[0.1,1.0]):
        self.n_samples = n_samples
        self.image_dim = image_dim
        self.noise_level = noise_level
        self.split = split
        self.shape_ratio_range = shape_ratio_range

    def _place_shape(self, image, shape, shape_size):
        # Randomly select bounding box for shape to be in
        if self.image_dim == shape_size:
                x1 = y1 = 0
                x2 = y2 = shape_size - 1
        else:
            x1 = np.random.randint(0, self.image_dim - shape_size)
            x2 = x1 + shape_size
            y1 = np.random.randint(0, self.image_dim - shape_size)
            y2 = y1 + shape_size
        
        if shape == 'r': # Unfilled square
            image[x1:x2+1, y1] = 1 # Upper line
            image[x1:x2+1, y2] = 1 # Lower line
            image[x1, y1:y2+1] = 1 # Leftmost line
            image[x2, y1:y2+1] = 1 # Rightmost line   
        elif shape == 'fr': # Filled Square
            image[x1:x2, y1:y2] = 1 # Whole Square
        elif shape == 'x': # Cross
            image[x1:x2+1, round(np.mean([y1,y2]))] = 1 # Horizontal line
            image[round(np.mean([x1,x2])), y1:y2+1] = 1 # Vertical line
        elif shape == 'hl': # Horizontal line
            image[x1:x2+1, round(np.mean([y1,y2]))] = 1
        elif shape == 'vl': # Vertical line
            image[round(np.mean([x1,x2])), y1:y2+1] = 1 # Vertical line
            

    def _apply_noise(self, image):
        noise_arr = np.random.choice([0,1], size=(self.image_dim, self.image_dim), p=[1-self.noise_level, self.noise_level])
        idx = np.where(noise_arr == 1)
        image[idx] = 1 - image[idx]

    def _generate_image(self, shape='fr'):
        image = np.zeros((self.image_dim, self.image_dim))
        low = round(self.shape_ratio_range[0] * self.image_dim)
        high = round(self.shape_ratio_range[1] * self.image_dim)
        shape_size = np.random.randint(low, high) if low != high else high
        self._place_shape(image, shape, shape_size)
        self._apply_noise(image)

        return image
        

    def _generate_sets(self):
        pass        

if __name__=='__main__':
    import matplotlib.pyplot as plt
    g = DataGenerator(noise_level=0.005, shape_ratio_range=[0.005,1.0])
    fig = plt.figure()
    plt.imshow(g._generate_image())
    plt.show()
    