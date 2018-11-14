# FILENAME: model.py
# AUTHORS: Jalocha, Kozlowski, Piekarski

import numpy as np


class Model:

    def __init__(self, img):
        self.alpha = 1.0
        self.beta = 2.0
        self.variance_matrix = self.calculate_pix_variance(img) / (self.calculate_pix_variance(img).max() or 1.0)
        self.number_of_ants = 512
        self.evaporation_rate = 0.05
        self.pheromone_decay = 0.005
        self.max_iter = 300
        self.epsilon = 0.01

    @staticmethod
    def calculate_target_fcn(in_image, ref_image):

        if in_image.shape != ref_image.shape:
            raise ValueError('images must have the same shape')

        TP = TN = FP = FN = 0
        for i in range(0, ref_image.shape[0]):
            for j in range(0, ref_image.shape[1]):
                if ref_image[i][j] != 0 and in_image[i][j] != 0:
                    TP += 1
                elif ref_image[i][j] != 0 and in_image[i][j] == 0:
                    FN += 1
                elif ref_image[i][j] == 0 and in_image[i][j] == 0:
                    TN += 1
                elif ref_image[i][j] == 0 and in_image[i][j] != 0:
                    FP += 1
        return TP / (TP + FP + FN)

    @staticmethod
    def calculate_pix_variance(image):
        variance_matrix = np.zeros((image.shape[0], image.shape[1]))
        for i in range(1, image.shape[0] - 1):  # edges to be considered later
            for j in range(1, image.shape[1] - 1):
                var1 = np.abs(int(image[i + 1][j]) - int(image[i - 1][j]))
                var2 = np.abs(int(image[i][j + 1]) - int(image[i][j - 1]))
                var3 = np.abs(int(image[i - 1][j - 1]) - int(image[i + 1][j + 1]))
                var4 = np.abs(int(image[i - 1][j + 1]) - int(image[i + 1][j - 1]))
                variance_matrix[i][j] = var1 + var2 + var3 + var4
        return variance_matrix

    @staticmethod
    def get_random_indices(arr, k):
        """Sample k random indices of an array without replacement.

        Returns a k by n array, where n = arr.ndim
        """

        linear_indices = np.random.choice(arr.size, k, replace=False)
        coordinates = np.unravel_index(linear_indices, arr.shape)
        array_of_coordinates = np.stack(tuple(coordinates), 1)
        return array_of_coordinates

    @staticmethod
    def get_neighbours(position, arr_shape):
        """Return valid 8-connected neighbours of a given point in an array.

        Positions are returned as a list of tuples.
        """

        offsets = (-1, 0, 1)
        offsets_2d = ((y, x) for y in offsets for x in offsets
                      if y != 0 or x != 0)

        all_neighbours = ((position[0] + y, position[1] + x) for (y, x) in offsets_2d)

        valid_neighbours = [(y, x) for (y, x) in all_neighbours
                            if (0 <= y < arr_shape[0]) and (0 <= x < arr_shape[1])]

        return valid_neighbours
