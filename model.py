# FILENAME: model.py
# AUTHORS: Jalocha, Kozlowski, Piekarski

import numpy as np


class Model:

    def __init__(self, in_image, ref_image, model_parameters):
        self.in_image = in_image
        self.ref_image = ref_image
        self.alpha = model_parameters["alpha"]
        self.beta = model_parameters["beta"]
        self.variance_matrix = self.calculate_pix_variance() / (self.calculate_pix_variance().max() or 1.0)
        self.number_of_ants = model_parameters["number_of_ants"]
        self.evaporation_rate = model_parameters["evaporation_rate"]
        self.pheromone_decay = model_parameters["pheromone_decay"]
        self.max_iter = model_parameters["max_iter"]
        self.construction_steps = model_parameters["construction_steps"]
        self.epsilon = model_parameters["epsilon"]
        self.initial_pheromone = model_parameters["initial_pheromone"]

    def calculate_target_fcn(self, image):

        if image.shape != self.ref_image.shape:
            raise ValueError('images must have the same shape')

        TP = TN = FP = FN = 0
        for i in range(0, self.ref_image.shape[0]):
            for j in range(0, self.ref_image.shape[1]):
                if self.ref_image[i][j] != 0 and image[i][j] != 0:
                    TP += 1
                elif self.ref_image[i][j] != 0 and image[i][j] == 0:
                    FN += 1
                elif self.ref_image[i][j] == 0 and image[i][j] == 0:
                    TN += 1
                elif self.ref_image[i][j] == 0 and image[i][j] != 0:
                    FP += 1
        return TP / (TP + FP + FN)

    def calculate_pix_variance(self):
        variance_matrix = np.zeros((self.in_image.shape[0], self.in_image.shape[1]))
        for i in range(1, self.in_image.shape[0] - 1):  # edges to be considered later
            for j in range(1, self.in_image.shape[1] - 1):
                var1 = np.abs(int(self.in_image[i + 1][j]) - int(self.in_image[i - 1][j]))
                var2 = np.abs(int(self.in_image[i][j + 1]) - int(self.in_image[i][j - 1]))
                var3 = np.abs(int(self.in_image[i - 1][j - 1]) - int(self.in_image[i + 1][j + 1]))
                var4 = np.abs(int(self.in_image[i - 1][j + 1]) - int(self.in_image[i + 1][j - 1]))
                variance_matrix[i][j] = var1 + var2 + var3 + var4
        return variance_matrix

    def get_random_indices(self):
        """Sample k random indices of an array without replacement.

        Returns a k by n array, where n = arr.ndim
        """

        linear_indices = np.random.choice(self.in_image.size, self.number_of_ants, replace=False)
        coordinates = np.unravel_index(linear_indices, self.in_image.shape)
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
