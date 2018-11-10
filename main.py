# FILENAME: main.py
# AUTHORS: Jalocha, Kozlowski, Piekarski
import random

import cv2
import numpy as np


def read_image(filepath):
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)


def calculate_target_fcn(in_image_path, ref_image_path):
    in_image = read_image(in_image_path)
    ref_image = read_image(ref_image_path)

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


def calculate_pix_variance(image):
    variance_matrix = np.zeros((image.shape[0], image.shape[1]))
    for i in range(1, image.shape[0]-1):  # edges to be considered later
        for j in range(1, image.shape[1]-1):
            var1 = np.abs(int(image[i+1][j]) - int(image[i-1][j]))
            var2 = np.abs(int(image[i][j+1]) - int(image[i][j-1]))
            var3 = np.abs(int(image[i-1][j-1]) - int(image[i+1][j+1]))
            var4 = np.abs(int(image[i-1][j+1]) - int(image[i+1][j-1]))
            variance_matrix[i][j] = var1 + var2 + var3 + var4
    return variance_matrix


def get_random_indices(arr, k):
    """Sample k random indices of an array without replacement.

    Returns a k by n array, where n = arr.ndim
    """

    linear_indices = np.random.choice(arr.size, k, replace=False)
    coordinates = np.unravel_index(linear_indices, arr.shape)
    array_of_coordinates = np.stack(tuple(coordinates), 1)
    return array_of_coordinates


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


def get_transition_weight(position, alpha, beta, pheromone_matrix, variance_matrix):
    """Get weight used to determine ant transition probabilities.

    Probability of a pixel being selected as the transition target increases
    with its variance and pheromone level.
    """

    pheromone = pheromone_matrix[position]
    visibility = variance_matrix[position]

    return pow(pheromone, alpha) * pow(visibility, beta)


def get_next_move(position, pheromone_matrix, variance_matrix, alpha, beta):
    """Return the next position that an ant should visit.

    A pixel from the neighbourhood is drawn using weighted random sampling.
    Probabilities are determined by pheromone and variance values at the
    respective positions.
    """

    neighbours = get_neighbours(position, pheromone_matrix.shape)
    weights = [get_transition_weight(pos, alpha, beta, pheromone_matrix, variance_matrix)
               for pos in neighbours]

    try:
        neighbour = random.choices(neighbours, weights)
    except IndexError:
        neighbour = random.choices(neighbours)

    return neighbour[0]


def init_pheromone_matrix(image_shape, initial_value):
    pheromone_matrix = np.full((image_shape[0], image_shape[1]), initial_value)

    return pheromone_matrix


def main():
    in_image_path = 'input_data/house_prewitt.png'
    ref_image_path = 'input_data/house_prewitt.png'
    img = read_image(ref_image_path)
    cv2.imshow('test_window', img)
    cv2.waitKey(0)
    print(calculate_target_fcn(in_image_path, ref_image_path))
    variance = calculate_pix_variance(img)
    print(variance)

    ant_positions = get_random_indices(variance, 5)
    print(f'ant_positions: {ant_positions}')

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
