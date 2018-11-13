# FILENAME: main.py
# AUTHORS: Jalocha, Kozlowski, Piekarski
import random

import cv2
import numpy as np


class AlgorithmParameters:
    def __init__(self, img):
        self.alpha = 1.0
        self.beta = 2.0
        self.variance_matrix = calculate_pix_variance(img) / (calculate_pix_variance(img).max() or 1.0)
        self.init_pheromones = init_pheromone_matrix(img.shape, 0.1)
        self.number_of_ants = 512
        self.evaporation_rate = 0.05  # poprawiamy zbieznosc
        self.pheromone_decay = 0.005
        self.max_iter = 300
        self.epsilon = 0.01


def read_image(filepath):
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)


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


def move_ants(ant_matrix, pheromone_matrix, variance_matrix, alpha, beta):
    for i, ant in enumerate(ant_matrix):
        ant_matrix[i] = get_next_move(ant, pheromone_matrix, variance_matrix, alpha, beta)


def init_pheromone_matrix(image_shape, initial_value):
    """Return initial pheromone matrix filled with constant value.

    """
    pheromone_matrix = np.full((image_shape[0], image_shape[1]), initial_value)

    return pheromone_matrix


def pheromone_matrix_update(pheromone_matrix, ant_positions, heuristic_matrix, evaporation_rate):
    """Return updated pheromone matrix after ants movement.

    After every step, the pheromone values are updated. Evaporation rate controls the degree of the
    updating of matrix.
    """
    for ant in ant_positions:
        pheromone_matrix[ant[0], ant[1]] = (1 - evaporation_rate) * pheromone_matrix[ant[0], ant[1]] + \
            evaporation_rate * heuristic_matrix[ant[0], ant[1]]


def pheromone_matrix_decay(pheromone_matrix, initial_pheromone_matrix, pheromone_decay):
    """ Return decayed pheromone matrix.

    After the movement of all ants, the pheromone matrix is updated.
    """
    for i in range(0, pheromone_matrix.shape[0]):
        for j in range(0, pheromone_matrix.shape[1]):
            pheromone_matrix[i][j] = (1 - pheromone_decay) * pheromone_matrix[i][j] + pheromone_decay * \
                                     initial_pheromone_matrix[i][j]


def calculate_threshold(pheromone_matrix, epsilon):
    """ Return a calculated threshold.

    """
    threshold = np.mean(pheromone_matrix)
    i = 0
    while True:
        mi1 = np.mean(pheromone_matrix[pheromone_matrix >= threshold])
        mi2 = np.mean(pheromone_matrix[pheromone_matrix < threshold])
        new_threshold = (mi1 + mi2)/2
        if (abs(threshold - new_threshold) < epsilon) or (i > 1000):
            break
        i = i + 1
    return new_threshold


def determine_edges(pheromone_matrix, epsilon):
    im_edges = np.zeros(pheromone_matrix.shape, np.uint8)
    thresh = calculate_threshold(pheromone_matrix, epsilon)
    for i in range(0, pheromone_matrix.shape[0]):
        for j in range(0, pheromone_matrix.shape[1]):
            if pheromone_matrix[i][j] >= thresh:
                im_edges[i][j] = 255

    return im_edges


def run_one_iteration(ant_matrix, pheromone_matrix, algorithm_parameters):
    move_ants(ant_matrix, pheromone_matrix, algorithm_parameters.variance_matrix,
              algorithm_parameters.alpha, algorithm_parameters.beta)
    pheromone_matrix_update(pheromone_matrix, ant_matrix,
                            algorithm_parameters.variance_matrix, algorithm_parameters.evaporation_rate)
    pheromone_matrix_decay(pheromone_matrix, algorithm_parameters.init_pheromones, algorithm_parameters.pheromone_decay)


def run_algorithm(im_in, im_ref, parameters, is_verbose):
    assert(im_in.shape == im_ref.shape)

    ant_matrix = get_random_indices(im_in, parameters.number_of_ants)
    pheromone_matrix = parameters.init_pheromones.copy()
    best_target_fcn = 0
    im_out = np.zeros(im_in.shape, np.uint8)

    for i in range(0, parameters.max_iter):
        run_one_iteration(ant_matrix, pheromone_matrix, parameters)
        im_temp = determine_edges(pheromone_matrix, parameters.epsilon)
        target_fcn = calculate_target_fcn(im_temp, im_ref)

        if target_fcn > best_target_fcn:
            best_target_fcn = target_fcn
            im_out = im_temp
            if is_verbose:
                print(f'Iter: {i}, target_fcn = {best_target_fcn}')

        if is_verbose and i % 50 == 0:
            print(f'Iteration no: {i}')

    return im_out, pheromone_matrix


def main():

    in_image_path = 'input_data/house_in.png'
    ref_image_path = 'input_data/house_prewitt.png'
    img_in = read_image(in_image_path)
    img_ref = read_image(ref_image_path)
    parameters = AlgorithmParameters(img_in)
    img_edge, pheromone = run_algorithm(img_in, img_ref, parameters, True)
    print(f'Target fun: {calculate_target_fcn(img_edge, img_ref)}')
    cv2.imshow('Im edge', img_edge)
    cv2.imshow('Ref', img_ref)
    cv2.imshow('Pheromone', pheromone / pheromone.max())
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
