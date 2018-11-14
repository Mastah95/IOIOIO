# FILENAME: algorithm.py
# AUTHORS: Jalocha, Kozlowski, Piekarski

import numpy as np
import random
import model


class Algorithm:

    def __init__(self, in_image, ref_image):
        self.in_image = in_image
        self.ref_image = ref_image
        self.model = model.Model(in_image)
        self.init_pheromones = self.init_pheromone_matrix(in_image.shape, 0.1)

    @staticmethod
    def get_transition_weight(position, alpha, beta, pheromone_matrix, variance_matrix):
        """Get weight used to determine ant transition probabilities.

        Probability of a pixel being selected as the transition target increases
        with its variance and pheromone level.
        """

        pheromone = pheromone_matrix[position]
        visibility = variance_matrix[position]

        return pow(pheromone, alpha) * pow(visibility, beta)

    def get_next_move(self, position, pheromone_matrix, variance_matrix, alpha, beta):
        """Return the next position that an ant should visit.

        A pixel from the neighbourhood is drawn using weighted random sampling.
        Probabilities are determined by pheromone and variance values at the
        respective positions.
        """

        neighbours = self.model.get_neighbours(position, pheromone_matrix.shape)
        weights = [self.get_transition_weight(pos, alpha, beta, pheromone_matrix, variance_matrix)
                   for pos in neighbours]

        try:
            neighbour = random.choices(neighbours, weights)
        except IndexError:
            neighbour = random.choices(neighbours)

        return neighbour[0]

    def move_ants(self, ant_matrix, pheromone_matrix, variance_matrix, alpha, beta):
        for i, ant in enumerate(ant_matrix):
            ant_matrix[i] = self.get_next_move(ant, pheromone_matrix, variance_matrix, alpha, beta)

    @staticmethod
    def init_pheromone_matrix(image_shape, initial_value):
        """Return initial pheromone matrix filled with constant value.

        """
        pheromone_matrix = np.full((image_shape[0], image_shape[1]), initial_value)

        return pheromone_matrix

    @staticmethod
    def pheromone_matrix_update(pheromone_matrix, ant_positions, heuristic_matrix, evaporation_rate):
        """Return updated pheromone matrix after ants movement.

        After every step, the pheromone values are updated. Evaporation rate controls the degree of the
        updating of matrix.
        """
        for ant in ant_positions:
            pheromone_matrix[ant[0], ant[1]] = (1 - evaporation_rate) * pheromone_matrix[ant[0], ant[1]] + \
                                               evaporation_rate * heuristic_matrix[ant[0], ant[1]]

    @staticmethod
    def pheromone_matrix_decay(pheromone_matrix, initial_pheromone_matrix, pheromone_decay):
        """ Return decayed pheromone matrix.

        After the movement of all ants, the pheromone matrix is updated.
        """
        for i in range(0, pheromone_matrix.shape[0]):
            for j in range(0, pheromone_matrix.shape[1]):
                pheromone_matrix[i][j] = (1 - pheromone_decay) * pheromone_matrix[i][j] + pheromone_decay * \
                                         initial_pheromone_matrix[i][j]

    @staticmethod
    def calculate_threshold(pheromone_matrix, epsilon):
        """ Return a calculated threshold.

        """
        threshold = np.mean(pheromone_matrix)
        i = 0
        while True:
            mi1 = np.mean(pheromone_matrix[pheromone_matrix >= threshold])
            mi2 = np.mean(pheromone_matrix[pheromone_matrix < threshold])
            new_threshold = (mi1 + mi2) / 2
            if (abs(threshold - new_threshold) < epsilon) or (i > 1000):
                break
            i = i + 1
        return new_threshold

    def determine_edges(self, pheromone_matrix, epsilon):
        im_edges = np.zeros(pheromone_matrix.shape, np.uint8)
        thresh = self.calculate_threshold(pheromone_matrix, epsilon)
        for i in range(0, pheromone_matrix.shape[0]):
            for j in range(0, pheromone_matrix.shape[1]):
                if pheromone_matrix[i][j] >= thresh:
                    im_edges[i][j] = 255

        return im_edges

    def run_one_iteration(self, ant_matrix, pheromone_matrix):
        self.move_ants(ant_matrix, pheromone_matrix, self.model.variance_matrix,
                       self.model.alpha, self.model.beta)
        self.pheromone_matrix_update(pheromone_matrix, ant_matrix,
                                     self.model.variance_matrix, self.model.evaporation_rate)
        self.pheromone_matrix_decay(pheromone_matrix, self.init_pheromones,
                                    self.model.pheromone_decay)

    def run_algorithm(self, is_verbose):
        assert (self.in_image.shape == self.ref_image.shape)

        ant_matrix = self.model.get_random_indices(self.in_image, self.model.number_of_ants)
        pheromone_matrix = self.init_pheromones.copy()
        best_target_fcn = 0
        im_out = np.zeros(self.in_image.shape, np.uint8)

        for i in range(0, self.model.max_iter):
            self.run_one_iteration(ant_matrix, pheromone_matrix)
            im_temp = self.determine_edges(pheromone_matrix, self.model.epsilon)
            target_fcn = self.model.calculate_target_fcn(im_temp, self.ref_image)

            if target_fcn > best_target_fcn:
                best_target_fcn = target_fcn
                im_out = im_temp
                if is_verbose:
                    print(f'Iter: {i}, target_fcn = {best_target_fcn}')

            if is_verbose and i % 50 == 0:
                print(f'Iteration no: {i}')

        return im_out, pheromone_matrix
