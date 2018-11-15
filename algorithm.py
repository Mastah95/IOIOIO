# FILENAME: algorithm.py
# AUTHORS: Jalocha, Kozlowski, Piekarski

import numpy as np
import random
import model


class Algorithm:

    def __init__(self, in_image, ref_image, model_parameters):
        self.model = model.Model(in_image, ref_image, model_parameters)
        self.init_pheromones = self.init_pheromone_matrix(in_image.shape, self.model.initial_pheromone)
        self.pheromone_matrix = self.init_pheromones.copy()
        self.ant_matrix = self.model.get_random_indices()

    def get_transition_weight(self, position, alpha, beta, variance_matrix):
        """Get weight used to determine ant transition probabilities.

        Probability of a pixel being selected as the transition target increases
        with its variance and pheromone level.
        """

        pheromone = self.pheromone_matrix[position]
        visibility = variance_matrix[position]

        return pow(pheromone, alpha) * pow(visibility, beta)

    def get_next_move(self, position, variance_matrix, alpha, beta):
        """Return the next position that an ant should visit.

        A pixel from the neighbourhood is drawn using weighted random sampling.
        Probabilities are determined by pheromone and variance values at the
        respective positions.
        """

        neighbours = self.model.get_neighbours(position, self.pheromone_matrix.shape)
        weights = [self.get_transition_weight(pos, alpha, beta, variance_matrix)
                   for pos in neighbours]

        try:
            neighbour = random.choices(neighbours, weights)
        except IndexError:
            neighbour = random.choices(neighbours)

        return neighbour[0]

    def move_ants(self, variance_matrix, alpha, beta):
        for i, ant in enumerate(self.ant_matrix):
            self.ant_matrix[i] = self.get_next_move(ant, variance_matrix, alpha, beta)

    @staticmethod
    def init_pheromone_matrix(image_shape, initial_value):
        """Return initial pheromone matrix filled with constant value.

        """
        pheromone_matrix = np.full((image_shape[0], image_shape[1]), initial_value)

        return pheromone_matrix

    def pheromone_matrix_update(self, heuristic_matrix, evaporation_rate):
        """Return updated pheromone matrix after ants movement.

        After every step, the pheromone values are updated. Evaporation rate controls the degree of the
        updating of matrix.
        """
        for ant in self.ant_matrix:
            self.pheromone_matrix[ant[0], ant[1]] = (1 - evaporation_rate) * self.pheromone_matrix[ant[0], ant[1]] + \
                                               evaporation_rate * heuristic_matrix[ant[0], ant[1]]

    def pheromone_matrix_decay(self, pheromone_decay):
        """ Return decayed pheromone matrix.

        After the movement of all ants, the pheromone matrix is updated.
        """
        for i in range(0, self.pheromone_matrix.shape[0]):
            for j in range(0, self.pheromone_matrix.shape[1]):
                self.pheromone_matrix[i][j] = (1 - pheromone_decay) * self.pheromone_matrix[i][j] + pheromone_decay * \
                                         self.init_pheromones[i][j]

    def calculate_threshold(self, epsilon):
        """ Return a calculated threshold.

        """
        threshold = np.mean(self.pheromone_matrix)
        i = 0
        while True:
            mi1 = np.mean(self.pheromone_matrix[self.pheromone_matrix >= threshold])
            mi2 = np.mean(self.pheromone_matrix[self.pheromone_matrix < threshold])
            new_threshold = (mi1 + mi2) / 2
            if (abs(threshold - new_threshold) < epsilon) or (i > 1000):
                break
            i = i + 1
        return new_threshold

    def determine_edges(self, epsilon):
        im_edges = np.zeros(self.pheromone_matrix.shape, np.uint8)
        thresh = self.calculate_threshold(epsilon)
        for i in range(0, self.pheromone_matrix.shape[0]):
            for j in range(0, self.pheromone_matrix.shape[1]):
                if self.pheromone_matrix[i][j] >= thresh:
                    im_edges[i][j] = 255

        return im_edges

    def run_one_iteration(self):
        self.move_ants(self.model.variance_matrix, self.model.alpha, self.model.beta)
        self.pheromone_matrix_update(self.model.variance_matrix, self.model.evaporation_rate)
        self.pheromone_matrix_decay(self.model.pheromone_decay)

    def run_algorithm(self, is_verbose):
        assert (self.model.in_image.shape == self.model.ref_image.shape)

        best_target_fcn = 0
        im_out = np.zeros(self.model.in_image.shape, np.uint8)

        for i in range(0, self.model.max_iter):
            self.run_one_iteration()
            im_temp = self.determine_edges(self.model.epsilon)
            target_fcn = self.model.calculate_target_fcn(im_temp)

            if target_fcn > best_target_fcn:
                best_target_fcn = target_fcn
                im_out = im_temp
                if is_verbose:
                    print(f'Iter: {i}, target_fcn = {best_target_fcn}')

            if is_verbose and i % 50 == 0:
                print(f'Iteration no: {i}')

        return im_out
