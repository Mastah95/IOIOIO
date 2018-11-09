import unittest
from itertools import product

import numpy as np

from main import get_neighbours, get_next_move


class TestGetNeighbours(unittest.TestCase):
    def test_finds_neighbourhood(self):
        matrix_shape = (10, 20)
        position = (5, 7)

        ys = (-1, 0, 1)
        xs = (-1, 0, 1)
        offsets = set(product(ys, xs)) - {(0, 0)}
        expected = ((position[0] + y, position[1] + x) for (y, x) in offsets)
        actual = get_neighbours(position, matrix_shape)
        self.assertSetEqual(set(expected), set(actual))

    def test_finds_edge_neighbourhood(self):
        matrix_shape = (7, 10)
        position = (5, 9)

        h, w = matrix_shape
        matrix_positions = set(product(range(h), range(w)))
        actual = set(get_neighbours(position, matrix_shape))

        self.assertTrue(actual.issubset(matrix_positions))
        self.assertEqual(len(actual), 5)

    def test_finds_corner_neighbourhood(self):
        matrix_shape = (7, 10)
        position = (6, 9)

        h, w = matrix_shape
        matrix_positions = set(product(range(h), range(w)))
        actual = set(get_neighbours(position, matrix_shape))

        self.assertTrue(actual.issubset(matrix_positions))
        self.assertEqual(len(actual), 3)

    def test_returns_empty_result_out_of_range(self):
        matrix_shape = (7, 10)
        position = (777, 999)

        actual = set(get_neighbours(position, matrix_shape))

        self.assertEqual(len(actual), 0)


class TestGetNextMove(unittest.TestCase):
    def test_is_valid_move(self):
        position = (5, 6)
        shape = (8, 10)
        pheromones = np.random.rand(*shape)
        variances = np.random.randint(0, 100, shape)
        alpha = 1.0
        beta = 1.0

        ys = (-1, 0, 1)
        xs = (-1, 0, 1)
        offsets = set(product(ys, xs)) - {(0, 0)}
        valid_moves = set((position[0] + y, position[1] + x)
                          for (y, x) in offsets)

        for _ in range(20):
            move = get_next_move(position, pheromones, variances, alpha, beta)
            self.assertTrue(move in valid_moves)

    def test_handles_zero_variances(self):
        """Check that results are provided if input variances are all zeros."""
        position = (5, 6)
        shape = (8, 10)
        pheromones = np.zeros(shape)
        variances = np.zeros_like(pheromones)
        alpha = 1.0
        beta = 1.0

        move = get_next_move(position, pheromones, variances, alpha, beta)
        self.assertIsNotNone(move)
