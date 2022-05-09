from unittest.mock import patch

from numpy import ndarray, array, array_equal

from src.Genetics.GeneticsObjects import Chromosome, SimulatedBinaryCrossoverHandler


class TestCrossoverMethod:

    def test_basic_crossover_works_as_expected(self):

        weights_a = {1: array([1,2,3,4]), 2: array([1,2,3,4])}
        weights_b = {1: array([-1,-2,-3,-4]), 2: array([0,0,0,0])}

        biases_a = {1: array([0,2,0,4]), 2: array([0,2,0,4])}
        biases_b = {1: array([0,-2,-3, 0]), 2: array([0,1,0,1])}

        parent_a = Chromosome(weights_a, biases_a)
        parent_b = Chromosome(weights_b, biases_b)

        with patch("numpy.random.random", return_value = array([0, 0, 0, 0])):
            crossover = SimulatedBinaryCrossoverHandler(20)
            chr_c, chr_d = crossover.crossover(parent_a, parent_b)

        expected_chromosome_c = Chromosome(
            {1: array([0, 0, 0, 0]), 2: array([0.5, 1, 1.5, 2])},
            {1: array([0, 0, -1.5, 2]), 2: array([0, 1.5, 0, 2.5])}
        )

        expected_chromosome_d = Chromosome(
            {1: array([0, 0, 0, 0]), 2: array([0.5, 1, 1.5, 2])},
            {1: array([0, 0, -1.5, 2]), 2: array([0, 1.5, 0, 2.5])}
        )

        for i in weights_a.keys():
            assert array_equal(chr_c.weights[i], expected_chromosome_c.weights[i])
            assert array_equal(chr_d.weights[i], expected_chromosome_d.weights[i])

        for j in biases_a.keys():
            assert array_equal(chr_c.biases[j], expected_chromosome_c.biases[j])
            assert array_equal(chr_d.biases[j], expected_chromosome_d.biases[j])

