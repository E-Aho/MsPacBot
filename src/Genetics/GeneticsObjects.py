from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Chromosome:
    def __init__(self, weights: dict, biases: dict):
        self.weights = weights
        self.biases = biases

    def __eq__(self, other):
        try:
            if self.weights == other.weights and self.biases == other.biases:
                return True
            return False
        except Exception:
            return False


class MutationMethod(ABC):

    @abstractmethod
    def _apply_mutation_(self, arr: np.ndarray) -> np.ndarray:
        pass

    def mutate(self, chromosome: Chromosome) -> None:
        for i in chromosome.weights.keys():
            chromosome.weights[i] = self._apply_mutation_(chromosome.weights[i])

        for j in chromosome.biases.keys():
            chromosome.biases[j] = self._apply_mutation_(chromosome.biases[j])


class CrossoverHandler(ABC):

    @abstractmethod
    def _apply_crossover_(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def crossover(self, parent_a: Chromosome, parent_b: Chromosome) -> Tuple[Chromosome, Chromosome]:
        weight_a = {}
        weight_b = {}

        bias_a = {}
        bias_b = {}

        for i in parent_a.weights.keys():
            w_a, w_b = self._apply_crossover_(parent_a.weights[i], parent_b.weights[i])
            weight_a[i] = w_a
            weight_b[i] = w_b

        for j in parent_a.biases.keys():
            b_a, b_b = self._apply_crossover_(parent_a.biases[j], parent_b.biases[j])
            bias_a[j] = b_a
            bias_b[j] = b_b

        return (
            Chromosome(weights=weight_a, biases=bias_a),
            Chromosome(weights=weight_b, biases=bias_b)
        )


class SimulatedBinaryCrossoverHandler(CrossoverHandler):
    def __init__(self, eta: float):
        self.eta = eta

    def _apply_crossover_(self, arr_a: np.ndarray, arr_b: np.ndarray):
        parent_shape = arr_a.shape

        mu = np.random.random(parent_shape)
        beta = np.empty(parent_shape)

        beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1.0/(self.eta + 1))
        beta[mu > 0.5] = (1.0 / (2.0 * (1.0 - mu[mu > 0.5]))) ** (1.0/(self.eta + 1))

        child_a = 0.5 * ((1 + beta) * arr_a + (1 - beta) * arr_b)
        child_b = 0.5 * ((1 - beta) * arr_a + (1 + beta) * arr_b)

        return child_a, child_b