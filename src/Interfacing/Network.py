from typing import Callable, Optional

import numpy as np
from ale_py import Action

from src.Genetics.GeneticsObjects import Chromosome
from src.constants import OUT_MAP


def leaky_relu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, x * 0.001)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum((0, x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0/(1.0 + np.exp(-x))


def linear(x: np.ndarray) -> np.ndarray:
    return x


class NeuralNetwork:

    """Simple implementation of a neural network in Numpy"""

    def __init__(
            self,
            layer_nodes: list,
            hidden_activation: Callable[[np.ndarray], np.ndarray],
            output_activation: Callable[[np.ndarray], np.ndarray],
            chromosome: Optional[Chromosome] = None,
            seed: Optional[int] = 101,
    ):

        self.weights = {}
        self.biases = {}
        self.seed = seed
        self.layer_nodes = layer_nodes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.chromosome = chromosome
        self.rand = np.random.RandomState(seed)

        if chromosome is not None:
            self.weights = chromosome.weights
            self.biases = chromosome.biases
        else:
            for layer in range(1, len(self.layer_nodes)):  # layer 0 is input layer
                self.weights[layer] = np.random.uniform(-1, 1, size=(self.layer_nodes[layer], self.layer_nodes[layer-1]))
                self.biases[layer] = np.random.uniform(-1, 1, size=(self.layer_nodes[layer], 1))
                self.activations[layer] = None
                self.chromosome = Chromosome(
                    weights=self.weights,
                    biases=self.biases
                )

    def feed_forward(self, x: np.ndarray) -> np.ndarray:
        a_prev = x[:, None]

        L = len(self.layer_nodes) - 1
        for l in range(1, L):
            W_curr = self.weights[l]
            b_curr = self.biases[l]
            z = np.dot(W_curr, a_prev) + b_curr
            a_curr = self.hidden_activation(z)
            a_prev = a_curr

        W = self.weights[L]
        b_curr = self.biases[L]
        z = np.dot(W, a_prev) + b_curr
        out = self.output_activation(z)
        return out

    def get_action(self, state: np.ndarray) -> Action:
        """Passes state into initialised network, and returns a given output"""
        if state.shape[0] != self.layer_nodes[0]:
            raise Exception("Invalid input shape: does not match expected input shape of network")

        ff_input = np.dot(state, 1/255)
        ff_output = self.feed_forward(ff_input)
        return OUT_MAP[np.argmax(ff_output.flatten())]

    def set_chromosome(self, chromosome: Chromosome):
        self.chromosome = chromosome
        self.weights = chromosome.weights
        self.biases = chromosome.biases

    def get_chromosome(self):
        return self.chromosome
