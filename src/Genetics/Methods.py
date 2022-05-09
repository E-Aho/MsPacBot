from abc import ABC
from typing import List

import numpy as np
from ale_py import ALEInterface

from src.Genetics.GeneticsObjects import Chromosome, MutationMethod, CrossoverHandler, \
    SimulatedBinaryCrossoverHandler
from src.Genetics.Populations import Individual, Population
from src.Interfacing.Network import NeuralNetwork, leaky_relu, linear
from src.constants import POP_SIZE, TOURNAMENT_SIZE, PARENT_POP_SIZE, AGE_LIMIT


def spawn_random_chromosome(architecture: List[int]) -> Chromosome:
    weights = {}
    biases = {}
    for layer in range(1, len(architecture)):
        weights[layer] = np.random.uniform(-1, 1, size=(architecture[layer], architecture[layer-1]))
        biases[layer] = np.random.uniform(-1, 1, size=(architecture[layer], 1))
    return Chromosome(
        weights=weights,
        biases=biases,
    )


class GaussianMutator(MutationMethod, ABC):

    """Mutates array with new distribution from N(0,1)"""

    def __init__(self, p_mutate: float):
        self.p_mutate = p_mutate

    def _apply_mutation_(self, arr: np.ndarray) -> np.ndarray:
        does_mutate_array = np.random.random(arr.shape) < self.p_mutate
        gaussian_mutation_values = np.random.normal(size=arr.shape)
        arr[does_mutate_array] += gaussian_mutation_values[does_mutate_array]  # TODO: Deepcopy or mutate in place here?
        return arr


def elitism_selection(population: Population, n_individuals: int = PARENT_POP_SIZE) -> List[Individual]:
    sorted_individuals = sorted(population.individuals, key=lambda individual: individual.get_fitness(), reverse=True)
    return sorted_individuals[:n_individuals]


def tournament_selection(population: Population, n_to_select: int, tournament_size: int = TOURNAMENT_SIZE) -> List[Individual]:
    selected_population = []
    unselected_pop = [individual for individual in population.individuals]
    for _ in range(n_to_select):
        unselected_indexes = list(range(len(unselected_pop)))
        tournament_indexes = np.random.choice(unselected_indexes, tournament_size)
        best_index = max(tournament_indexes, key=lambda i: unselected_pop[i].get_fitness())
        best_individual = unselected_pop.pop(best_index)
        selected_population.append(best_individual)
    return selected_population


def get_next_generation(
        population: Population,
        mutation_function: MutationMethod,
        pop_size: int = POP_SIZE,
        crossover_handler: CrossoverHandler = SimulatedBinaryCrossoverHandler(0.2),
        include_parents: bool = True,
        tournament_size: int = TOURNAMENT_SIZE,
        age_limit: int = AGE_LIMIT,
        parent_pop_size: int = PARENT_POP_SIZE,
) -> Population:
    new_pop = []

    network_architecture = population.individuals[0].network
    env = population.individuals[0].ale_env

    population.increase_age()

    parent_pool = elitism_selection(population, n_individuals=parent_pop_size)
    parent_population = Population(parent_pool)

    for individual in parent_pool:  # Include parents in pool if they aren't too old
        if include_parents and individual.age < age_limit:
            new_pop.append(individual)

    while len(new_pop) < pop_size:  # Fill rest of the population with offspring of children

        parent_a, parent_b = tournament_selection(parent_population, n_to_select=2, tournament_size=tournament_size)
        chrom_a, chrom_b = crossover_handler.crossover(parent_a.chromosome, parent_b.chromosome)

        mutation_function.mutate(chrom_a)
        mutation_function.mutate(chrom_b)

        child_a = Individual(
            chromosome=chrom_a,
            network=NeuralNetwork(
                chromosome=chrom_a,
                layer_nodes=network_architecture.layer_nodes,
                hidden_activation=network_architecture.hidden_activation,
                output_activation=network_architecture.output_activation,
                seed=network_architecture.seed
            ),
            env=env
        )
        child_b = Individual(
            chromosome=chrom_b,
            network=NeuralNetwork(
                chromosome=chrom_b,
                layer_nodes=network_architecture.layer_nodes,
                hidden_activation=network_architecture.hidden_activation,
                output_activation=network_architecture.output_activation,
                seed=network_architecture.seed
            ),
            env=env
        )

        new_pop.append(child_a)
        new_pop.append(child_b)

    return Population(new_pop)


def spawn_population(
        architecture: List[int],
        ale_env: ALEInterface,
        population_size: int = POP_SIZE,
) -> Population:

    individuals = []
    for _ in range(population_size):
        chromosome = spawn_random_chromosome(architecture)
        network = NeuralNetwork(
            layer_nodes=architecture,
            hidden_activation=leaky_relu,
            output_activation=linear,
            chromosome=chromosome,
        )

        individuals.append(
            Individual(
                chromosome=chromosome,
                network=network,
                env=ale_env,
            )
        )

    return Population(individuals)
