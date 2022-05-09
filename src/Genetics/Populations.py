import statistics
from functools import lru_cache
from typing import List

from ale_py import ALEInterface

from src.Genetics.GeneticsObjects import Chromosome
from src.Interfacing.Network import NeuralNetwork


class Individual:
    def __init__(self, chromosome: Chromosome, network: NeuralNetwork, env: ALEInterface):
        self.chromosome = chromosome
        self.network = network
        self.ale_env = env
        self.fitness = None
        self.age = 0

    def get_fitness(self):
        if self.fitness is not None:
            return self.fitness
        self.fitness = self.calculate_fitness()
        return self.fitness

    def calculate_fitness(self) -> int:
        total_reward = 0
        while not self.ale_env.game_over():
            game_state = self.ale_env.getRAM()
            action = self.network.get_action(game_state)
            total_reward += self.ale_env.act(action)
        self.ale_env.reset_game()
        return total_reward


class Population:

    def __init__(self, individuals: List[Individual]):
        self.individuals = individuals

    def population_size(self) -> int:
        return len(self.individuals)

    @lru_cache()
    def get_fitnesses(self):
        return [individual.get_fitness() for individual in self.individuals]

    def get_average_fitness(self) -> float:
        return sum(self.get_fitnesses()) / float(self.population_size())

    def get_fitness_stdev(self) -> float:
        return statistics.stdev([individual.get_fitness() for individual in self.individuals])

    def get_fittest_individual(self) -> Individual:
        return max(self.individuals, key=lambda i: i.get_fitness())

    def increase_age(self):
        for i in self.individuals:
            i.age += 1