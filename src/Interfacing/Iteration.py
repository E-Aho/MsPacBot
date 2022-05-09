import pickle
from pathlib import Path

import ale_py.roms as roms
from ale_py import ALEInterface

from src.Genetics.GeneticsObjects import SimulatedBinaryCrossoverHandler
from src.Genetics.Methods import GaussianMutator, get_next_generation, spawn_population
from src.constants import ITER_DEPTH, POP_SIZE, OUT_MAP, P_MUTATE, ETA, PARENT_POP_SIZE, TOURNAMENT_SIZE, AGE_LIMIT

class Config:
    def __init__(self,
                 population_size: int = POP_SIZE,
                 parent_pop_size: int = PARENT_POP_SIZE,
                 tournament_size: int = TOURNAMENT_SIZE,
                 iteration_depth: int = ITER_DEPTH,
                 age_limit: int = AGE_LIMIT,
                 p_mutate: float = P_MUTATE,
                 eta: int = ETA,
                 do_save_best_individual: bool = False,
                 iteration_name: str = ""
                 ):
        self.iteration_depth = iteration_depth
        self.population_size = population_size
        self.parent_pop_size = parent_pop_size
        self.tournament_size = tournament_size
        self.age_limit = age_limit
        self.p_mutate = p_mutate
        self.eta = eta
        self.do_save_best_individual = do_save_best_individual
        self.iteration_name = iteration_name

    def to_dict(self):
        return self.__dict__


def main_iteration(
        population_size: int = POP_SIZE,
        parent_pop_size: int = PARENT_POP_SIZE,
        tournament_size: int = TOURNAMENT_SIZE,
        age_limit: int = AGE_LIMIT,
        p_mutate: float = P_MUTATE,
        eta: int = ETA,
        iteration_depth: int = ITER_DEPTH,
        do_save_best_individual: bool = False,
        iteration_name: str = ""
):

    ale = ALEInterface()
    ale.setInt('random_seed', 101)
    ale.setBool('display_screen', False)
    ale.setBool('sound', False)
    ale.loadROM(roms.MsPacman)

    input_dim = ale.getRAM().shape[0]
    output_dim = len(OUT_MAP.keys())

    architecture = [input_dim, 64, 32, output_dim]

    best_scores = []
    mean_score = []
    st_dev = []
    all_scores = []

    initial_pop = spawn_population(
        architecture=architecture,
        ale_env=ale,
        population_size=population_size,
    )

    pop = initial_pop
    all_best_score = 0
    best_individual = None

    for generation in range(iteration_depth + 1):
        print(f"Computing scores for gen {generation}...")
        fitnesses = pop.get_fitnesses()
        best_score = pop.get_fittest_individual().get_fitness()
        print(f"Best score for gen {generation}: {best_score}")
        if best_score > all_best_score:
            best_individual = pop.get_fittest_individual()
            best_score = best_score

        all_scores.append(fitnesses)
        best_scores.append(best_score)
        st_dev.append(pop.get_fitness_stdev())
        mean_score.append(pop.get_average_fitness())

        pop = get_next_generation(
            population=pop,
            mutation_function=GaussianMutator(p_mutate=p_mutate),
            crossover_handler=SimulatedBinaryCrossoverHandler(eta=eta),
            pop_size=population_size,
            age_limit=age_limit,
            parent_pop_size=parent_pop_size,
            tournament_size=tournament_size,
        )

    output = {
        "best_scores": best_scores,
        "mean_scores": mean_score,
        "st_devs": st_dev,
        "conf": {
            "pop_size": population_size,
            "parent_pop_size": parent_pop_size,
            "tournament_size": tournament_size,
            "age_limit": age_limit,
            "p_mutate": p_mutate,
            "eta": eta,
        }
    }

    if do_save_best_individual:
        with open(Path("individuals")/iteration_name, "wb") as file:
            pickle.dump(best_individual.chromosome, file)
    return output


if __name__ == "__main__":
    main_iteration()
