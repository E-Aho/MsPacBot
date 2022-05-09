
from pathlib import Path

from src.Interfacing.Iteration import Config, main_iteration
import json

result_path = Path("results/")
vid_path = Path("vids/")


def main():

    repeats = 1

    configs = [
        Config(
            age_limit=age_lim,
            tournament_size=tourn_size,
            p_mutate=p_mutate,
            parent_pop_size=parent_sz,
            do_save_best_individual=True,
            iteration_name=f"A{age_lim}T{tourn_size}P{int(p_mutate*100)}Pr{parent_sz}",
            population_size=500,
            iteration_depth=1000,
        )
        for age_lim in [50]   # age until parents are not included in next generation
        for tourn_size in [10]  # number of parents in tournament selection, must be less than n_parents
        for parent_sz in [40]   # number of parents selected from population (remember: alive parents are in next gen)
        for p_mutate in [0.6]   # percentage chance each chromosome will mutate in offspring

    ]
    for config in configs:
        print(
            f"Computing for:"
            f" Age: {config.age_limit},"
            f" Tournament: {config.tournament_size},"
            f" parent: {config.parent_pop_size},"
            f" p_mutate: {config.p_mutate}")

        for _ in range(repeats):
            print(f"Iteration {_+1}/{repeats}\n\n\n")
            iter_result = main_iteration(**config.to_dict())

            file_path = result_path/(config.iteration_name + ".json")

            if file_path.is_file():
                with open(file_path, "r") as file:
                    data = json.loads(file.read())

                data.append(iter_result)
                with open(file_path, "w") as file:
                    json.dump(data, file)

            else:
                with open(file_path, "w") as file:
                    json.dump([iter_result], file)


if __name__ == "__main__":
    main()
