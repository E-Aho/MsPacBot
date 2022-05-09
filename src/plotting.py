import json
import re
from math import log10
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class ResultType:
    def __init__(
            self,
            age_lim: int,
            tournament_size: int,
            p_mutate: int,
            parent_size: int,
    ):
        self.age_lim = age_lim
        self.tournament_size = tournament_size
        self.p_mutate = p_mutate
        self.parent_size = parent_size

    def __eq__(self, other):
        if self.__dict__ == other.__dict__:
            return True
        return False

    def __lt__(self, other):
        if self.age_lim != other.age_lim:
            return self.age_lim < other.age_lim
        elif self.tournament_size != other.tournament_size:
            return self.tournament_size < other.tournament_size
        elif self.p_mutate != other.p_mutate:
            return self.p_mutate < other.p_mutate
        else:
            return self.parent_size < other.parent_size

    def filename(self):
        age = self.age_lim
        t = self.tournament_size
        p = self.p_mutate
        pr = self.parent_size
        return f"A{age}T{t}P{p}Pr{pr}.json"

ages_tested = [1, 2, 4, 8, 16, 32 ,64, 100]
tournaments_tested = [2, 4, 8, 16, 32]
probabilities_tested = [5, 10, 20, 40, 80]
parents_tested = [10, 20, 40, 80, 160]


age_experiment = {
    age: ResultType(age, 20, 30, 30)
    for age in ages_tested
}

tournament_experiment = {
    tourn: ResultType(20, tourn, 50, 40)
    for tourn in tournaments_tested
}

mutate_experiment = {
    p: ResultType(20, 10, p, 40)
    for p in [5, 10, 20, 40, 80]
}

parent_experiment = {
    parent: ResultType(20, 10, 50, parent)
    for parent in [10, 20, 40, 80, 160]
}

def parse_file_to_results(filename: str) -> dict:
    with open(Path("results")/filename) as file:
        return json.loads(file.read())

def plot_age_results():
    result_map = {
        a.age_lim: parse_file_to_results(a.filename()) for _, a in age_experiment.items()
    }

    plt.style.use("science")
    final_scores = {}
    final_stdev = {}

    for age, results in result_map.items():
        final_scores[age] = []
        final_stdev[age] = []
        plt.figure()

        x = np.arange(stop=101)
        for iter in results:
            plt.plot(x, iter["best_scores"])
            plt.title(f"$age\_limit = {age}$")
            plt.xlabel("Generation")
            plt.ylabel("High score")
            plt.xlim(0, 101)
            plt.ylim(1000, 5500)
            final_scores[age].append(iter["best_scores"][-1])
            final_stdev[age].append(iter["st_devs"][-1])


        plt.savefig(Path(f"plots/age_{age}"))

    # Plot for high scores
    plt.figure()
    for a, results in final_scores.items():
        a = log10(a)
        plt.scatter((a, a, a), results, marker="x", s=8)
    plt.xlim(0, 2.1)
    plt.xlabel("log(age limit)")
    plt.ylabel("Final scores")
    plt.savefig(Path(f"plots/age_total"))
    plt.close()

    plt.figure()
    for a, results in final_stdev.items():
        a = log10(a)
        plt.scatter((a, a, a), results, marker="x", s=8)
    plt.xlim(0, 2.1)
    plt.xlabel("log(age limit)")
    plt.ylabel("Std deviation")
    plt.savefig(Path(f"plots/age_stddev"))
    plt.close()


def plot_tournament_results():
    result_map = {
        t.tournament_size: parse_file_to_results(t.filename()) for _, t in tournament_experiment.items()
    }

    plt.style.use("science")
    final_scores = {}
    final_stdev = {}

    for tourn, results in result_map.items():
        final_scores[tourn] = []
        final_stdev[tourn] = []
        plt.figure()

        x = np.arange(stop=101)
        for iter in results:
            plt.plot(x, iter["best_scores"])
            plt.title(f"$tournament\_size = {tourn}$")
            plt.xlabel("Generation")
            plt.ylabel("High score")
            plt.xlim(0, 101)
            plt.ylim(1000, 5500)
            final_scores[tourn].append(iter["best_scores"][-1])
            final_stdev[tourn].append(iter["st_devs"][-1])

        plt.savefig(Path(f"plots/tournament_{tourn}"))

    # Plot for high scores
    plt.figure()
    for t, results in final_scores.items():
        t = log10(t)
        plt.scatter((t, t, t), results, marker="x", s=8)
    plt.xlabel("log(tournament size)")
    plt.ylabel("Final scores")
    plt.savefig(Path(f"plots/tournament_total"))
    plt.close()

    plt.figure()
    for t, results in final_stdev.items():
        t = log10(t)
        plt.scatter((t, t, t), results, marker="x", s=8)
    plt.xlabel("log(tournament size)")
    plt.ylabel("Std deviation")
    plt.savefig(Path(f"plots/tournament_stddev"))
    plt.close()


def plot_mutate_results():
    result_map = {
        p.p_mutate: parse_file_to_results(p.filename()) for _, p in mutate_experiment.items()
    }

    plt.style.use("science")
    final_scores = {}
    final_stdev = {}

    for p_mutate, results in result_map.items():
        final_scores[p_mutate] = []
        final_stdev[p_mutate] = []
        plt.figure()

        x = np.arange(stop=101)
        for iter in results:
            plt.plot(x, iter["best_scores"])
            plt.title(f"$p\_ mutate = {p_mutate * 0.01}$")
            plt.xlabel("Generation")
            plt.ylabel("High score")
            plt.xlim(0, 101)
            plt.ylim(1000, 5500)
            final_scores[p_mutate].append(iter["best_scores"][-1])
            final_stdev[p_mutate].append(iter["st_devs"][-1])

        plt.savefig(Path(f"plots/mutate_{p_mutate}"))

    # Plot for high scores
    plt.figure()
    for p, results in final_scores.items():
        p = 0.01 * p
        plt.scatter((p, p, p), results, marker="x", s=8)
    plt.xlim(0, 1)
    plt.xlabel("Probability of mutation")
    plt.ylabel("Final scores")
    plt.savefig(Path(f"plots/mutate_total"))
    plt.close()

    plt.figure()
    for p, results in final_stdev.items():
        p = 0.01 * p
        plt.scatter((p, p, p), results, marker="x", s=8)
    plt.xlim(0, 1)
    plt.xlabel("Probability of mutation")
    plt.ylabel("Std deviation")
    plt.savefig(Path(f"plots/mutate_stddev"))
    plt.close()


def plot_parent_results():
    result_map = {
        p.parent_size: parse_file_to_results(p.filename()) for _, p in parent_experiment.items()
    }

    plt.style.use("science")
    final_scores = {}
    final_stdev = {}

    for parent, results in result_map.items():
        final_scores[parent] = []
        final_stdev[parent] = []
        plt.figure()

        x = np.arange(stop=101)
        for iter in results:
            plt.plot(x, iter["best_scores"])
            plt.title(f"$n\_parent = {parent}$")
            plt.xlabel("Generation")
            plt.ylabel("High score")
            plt.xlim(0, 101)
            plt.ylim(1000, 5500)
            final_scores[parent].append(iter["best_scores"][-1])
            final_stdev[parent].append(iter["st_devs"][-1])

        plt.savefig(Path(f"plots/parent_{parent}"))

    # Plot for high scores
    plt.figure()
    for p, results in final_scores.items():
        p = log10(p)
        plt.scatter((p, p, p), results, marker="x", s=8)
    plt.xlabel("log(number of parents selected)")
    plt.ylabel("Final scores")
    plt.savefig(Path(f"plots/parent_total"))
    plt.close()

    plt.figure()
    for p, results in final_stdev.items():
        p = log10(p)
        plt.scatter((p, p, p), results, marker="x", s=8)
    plt.xlabel("log(number of parents selected)")
    plt.ylabel("Std deviation")
    plt.savefig(Path(f"plots/parent_stddev"))
    plt.close()


if __name__ == "__main__":
    plot_age_results()
    plot_tournament_results()
    plot_mutate_results()
    plot_parent_results()
