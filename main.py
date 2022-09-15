import random

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# create a random points for our population
def initialization_of_population(size):
    population = []
    for i in range(size):
        chromosome = (((np.random.rand(2) - 0.5) * 2.0) * 5.0)
        population.append(chromosome)  # add random number to population
    return population


def fitness_score(population):
    scores = []
    for chromosome in population:
        score = (chromosome[0] ** 2 + chromosome[1] - 11) ** 2 + (chromosome[0] + chromosome[1] ** 2 - 7) ** 2
        scores.append(score)
    # scores, pop = np.array(scores), np.array(population.copy())
    # inds = np.argsort(scores)
    # scores = list(scores[inds][::-1])
    # pop = list(pop[inds, :][::-1])
    return scores, population


def selection(scores, pop_after_fit):
    population_nextgen = []
    # elitism
    # population_nextgen.append(pop_after_fit[0].copy())
    for i in range(0, len(pop_after_fit)):
        for j in range(2):
            challenger1 = random.randint(0, len(pop_after_fit) - 1)
            challenger2 = random.randint(0, len(pop_after_fit) - 1)
            if scores[challenger1] < scores[challenger2]:
                population_nextgen.append(pop_after_fit[challenger1])
            else:
                population_nextgen.append(pop_after_fit[challenger2])
    return population_nextgen


def crossover(pop_after_sel, sz):
    population_nextgen = []
    population_nextgen.append(pop_after_sel[0].copy())
    for i in range(1, sz):
        child = pop_after_sel[2 * i + 0].copy()
        parent2 = pop_after_sel[2 * i + 1].copy()
        loc = np.random.randint(0, 1)
        child[loc] = parent2[loc]
        population_nextgen.append(child)
    return population_nextgen


def mutation(pop_after_cross, mutation_rate, sz):
    population_nextgen = []
    population_nextgen.append(pop_after_cross[0].copy())
    for i in range(1, sz):
        chromosome = pop_after_cross[i].copy()
        for j in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[j] = (((np.random.rand() - 0.5) * 2.0) * 5.0)
        population_nextgen.append(chromosome.copy())
    return population_nextgen


def generation(sz, mutation_rate, n_gen):
    best_chromosome = []
    best_score = []
    population_nextgen = initialization_of_population(sz)
    stats_min = np.zeros(n_gen)
    stats_avg = np.zeros(n_gen)
    stats_max = np.zeros(n_gen)

    stored_points = []

    for i in tqdm(range(n_gen)):
        # check fitness
        scores, population_after_fit = fitness_score(population_nextgen)
        # selection
        population_selection = selection(scores, population_after_fit)
        # crossover
        pop_after_cross = crossover(population_selection, sz)
        # mutation
        population_nextgen = mutation(pop_after_cross, mutation_rate, sz)

        scores, population_after_fit = fitness_score(population_nextgen)
        stored_points.append(population_after_fit)
        best_chromosome.append(population_after_fit[0].copy())
        best_score.append(scores[0].copy())
        stats_min[i] = np.min(scores)
        stats_max[i] = np.amax(scores)
        stats_avg[i] = np.mean(scores)
    return best_chromosome, best_score, stats_min, stats_avg, stats_max, stored_points


s = 100
generations = 100
chromosome, score, stats_min, stats_avg, stats_max, stored_points = generation(sz=s, mutation_rate=0.05,
                                                                               n_gen=generations)

print(score)
print(chromosome)

# plotting
plt.plot(stats_min, "r")
plt.plot(stats_avg, "b")
plt.plot(stats_max, "g")
plt.ylabel("accuracy")
plt.xlabel("generation")
plt.show()