import random

import numpy as np


# create a random points for our population
def initialization_of_population(size):
    population = []
    for i in range(size):
        chromosome = (((np.random.rand(2) - 0.5) * 4.0) * 5.0)
        population.append(chromosome)  # add random number to population
    return population


def fitness_score(population):
    scores = []
    for chromosome in population:
        score = (chromosome[0] ** 2 + chromosome[1] - 11) ** 2 + (chromosome[0] + chromosome[1] ** 2 - 7) ** 2
        scores.append(score)
    scores, pop = np.array(scores), np.array(population.copy())
    inds = np.argsort(scores)
    scores = list(scores[inds][::-1])
    pop = list(pop[inds, :][::-1])
    return scores, pop


def selection(scores, pop_after_fit):
    population_nextgen = []
    # elitism
    population_nextgen.append(pop_after_fit[0].copy())
    max_val = np.sum(scores)
    for i in range(1, len(scores)):
        for k in range(2):
            pick = random.uniform(0, max_val)
            current = 0
            for j in range(len(scores)):
                print(j)
                current += scores[j]
                if current > pick:
                    break
                population_nextgen.append(pop_after_fit[j].copy())
    print("size of nexGen ", len(population_nextgen))
    return population_nextgen


population = initialization_of_population(5)
scores, pop_after_fit = fitness_score(population)

for i in population:
    print("original population ", i)

for i in selection(scores, pop_after_fit):
    print("population after selection", i)
