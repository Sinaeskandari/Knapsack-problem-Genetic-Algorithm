# Q3_graded
# Do not change the above line.

import numpy as np
from random import choices, seed, randint, random

knapsack_size = 25
weights = np.array([2, 4, 1, 3, 5, 1, 7, 4]).reshape(-1, 1)
values = np.array([30, 10, 20, 50, 70, 15, 40, 25]).reshape(-1, 1)

population_size = 100
population = np.random.randint(2, size=(population_size, weights.shape[0]))

# Q3_graded
# Do not change the above line.

def fitness(population, values, weights, knapsack_size, normalize=True):
    total_value = np.dot(population, values)
    # if total weight of chromosome is bigger than knapsack size then value is 0
    total_weight = np.dot(population, weights) > knapsack_size
    total_value[total_weight] = 0
    if normalize:
        return total_value / total_value.sum(axis=0)
    return total_value

# Q3_graded
# Do not change the above line.

def selection(fitness, population):
    f = list(fitness.reshape(-1))
    # choose half of population randomly based on weights
    return choices(population, weights=f, k=len(f) // 2)


# Q3_graded
# Do not change the above line.

def crossover(parents):
    # each two parents generate two childs
    new_population = []
    for i in range(len(parents)):
        p1 = parents[i]
        p2 = parents[(i + 1) % len(parents)]
        index = randint(1, population.shape[1] - 1)
        child1_left = p1[:index]
        child1_right = p2[index:]
        new_population.append(np.concatenate((child1_left, child1_right)))
        child2_left = p2[:index]
        child2_right = p1[index:]
        new_population.append(np.concatenate((child1_left, child1_right)))
    return np.array(new_population)

# Q3_graded
# Do not change the above line.

def flip_coin(p):
    # random function for mutation
    r = random()
    return r < p

def mutation(population, mutation_rate):
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            if flip_coin(mutation_rate):
                population[i][j] = 1 - population[i][j]
    return population

# Q3_graded
# Do not change the above line.

def ga(knapsack_size, weights, values, population, epochs=50, mutation_rate=0.01):
    current_population = population
    for i in range(epochs):
        fitness_values_normalized = fitness(current_population, values, weights, knapsack_size)
        fitness_values = fitness(current_population, values, weights, knapsack_size, normalize=False)
        parents = selection(fitness_values_normalized, population)
        current_population = mutation(crossover(parents), mutation_rate)
        print(f'epoch {i + 1} ----> total fitness={fitness_values.sum(axis=0)}')
    fitness_values = fitness(current_population, values, weights, knapsack_size, normalize=False)
    return fitness_values.max(axis=0), current_population[np.unravel_index(np.argmax(fitness_values, axis=None), fitness_values.shape)[0]]

ga(knapsack_size, weights, values, population)

