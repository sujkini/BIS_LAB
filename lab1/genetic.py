import numpy as np

def fitness_function(binary_string):
    # Example: Maximize the number of 1s in the binary string
    return np.sum(binary_string)

def initialize_population(pop_size, chromosome_length):
    return np.random.randint(2, size=(pop_size, chromosome_length))

def evaluate_fitness(population):
    return np.array([fitness_function(individual) for individual in population])

def select_parents(population, fitness):
    probabilities = fitness / fitness.sum()  # Convert fitness to probabilities
    indices = np.random.choice(len(population), size=2, replace=False, p=probabilities)
    return population[indices[0]], population[indices[1]]

def crossover(parent1, parent2, crossover_rate=0.8):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    return parent1.copy(), parent2.copy()

def mutate(individual, mutation_rate=0.1):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]  # Flip bit
    return individual

def genetic_algorithm(
    pop_size=100,
    chromosome_length=20,
    generations=50,
    crossover_rate=0.8,
    mutation_rate=0.1,
):
    # Initialize population
    population = initialize_population(pop_size, chromosome_length)

    for generation in range(generations):
        fitness = evaluate_fitness(population)
        next_generation = []

        for _ in range(pop_size // 2):
            # Select parents
            parent1, parent2 = select_parents(population, fitness)

            # Perform crossover
            child1, child2 = crossover(parent1, parent2, crossover_rate)

            # Perform mutation
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            next_generation.append(child1)
            next_generation.append(child2)

        population = np.array(next_generation)
        best_fitness = np.max(fitness)
        best_individual = population[np.argmax(fitness)]
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

    return best_individual, best_fitness

if __name__ == "__main__":
    best_solution, best_value = genetic_algorithm()
    print("Best Solution:", best_solution)
    print("Best Fitness Value:", best_value)
