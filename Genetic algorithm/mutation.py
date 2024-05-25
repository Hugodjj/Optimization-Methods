import random


def mutation(crossover_generation, sigma, initial_pop, mutation_rate, function):
    """
    Perform mutation on the crossed generation.

    Parameters:
    crossover_generation (list): A list of individuals after crossover.
    sigma (float): Standard deviation for Gaussian distribution used in mutation.
    initial_pop (int): The initial population size.
    mutation_rate (float): The rate of mutation.
    function (callable): The function to evaluate the fitness of individuals.

    Returns:
    list: A new population after mutation.
    """
    new_population = []
    num_dimensions = len(crossover_generation[0])  # Get the number of dimensions of the individuals

    # Calculate the number of individuals to be mutated (% of the population)
    num_mutated_individuals = int(len(crossover_generation) * mutation_rate)

    # Randomly select the indices of the individuals to be mutated
    mutated_indices = random.sample(range(len(crossover_generation)), num_mutated_individuals)

    for idx, individual in enumerate(crossover_generation):
        if idx in mutated_indices:
            # Apply creep mutation only to the selected individuals
            new_individual = [i + random.gauss(0, sigma) for i in individual]
            # If the new individual has fewer dimensions than expected, fill with zeros
            while len(new_individual) < num_dimensions:
                new_individual.append(0)
            new_population.append(new_individual)
        else:
            # Add non-mutated individuals to the new population
            new_population.append(individual)

    # Sort individuals by fitness value
    sorted_new_population = sorted(new_population, key=lambda x: function(x))

    # Select only the best individuals based on the initial population
    selected_new_population = sorted_new_population[:initial_pop]

    return selected_new_population