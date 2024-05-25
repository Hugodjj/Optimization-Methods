from cross_generation import cross_generation
from mutation import mutation
from generate_initial_points import generate_initial_points
from tournament import tournament
from calculate_fitness import calculate_fitness

def genetic_algorithm(dim, initial_pop, num_generations, tournament_size, mutation_rate, sigma, lower_limit, upper_limit, function):
    """
    Perform a genetic algorithm to optimize a given function.

    Parameters:
    dim (int): Number of dimensions for each individual.
    initial_pop (int): Initial population size.
    num_generations (int): Number of generations to run the algorithm.
    tournament_size (int): Size of the tournament for selection.
    mutation_rate (float): The rate of mutation.
    sigma (float): Standard deviation for Gaussian distribution used in mutation.
    lower_limit (float): Lower limit for the values of the points.
    upper_limit (float): Upper limit for the values of the points.
    function (callable): The function to be optimized.

    Returns:
    tuple: A tuple containing:
        - list: A list of populations over the generations.
        - list: A list of the best fitness values over the generations.
    """
    # Initial generation
    populations = []
    best_fitness = []
    initial_points = generate_initial_points(dim, initial_pop, lower_limit, upper_limit)
    fx_values = calculate_fitness(initial_points, function)
    populations.append(initial_points)

    for _ in range(num_generations):
        # Select survivors
        survivors = tournament(fx_values, tournament_size, initial_pop)

        # Perform crossover
        crossed_generation = cross_generation(survivors, initial_pop, tournament_size)

        # Perform mutation
        new_population = mutation(crossed_generation, sigma, initial_pop, mutation_rate, function)

        # Update the fitness values of the new population
        fx_values = calculate_fitness(new_population, function)

        # Store the new population
        populations.append(new_population)

        # Save the best fitness value of the current generation
        best_fitness.append(min(fx_values.values()))

    # Return all populations
    return populations, best_fitness