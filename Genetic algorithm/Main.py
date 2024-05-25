from genetic_algorithm import genetic_algorithm
from functions import function
from generate_gif import generate_gif

dim = 2
initial_pop = 100
num_generations = 100
tournament_size = 0.8
mutation_rate = 0.05
sigma = 0.1
lower_limit = -5
upper_limit = 5

def Main():
    generate_gif(genetic_algorithm, dim, initial_pop, num_generations, tournament_size, mutation_rate, sigma, lower_limit, upper_limit, function)
    genetic_algorithm(dim, initial_pop, num_generations, tournament_size, mutation_rate, sigma, lower_limit, upper_limit, function)

Main()
