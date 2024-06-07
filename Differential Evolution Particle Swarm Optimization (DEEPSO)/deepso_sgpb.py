from deepso_utils_functions import generate_communication_matrix, local_search, evaluate_fitness, generate_initial_particles, update_best_positions
from random import randint, uniform
import numpy as np

# Função para realizar recombinação linear
def linear_recombination(Xr1, Xr2):
    alpha = uniform(0,1)
    return [alpha * Xr1[i] + (1 - alpha) * Xr2[i] for i in range(len(Xr1))]

# Função para atualizar velocidades e posições
def update_velocities_positions_deepso_pbsg(positions, velocities, global_best_position, wi, wp, wc, lower_limit, upper_limit,communication_rate, mutation_rate, best_individuals,function,step_size):
    for i in range(len(positions)):
        dimension = len(positions[i])

        Xr1_index = randint(0, len(best_individuals) - 1)
        Xr1 = best_individuals[Xr1_index]

        Xr2_index = randint(0, len(positions) - 1)
        while Xr2_index == i:
            Xr2_index = randint(0, len(positions) - 1)
        Xr2 = positions[Xr2_index]

        Xr = linear_recombination(Xr1,Xr2)
        communication_matrix = generate_communication_matrix(dimension,communication_rate)

        wi = wi * (1 + mutation_rate * np.random.normal(0,1))
        wp = wp * (1 + mutation_rate * np.random.normal(0,1))
        wc = wc * (1 + mutation_rate * np.random.normal(0,1))

        wi = np.clip(wi,0,1)
        wp = np.clip(wp,0,1)
        wc = np.clip(wc,0,1)

        possible_global_best_position = global_best_position + global_best_position * (mutation_rate * np.random.normal(0,1))

        if  function(global_best_position) > function(possible_global_best_position):
          global_best_position = possible_global_best_position

        inertia = wi * velocities[i]
        cognitive = wp * (Xr - positions[i])
        social = wc * (communication_matrix @ (global_best_position - positions[i]))

        velocities[i] = inertia + cognitive + social
        positions[i] = positions[i] + velocities[i]

        # Garantir que as posições estão dentro do espaço de busca
        positions[i] = np.clip(positions[i], lower_limit, upper_limit)

        # Aplicar a busca local na nova posição
        positions[i] = local_search(function, positions[i], step_size, lower_limit, upper_limit)

    return positions, velocities


# Função principal do Deepso
def deepso_sgpb(function, dimension, num_particles, lower_limit, upper_limit, wi, wp, wc, max_iter,communication_rate, mutation_rate, step_size):
    positions, velocities = generate_initial_particles(dimension, num_particles, lower_limit, upper_limit)

    personal_best_positions = positions.copy()
    personal_best_values = [float('inf')] * num_particles
    global_best_position = None
    global_best_value = float('inf')

    all_iterations = []
    all_iterations_positions = []
    best_individuals = []

    for _ in range(max_iter):
        all_iterations.append(global_best_value)
        all_iterations_positions.append([pos.copy() for pos in positions])

        fitness_values = evaluate_fitness(function, positions)
        personal_best_positions, personal_best_values, global_best_position, global_best_value = update_best_positions(positions, fitness_values, personal_best_positions, personal_best_values, global_best_position, global_best_value)

        # Atualizar a lista dos 10% melhores indivíduos
        num_best_individuals = max(1, len(positions) // 10)
        sorted_indices = np.argsort(fitness_values)
        best_individuals = [positions[i] for i in sorted_indices[:num_best_individuals]]

        positions, velocities = update_velocities_positions_deepso_pbsg(positions, velocities, global_best_position, wi, wp, wc, lower_limit, upper_limit, communication_rate, mutation_rate, best_individuals,function,step_size)

    return all_iterations, all_iterations_positions, global_best_position, global_best_value