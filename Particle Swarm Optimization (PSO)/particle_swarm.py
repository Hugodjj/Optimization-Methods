from evaluate_fitness import evaluate_fitness
from generate_initial_particles import generate_initial_particles
from update_best_positions import update_best_positions
from update_velocities_positions import update_velocities_positions


# Função principal do PSO
def pso(function, dimension, num_particles, lower_limit, upper_limit, w, c1, c2, max_iter=100):
    positions, velocities = generate_initial_particles(dimension, num_particles, lower_limit, upper_limit)
    
    personal_best_positions = positions.copy()
    personal_best_values = [float('inf')] * num_particles
    global_best_position = None
    global_best_value = float('inf')
    
    all_iterations_values = []
    all_iterations_positions = []
    
    for _ in range(max_iter):
        fitness_values = evaluate_fitness(function, positions)
        personal_best_positions, personal_best_values, global_best_position, global_best_value = update_best_positions(positions, fitness_values, personal_best_positions, personal_best_values, global_best_position, global_best_value)
        
        positions, velocities = update_velocities_positions(positions, velocities, personal_best_positions, global_best_position, w, c1, c2, lower_limit, upper_limit)
        
        all_iterations_values.append(global_best_value)

        all_iterations_positions.append([pos.copy() for pos in positions])
    

    return all_iterations_positions, all_iterations_values, global_best_position, global_best_value