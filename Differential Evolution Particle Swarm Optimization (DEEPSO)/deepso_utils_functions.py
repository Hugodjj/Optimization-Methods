from random import random, randint, uniform
import numpy as np
from matplotlib import pyplot as plt

# Função para gerar pontos iniciais (posições e velocidades)
def generate_initial_particles(dimension, num_particles, lower_limit, upper_limit):
    positions = []
    velocities = []
    for _ in range(num_particles):
        position = [uniform(lower_limit, upper_limit) for _ in range(dimension)]
        velocity = [uniform(-1, 1) for _ in range(dimension)]
        positions.append(np.array(position))
        velocities.append(np.array(velocity))
    return positions, velocities

# Função para avaliar o fitness
def evaluate_fitness(function, positions):
    return [function(position) for position in positions]

# Função para atualizar as melhores posições locais e global
def update_best_positions(positions, fitness_values, personal_best_positions, personal_best_values, global_best_position, global_best_value):
    for i, fitness_value in enumerate(fitness_values):
        if fitness_value < personal_best_values[i]:
            personal_best_values[i] = fitness_value
            personal_best_positions[i] = positions[i].copy()
        if fitness_value < global_best_value:
            global_best_value = fitness_value
            global_best_position = positions[i].copy()
    return personal_best_positions, personal_best_values, global_best_position, global_best_value

def generate_communication_matrix(dimension, communication_rate):
    # Criar uma matriz de zeros
    C = np.zeros((dimension, dimension))

    # Gerar valores aleatórios entre 0 e 1 para a diagonal
    random_values = np.random.uniform(size=dimension)

    # Definir 1 na diagonal onde os valores aleatórios são menores ou iguais a communication_rate
    C[np.diag_indices(dimension)] = random_values <= communication_rate

    return C

# Função para realizar uma busca local na vizinhança de uma posição
def local_search(function, position, step_size, lower_limit, upper_limit):
    best_position = position.copy()
    best_fitness = function(position)

    for dim in range(len(position)):
        direction = uniform(-1, 1)  # Direção aleatória entre -1 e 1
        new_position = position.copy()
        new_position[dim] += direction * step_size
        new_position = np.clip(new_position, lower_limit, upper_limit)
        new_fitness = function(new_position)
        if new_fitness < best_fitness:
            best_fitness = new_fitness
            best_position = new_position

    return best_position

# Função para verificar se um mínimo já foi encontrado
def is_new_minimum(new_position, known_minima, tolerance=1e-9):
    for known_position in known_minima:
        if np.linalg.norm(new_position - known_position) < tolerance:
            return False
    return True

# Função para redirecionar partículas para novas posições
def redirect_particles(dimension, num_particles, lower_limit, upper_limit):
    return [np.array([uniform(lower_limit, upper_limit) for _ in range(dimension)]) for _ in range(num_particles)]