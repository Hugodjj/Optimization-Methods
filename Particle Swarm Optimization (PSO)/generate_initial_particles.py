from random import uniform
import numpy as np

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