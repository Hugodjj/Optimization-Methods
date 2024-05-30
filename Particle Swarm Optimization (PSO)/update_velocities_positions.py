import numpy as np
from random import random

# Função para atualizar velocidades e posições
def update_velocities_positions(positions, velocities, personal_best_positions, global_best_position, w, c1, c2, lower_limit, upper_limit):
    for i in range(len(positions)):
        inertia = w * velocities[i]
        cognitive = c1 * random() * (personal_best_positions[i] - positions[i])
        social = c2 * random() * (global_best_position - positions[i])
        
        velocities[i] = inertia + cognitive + social
        positions[i] = positions[i] + velocities[i]

        # Ensure the position is within bounds
        positions[i] = np.clip(positions[i], lower_limit, upper_limit)
    return positions, velocities