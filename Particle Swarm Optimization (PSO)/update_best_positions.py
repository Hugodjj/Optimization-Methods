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