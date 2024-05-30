# Função para avaliar o fitness
def evaluate_fitness(function, positions):
    return [function(position) for position in positions]