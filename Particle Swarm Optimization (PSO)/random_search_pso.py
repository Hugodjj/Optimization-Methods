from random import uniform
import numpy as np
from particle_swarm import pso
from functions import function

# Função para executar a busca aleatória de parâmetros
def random_search_pso(function, dimension, num_particles, lower_limit, upper_limit, max_iter, executions_num):
    best_params = None
    best_global_best_value = float('inf')
    
    for _ in range(executions_num):
        w = uniform(0.1, 0.9)
        c1 = uniform(0.1, 2.0)
        c2 = uniform(0.1, 2.0)
        
        _ , _, _, global_best_value = pso(function, dimension, num_particles, lower_limit, upper_limit, w=w, c1=c1, c2=c2, max_iter=max_iter)
        
        if global_best_value < best_global_best_value:
            best_global_best_value = global_best_value
            best_params = {'w': w, 'c1': c1, 'c2': c2}
    
    return best_params['w'],best_params['c1'],best_params['c2'],global_best_value