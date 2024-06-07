from matplotlib import pyplot as plt
import numpy as np
from particle_swarm import pso
from generate_gif import generate_gif
from functions import function
from random_search_pso import random_search_pso

# Parâmetros do PSO
dimension = 100
num_particles = 30
lower_limit = -2.048
upper_limit = 2.048
max_iter = 10000
executions_num = 100

# Best parameters 2 dim
w = 0.33135727305417906
c1 = 1.8152741857990424
c2 = 1.6587772220768413

# Best parameters 10 dim
#w = 0.6839866092319149
#c1 = 1.3795707646107036
#c2 = 0.810332567052204

# Best parameters 100 dim
#w = 0.8459442594029418
#c1 = 1.7350045871406414
#c2 = 1.6487351502210512

def main():
    
    w, c1, c2, global_best = random_search_pso(function, dimension, num_particles, lower_limit, upper_limit, max_iter, executions_num)
    
    generate_gif(pso,dimension,num_particles,max_iter,lower_limit,upper_limit,w,c1,c2,function)
    

main()
