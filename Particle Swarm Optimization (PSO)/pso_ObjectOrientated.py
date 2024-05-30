import numpy as np
from random import uniform, random

class Particle:
    def __init__(self, position, velocity):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.best_position = np.array(position)
        self.best_value = float('inf')

class PSO:
    def __init__(self, function, dimension, num_particles, lower_limit, upper_limit, w=0.5, c1=1.5, c2=1.5, max_iter=100):
        self.function = function
        self.dimension = dimension
        self.num_particles = num_particles
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        
        self.particles = self._initialize_particles()
        self.global_best_position = None
        self.global_best_value = float('inf')

    def _initialize_particles(self):
        particles = []
        for _ in range(self.num_particles):
            position = [uniform(self.lower_limit, self.upper_limit) for _ in range(self.dimension)]
            velocity = [uniform(-1, 1) for _ in range(self.dimension)]
            particles.append(Particle(position, velocity))
        return particles

    def optimize(self):
        for _ in range(self.max_iter):
            for particle in self.particles:
                fitness_value = self.function(particle.position)
                
                if fitness_value < particle.best_value:
                    particle.best_value = fitness_value
                    particle.best_position = particle.position.copy()
                
                if fitness_value < self.global_best_value:
                    self.global_best_value = fitness_value
                    self.global_best_position = particle.position.copy()
            
            for particle in self.particles:
                inertia = self.w * particle.velocity
                cognitive = self.c1 * random() * (particle.best_position - particle.position)
                social = self.c2 * random() * (self.global_best_position - particle.position)
                
                particle.velocity = inertia + cognitive + social
                particle.position = particle.position + particle.velocity

                # Ensure the position is within bounds
                particle.position = np.clip(particle.position, self.lower_limit, self.upper_limit)
        
        return self.global_best_position, self.global_best_value

# Função de teste (por exemplo, função Rastrigin)
def rastrigin_function(position):
    A = 10
    return A * len(position) + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in position])

# Parâmetros do PSO
dimension = 2
num_particles = 30
lower_limit = -5.12
upper_limit = 5.12
max_iter = 100

# Inicializando e executando o PSO
pso = PSO(function=rastrigin_function, dimension=dimension, num_particles=num_particles, lower_limit=lower_limit, upper_limit=upper_limit, max_iter=max_iter)
best_position, best_value = pso.optimize()

print(f"Melhor posição encontrada: {best_position}")
print(f"Melhor valor encontrado: {best_value}")