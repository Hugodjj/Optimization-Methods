from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


# Função para gerar GIF
def generate_gif(pso_func, dim, num_particles, max_iter, lower_limit, upper_limit,w,c1,c2, function):
    
    all_Iterations, _, _, _  = pso_func(function, dim, num_particles, lower_limit, upper_limit,w,c1,c2, max_iter=max_iter)
    
    def plot_Iteration(population, i, ax):
        x = np.linspace(lower_limit, upper_limit, 100)
        y = np.linspace(lower_limit, upper_limit, 100)
        X, Y = np.meshgrid(x, y)
        Z = function((X, Y))
        Iteration = population
        x = [point[0] for point in Iteration]
        y = [point[1] for point in Iteration]

        ax.clear()
        ax.contourf(X, Y, Z, levels=50, cmap='rainbow', alpha=0.5)
        best = min(Iteration, key=lambda x: function(x))
        ax.plot(best[0], best[1], 'red', marker='X', markersize=10, alpha=1, label="Best fit")
        ax.annotate(f"({best[0]:.5f}, {best[1]:.5f})", (best[0], best[1]), textcoords="offset points", xytext=(5, 5), ha='center')
        ax.plot(x, y, 'bo')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Iteration {i+1}')
        ax.legend()

    # Example array of arrays of points
    points = all_Iterations[:max_iter]

    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Function to update plot for each frame
    def update_plot(i):
        plot_Iteration(points[i], i, ax)

    # Generate and save GIF
    gif_filename = f'PSO_Convergence_{dim}dim_{max_iter}iterations_{num_particles}particles.gif'
    ani = FuncAnimation(fig, update_plot, frames=len(points), interval=250)

    # Save the GIF
    ani.save(gif_filename, writer='pillow')

    plt.close()