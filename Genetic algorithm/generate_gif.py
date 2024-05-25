from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def generate_gif(genetic_algorithm_func, dim, initial_pop, num_generations, tournament_size, mutation_rate, sigma, lower_limit, upper_limit, function):
    """
    Generate a GIF to visualize the genetic algorithm's progress.

    Parameters:
    genetic_algorithm_func (function): The genetic algorithm function to execute.
    dim (int): Number of dimensions for each individual.
    initial_pop (int): Initial population size.
    num_generations (int): Number of generations to run the algorithm.
    tournament_size (float): Size of the tournament for selection.
    mutation_rate (float): The rate of mutation.
    sigma (float): Standard deviation for Gaussian distribution used in mutation.
    lower_limit (float): Lower limit for the values of the points.
    upper_limit (float): Upper limit for the values of the points.
    function (callable): The function to be optimized.
    """
    # Execute the genetic algorithm to obtain all generations
    all_generations, _ = genetic_algorithm_func(dim, initial_pop, num_generations, tournament_size, mutation_rate, sigma, lower_limit, upper_limit, function)

    def plot_generation(population, i, ax):
        x = np.linspace(lower_limit, upper_limit, 100)
        y = np.linspace(lower_limit, upper_limit, 100)
        X, Y = np.meshgrid(x, y)
        Z = function((X, Y))

        generation = population
        x = [point[0] for point in generation]
        y = [point[1] for point in generation]

        ax.clear()
        ax.contourf(X, Y, Z, levels=50, cmap='rainbow', alpha=0.5)
        best = min(generation, key=lambda x: function(x))
        ax.plot(best[0], best[1], 'red', marker='X', markersize=10, alpha=1, label="Best fit")
        ax.annotate(f"({best[0]:.5f}, {best[1]:.5f})", (best[0], best[1]), textcoords="offset points", xytext=(5, 5), ha='center')
        ax.plot(x, y, 'bo')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Generation {}'.format(i + 1))
        ax.legend()

    # Example array of arrays of points
    points = all_generations[:num_generations]

    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Function to update plot for each frame
    def update_plot(i):
        plot_generation(points[i], i, ax)

    # Generate and save GIF
    gif_filename = f'Rosenbrock_Convergence_{dim}dim_{num_generations}generations_{initial_pop}individuals.gif'
    ani = FuncAnimation(fig, update_plot, frames=len(points), interval=250)

    # Save the GIF
    ani.save(gif_filename, writer='pillow')

    plt.close()
