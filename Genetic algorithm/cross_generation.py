import random

def cross_generation(survivors, initial_pop, tournament_size):
    """
    Perform crossover to generate a new population.

    Parameters:
    survivors (list): A list of survivors from the tournament selection.
    initial_pop (int): The initial population size.
    tournament_size (int): The size of the tournament.

    Returns:
    list: A new population after crossover.
    """
    num_offspring = (initial_pop * tournament_size // 2)

    offspring_survivors = []
    parent_survivors = survivors[:int(num_offspring)]  # Selecting the tournament survivors
    crossover_generation = survivors

    for _ in range(int(num_offspring // 2)):
        # Weighted random crossover
        alpha = random.uniform(0, 1)
        parent1, parent2 = random.sample(parent_survivors, 2)
        child = [alpha * parent1[i] + (1 - alpha) * parent2[i] for i in range(len(parent1))]  # Pairwise crossover of parents
        new_child = child
        offspring_survivors.append(new_child)

    crossover_generation.extend(offspring_survivors)
    return crossover_generation