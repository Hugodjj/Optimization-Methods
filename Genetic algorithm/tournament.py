import random

def tournament(points_and_fitness, tournament_size, initial_pop):
    """
    Perform a tournament selection to determine survivors.

    Parameters:
    points_and_fitness (dict): A dictionary of points and their fitness values.
    tournament_size (int): The size of the tournament.
    initial_pop (int): The initial population size.

    Returns:
    list: A list of survivors.
    """
    num_survivors = initial_pop * tournament_size // 2

    survivors = []
    for _ in range(int(num_survivors)):
        participants = random.sample(list(points_and_fitness.items()), 3)
        winner = min(participants, key=lambda x: x[1])[0]
        survivors.append(winner)

    survivors.extend(points_and_fitness.keys())
    return survivors