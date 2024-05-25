def calculate_fitness(initial_points, function):
    """
    Calculate the fitness of the initial points.

    Parameters:
    initial_points (list): A list of points to evaluate.
    function (callable): The function to evaluate the fitness of the points.

    Returns:
    dict: A dictionary with points as keys and their fitness values as values.
    """
    fx_values = {tuple(point): function(point) for point in initial_points}
    return fx_values