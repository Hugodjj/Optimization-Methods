from random import uniform

# Function to generate points for the initial population
def generate_initial_points(dimension, quantity, lower_limit, upper_limit):
    """
    Generate initial points for the population.

    Parameters:
    dimension (int): Number of dimensions for each point.
    quantity (int): Number of points to generate.
    lower_limit (float): Lower limit for the values of the points.
    upper_limit (float): Upper limit for the values of the points.

    Returns:
    list: A list of tuples representing the initial points.
    """
    numbers = []
    for _ in range(quantity):
        number = tuple(uniform(lower_limit, upper_limit) for _ in range(dimension))
        numbers.append(number)
    return numbers