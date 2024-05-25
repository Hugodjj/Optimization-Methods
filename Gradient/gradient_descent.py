import sympy as sym
import numpy as np


a = sym.symbols("a")

def gradient_descent(x0, alpha, max_iterations, tolerance, min_delta_f=1e-6):
    """
    Performs the gradient descent method for optimization.

    Parameters:
    x0 : array
        The initial point to start the optimization.
    alpha : float
        Step size to control the magnitude of updates.
    max_iterations : int
        The maximum number of allowed iterations.
    tolerance : float
        Tolerance for convergence. The algorithm terminates when the gradient norm is less than this tolerance.
    min_delta_f : float, optional
        Additional stopping criterion based on the minimum variation of the function value between consecutive iterations.
        (Implemented based on the explanation of stopping criteria on page 109 of Professor Takahashi's lecture notes)

    Returns:
    x_history : ndarray
        A 2D array containing the sequence of points visited during the algorithm's iterations.
    """
    x_history = [x0]
    iteration = 0
    prev_f = np.inf  # Function value at the previous iteration
    while iteration < max_iterations:
        grad = GradientF(x0)
        x_new = x0 - alpha * grad
        f_new = F(x_new[0], x_new[1])  # Function value at the new position
        if np.linalg.norm(grad) < tolerance:
            break
        delta_f = abs(f_new - prev_f)  # Variation in function value
        if delta_f < min_delta_f:
            break
        x0 = x_new
        x_history.append(x0)
        prev_f = f_new
        iteration += 1
    return np.array(x_history)