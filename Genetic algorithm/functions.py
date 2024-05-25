import numpy as np


def function2(X, a=1, b=100): #Rosenbrock
  x,y = X
  return (a - x)**2 + b * (y - x**2)**2

def function(X, a=1, b=100):
    return sum(a * (X[i+1] - X[i]**2)**2 + b * (X[i] - 1)**2 for i in range(len(X)-1))

def function1(x): # Levy
    num_dimensoes = len(x)
    w = [1 + (x[i] - 1) / 4 for i in range(num_dimensoes)]

    term1 = (np.sin(np.pi * w[0])) ** 2
    term2 = sum([(w[i] - 1) ** 2 * (1 + 10 * (np.sin(np.pi * w[i] + 1)) ** 2) for i in range(num_dimensoes - 1)])
    term3 = (w[num_dimensoes - 1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[num_dimensoes - 1])) ** 2)

    return term1 + term2 + term3

def function3(X): #BEALE FUNCTION
  x,y = X
  return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2

def function123(X): #LEVY FUNCTION N. 13
    x, y = X
    w1 = 1 + (x - 1) / 4
    w2 = 1 + (y - 1) / 4

    term1 = np.sin(np.pi * w1) ** 2
    term2 = (w1 - 1) ** 2 * (1 + 10 * np.sin(np.pi * w1 + 1) ** 2)
    term3 = (w2 - 1) ** 2 * (1 + np.sin(2 * np.pi * w2) ** 2)

    return term1 + term2 + term3