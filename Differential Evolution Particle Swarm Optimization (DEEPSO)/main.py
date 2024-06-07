from deepso_sg import deepso
from deepso_pb import deepso_pb
from deepso_sgpb import deepso_sgpb
import numpy as np
from matplotlib import pyplot as plt

def rosenbrock(X, a=1, b=100):
    return sum(a * (X[i+1] - X[i]**2)**2 + b * (X[i] - 1)**2 for i in range(len(X)-1))

def himmelblau(X):
    x1, x2 = X
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2


# Parâmetros do DEEPSO
dimension = 2
num_particles = 30
lower_limit = -2.048
upper_limit = 2.048
max_iter = 100
executions_num = 100

wi = 0.34062173067065427
wp = 1.4717120642717338
wc = 0.7633790847796198
communication_rate = 0.6805795226782481
mutation_rate = 0.1703704934990093
step_size = 0.48568895792374933

def main():
    all_runs_iterations_sg = []
    all_runs_iterations_pb = []
    all_runs_iterations_sgpb = []

    for _ in range(30):
        all_iterations_sg, _, _, _ = deepso(rosenbrock, dimension, num_particles, lower_limit, upper_limit, wi, wp, wc, max_iter,communication_rate, mutation_rate, step_size)
        all_runs_iterations_sg.append(all_iterations_sg)

        all_iterations_pb, _, _, _ = deepso_pb(rosenbrock, dimension, num_particles, lower_limit, upper_limit, wi, wp, wc, max_iter,communication_rate, mutation_rate, step_size)
        all_runs_iterations_pb.append(all_iterations_pb)

        all_iterations_sgpb, _, _, _ = deepso_sgpb(rosenbrock, dimension, num_particles, lower_limit, upper_limit, wi, wp, wc, max_iter,communication_rate, mutation_rate, step_size)
        all_runs_iterations_sgpb.append(all_iterations_sgpb)

    # Calculando a média das iterações (DEEPSO Sg, Pb e SgPb)
    all_runs_iterations_sg = np.array(all_runs_iterations_sg)
    mean_convergence_curve = np.mean(all_runs_iterations_sg, axis=0)

    all_runs_iterations_pb = np.array(all_runs_iterations_pb)
    mean_convergence_curve_pb = np.mean(all_runs_iterations_pb, axis=0)

    all_runs_iterations_sgpb = np.array(all_runs_iterations_sgpb)
    mean_convergence_curve_sgpb = np.mean(all_runs_iterations_sgpb, axis=0)

    # Plotando a curva de convergência média
    plt.figure(figsize=(10, 6))

    plt.plot(mean_convergence_curve, label='(Sg) Curva de convergência média',color='Red')
    plt.plot(mean_convergence_curve_pb,linestyle = 'dashed', label='(Pb) Curva de convergência média',color='Black')
    plt.plot(mean_convergence_curve_sgpb,linestyle = 'dotted', label='(SgPb) Curva de convergência média',color='Blue')

    plt.plot(len(mean_convergence_curve), mean_convergence_curve[-1], marker='o',markersize = 8, color='Red', label=f'Best (Sg): {mean_convergence_curve[-1]}')
    plt.plot(len(mean_convergence_curve_pb), mean_convergence_curve_pb[-1], marker='o',markersize = 6, color='Black', label=f'Best (Pb): {mean_convergence_curve_pb[-1]}')
    plt.plot(len(mean_convergence_curve_sgpb), mean_convergence_curve_sgpb[-1], marker='o',markersize = 4, color='Blue', label=f'Best (SgPb): {mean_convergence_curve_sgpb[-1]}')

    plt.xlabel('Iterações')
    plt.ylabel('Melhor Valor Global')
    plt.title('Curva de Convergência Média do DEEPSO Rosenbrock')
    plt.legend()
    plt.grid(True)
    plt.show()

main()