from Data_gen import data_gen

from FA import *
from SSpiderA import *
from plot_res import plot_res
import numpy as np

from Save_load import *
from Classifiers import *
import numpy as np

def full_analysis():
    #data_gen()

    def fitness_function(solution):
        return np.sum(solution ** 2)

    problem_dict1 = {
        "fit_func": fitness_function,
        "lb": [-10, -15, -4, -2, -8],
        "ub": [10, 15, 12, 8, 20],
        "minmax": "min",
    }
    epoch = 1000
    pop_size = 50
    max_sparks = 50
    p_a = 0.04
    p_b = 0.8
    max_ea = 40
    m_sparks = 5
    model = OriginalFA(epoch, pop_size, max_sparks, p_a, p_b, max_ea, m_sparks)
    best_position, best_fitness = model.solve(problem_dict1)
    print(f"Solution: {best_position}, Fitness: {best_fitness}")

    save("FO",best_position)


    def fitness_function(solution):
        return np.sum(solution ** 2)

    problem_dict1 = {
        "fit_func": fitness_function,
        "lb": [-10, -15, -4, -2, -8],
        "ub": [10, 15, 12, 8, 20],
        "minmax": "min",
    }
    epoch = 1000
    pop_size = 50
    r_a = 1.0
    p_c = 0.7
    p_m = 0.1
    model = OriginalSSpiderA(epoch, pop_size, r_a, p_c, p_m)
    best_position, best_fitness = model.solve(problem_dict1)
    print(f"Solution: {best_position}, Fitness: {best_fitness}")

    save("Social_spider", best_position)

    X_train = load("x_train")
    Y_train = load("y_train")
    X_test = load("x_test")
    Y_test = load("y_test")

    stacked_autoencoder(X_train, Y_train, X_test, Y_test)

    cnn(X_train, Y_train, X_test, Y_test)

    quantum_neural_network(X_train, Y_train, X_test, Y_test)
#a =1
#if a ==0:
#    full_analysis()
full_analysis()
#plot_res()
