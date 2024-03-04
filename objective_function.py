import numpy as np
from Save_load import *
from Classifiers import *
from FA import *
from SSpiderA import *

def obj_fun(soln):
    X_train=load('cur_X_train')
    X_test=load('cur_X_test')
    y_train=load('cur_y_train')
    y_test=load('cur_y_test')

    # Feature selection
    soln = np.round(soln)
    X_train=X_train[:,np.where(soln==1)[0]]
    X_test = X_test[:, np.where(soln == 1)[0]]
    pred, met = pro_classifers(X_train, y_train, X_test, y_test)
    fit = 1/met[0]
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

    return fit