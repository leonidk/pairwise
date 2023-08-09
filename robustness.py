# a script to evaluate robustness of pairwise optimization to user error
# (c) Katherine Shih 2023

import cma
import pickle
from pymoo.problems import get_problem

import functools
import numpy as np
import matplotlib.pyplot as plt

# noisy channel model
# return wrong answer with p = err
def noisy_compare(i, j, err, dim = 0):
    val = 2 * (i[dim] > j[dim]) - 1
    if np.random.rand() < err:
        return -val
    return val

# script config
f_test_list = ["ackley"]  # , "rosenbrock", "sphere", "zakharov"]
n_var_list = [int(x) for x in np.logspace(2, 5, 4, base=2)]
p_err_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
max_eval = 2000

# plot config
plt.style.use("fivethirtyeight")
plt.style.use("seaborn-white")
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    "color", plt.cm.inferno(np.linspace(0.2, 0.9, len(p_err_list)))
)

# CMA-ES
for f_test in f_test_list:
    for n_var in n_var_list:
        plt.figure(figsize=(7, 4))
        for p_err in p_err_list:
            np.random.seed(4563)
            init_guess = np.random.rand(n_var)

            f_func = get_problem(f_test, n_var)

            es = cma.CMAEvolutionStrategy(init_guess, 0.5)
            max_iter = int(round(max_eval / es.popsize))

            best_sol = []
            best_sol_res = []
            # sample batches and noisy sort
            for ii in range(max_iter):
                solutions = es.ask()
                results = [(f_func.evaluate(x)[0], x) for x in solutions]
                res = sorted(
                    results,
                    key=functools.cmp_to_key(lambda i, j: noisy_compare(i, j, p_err)),
                )
                es.tell([x[1] for x in res], np.arange(len(res)))
                best_sol.append(res[0])

                # final sort (so far)
                best_sol_res.append(
                    sorted(
                        best_sol,
                        key=functools.cmp_to_key(
                            lambda i, j: noisy_compare(i, j, p_err)
                        ),
                    )[0][0]
                )
            plt.plot(
                [(i + 1) * es.popsize for i in range(len(best_sol_res))], best_sol_res
            )
            plt.text(
                max_eval * 1.005,
                best_sol_res[-1],
                "{:.0f}%".format(100 * p_err),
                va="center",
                fontsize=14,
            )
        plt.yscale("log")
        plt.grid(True)
        plt.xlabel("Function Evaluations")
        plt.xlim(0, max_eval)
        plt.gca().spines[["right", "top"]].set_visible(False)
        plt.title(f_test + " " + str(n_var))
        plt.tight_layout()
        # plt.savefig('{}_{}_{}.pdf'.format(f_test,n_var,p_err),bbox_inches='tight', facecolor='w',
        #            pad_inches=0)
        plt.show()
