import numpy as np
from vaidya import vaidya, get_init_polytope
from ardd import arddsc
from operator import itemgetter
from scipy.stats import entropy


def combined_method(x_0, y_0, dF_dx, F, vaidya_params, arddsc_params):
    d, eps, eta, K, newton_steps, stepsize = itemgetter('d', 'eps', 'eta', 'K', 'newton_steps', 'stepsize')(vaidya_params)
    L, mu, N, tau = itemgetter('L', 'mu', 'N', 'tau')(arddsc_params)

    def get_y_opt(x, y, last=False):
        oracle_y = lambda y: -F(x, y)
        pts, vals = arddsc(y, oracle_y, L, mu, N, tau, last=last)
        optimal_point = min(zip(pts, vals), key=lambda elem: elem[1])[0]
        return optimal_point, len(pts) - 1

    def oracle_x(x, y):
        y_opt, n_grad_evals = get_y_opt(x, y)
        return dF_dx(x, y_opt), y_opt, n_grad_evals

    A_0, b_0 = get_init_polytope(d)

    xs, ys, aux_evals = vaidya(A_0, b_0, x_0, y_0, eps, eta, K, oracle_x,
                               newton_steps=newton_steps, stepsize=stepsize)
    entr = [entropy(np.append(np.squeeze(x_), 1 - x_.sum())) for x_ in xs]
    best_idx = min(enumerate(entr), key=lambda elem: elem[1])[0]
    x_op = xs[best_idx]
    y_op, _ = get_y_opt(x_op, ys[best_idx], last=True)
    return xs, ys, aux_evals, y_op, best_idx
