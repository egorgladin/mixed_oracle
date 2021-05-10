import numpy as np
from vaidya import vaidya, get_init_polytope
from ardd import arddsc
from operator import itemgetter


def combined_method(x_0, y_0, dF_dx, F, vaidya_params, arddsc_params):
    d, eps, K, newton_steps = itemgetter('d', 'eps', 'K', 'newton_steps')(vaidya_params)
    L, mu, N, tau = itemgetter('L', 'mu', 'N', 'tau')(arddsc_params)

    def get_y_opt(x, y_0):
        oracle_y = lambda y: -F(x, y)
        pts = arddsc(y_0, oracle_y, L, mu, N, tau)
        return pts[-1], len(pts)

    def oracle_x(x, y_0):
        y_opt, n_grad_evals = get_y_opt(x, y_0)
        return dF_dx(x, y_opt), y_opt, n_grad_evals

    A_0, b_0 = get_init_polytope(d)

    xs, ys, aux_evals = vaidya(A_0, b_0, x_0, y_0, eps, K, oracle_x, newton_steps=newton_steps)
    return xs, ys, aux_evals
