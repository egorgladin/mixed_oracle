import numpy as np


def get_grad_approx(x, tau, e, oracle, dim):
    val = oracle(x)
    return val, (oracle(x + tau * e) - val) * e / tau  # dim *


def sample_spherical(dim, seed_me=None):
    if seed_me:
        np.random.seed(seed_me)
    vec = np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return np.expand_dims(vec, axis=1)


def ardd(x_0, oracle, L, N, tau):
    """Use ARDD to minimize g(y)."""
    dim = len(x_0)
    pts = [x_0.copy()]
    vals = []

    z = x_0.copy()
    w = x_0.copy()
    for k in range(N):
        e = sample_spherical(dim, k)

        t = 2 / (k + 2)
        x = t * z + (1 - t) * w

        val, grad_approx = get_grad_approx(x, tau, e, oracle, dim)
        vals.append(val)
        w = x - grad_approx / (2 * L)
        pts.append(w)

        alpha = (k + 1) / (96 * dim**2 * L)
        z -= alpha * grad_approx

    vals.append(oracle(pts[-1]))
    return pts, vals


def arddsc(x_0, oracle, L, mu, N, tau, last=False):
    """Use ARDDsc to minimize g(y)."""
    pts = [x_0.copy()]
    vals = [oracle(x_0)]
    dim = len(x_0)

    u = x_0
    a = 384 * dim**2
    if last:
        factor = 1000
    else:
        factor = 1000000
    N_0 = int(np.ceil(np.sqrt(8 * a * L / (mu * factor))))

    for k in range(N):
        inner_pts, inner_vals = ardd(u, oracle, L, N_0, tau)
        pts += inner_pts[1:]
        vals += inner_vals[1:]
        u = inner_pts[-1]

    return pts, vals
