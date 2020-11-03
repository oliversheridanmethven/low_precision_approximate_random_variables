"""
Author:

    Oliver Sheridan-Methven, October 2020.

Description:

    The various plots for the article.
"""

import plotting_configuration
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, ncx2
from scipy.integrate import quad as integrate
from approximate_random_variables.approximate_gaussian_distribution import construct_piecewise_constant_approximation, construct_symmetric_piecewise_polynomial_approximation, rademacher_approximation
from approximate_random_variables.approximate_non_central_chi_squared import construct_inverse_non_central_chi_squared_interpolated_polynomial_approximation
from mpmath import mp, mpf
from timeit import default_timer as timer
from functools import wraps
from progressbar import progressbar
import json
from statsmodels.distributions.empirical_distribution import ECDF as ecdf

norm_inv = norm.ppf


def time_function(func):
    """ A decorator to time a function. """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = timer()
        results = func(*args, **kwargs)
        elapsed_time = timer() - start_time
        return results, elapsed_time

    return wrapper


def plot_piecewise_linear_gaussian_approximation(savefig=False, plot_from_json=True):
    if plot_from_json:
        with open('piecewise_linear_gaussian_approximation.json', "r") as input_file:
            results = json.load(input_file)
        u, exact, approximate = results['uniforms'], results['exact'], results['approximate']
    else:
        u = np.linspace(0, 1, 1000)[1:-1]
        norm_inv_approx = construct_symmetric_piecewise_polynomial_approximation(norm.ppf, n_intervals=5, polynomial_order=1)
        exact, approximate = norm_inv(u), norm_inv_approx(u)
    plt.clf()
    plt.plot(u, exact, 'k--', label=r'$\Phi^{-1}(x)$')
    plt.plot(u, approximate, 'k,', label=r'__nolegend__')
    plt.plot([], [], 'k-', label=r'$\widetilde{\Phi}^{-1}(x)$')
    plt.xlabel(r"$x$")
    plt.xticks([0, 1])
    plt.yticks([-3, 0, 3])
    plt.ylim(-3, 3)
    plt.legend(frameon=False)
    if savefig:
        plt.savefig('piecewise_linear_gaussian_approximation.pdf', format='pdf', bbox_inches='tight', transparent=True)
        if not plot_from_json:
            with open('piecewise_linear_gaussian_approximation.json', "w") as output_file:
                output_file.write(json.dumps({'uniforms': u.tolist(), 'exact': norm_inv(u).tolist(), 'approximate': norm_inv_approx(u).tolist()}, indent=4))

def plot_piecewise_linear_gaussian_approximation_pdf(savefig=False, plot_from_json=True):
    f = construct_symmetric_piecewise_polynomial_approximation(norm.ppf, n_intervals=4 + 1, polynomial_order=1)
    u = np.linspace(0, 1, 10000)
    z = norm.ppf(u)
    u = np.linspace(0, 1, 2000000)  # Lots of points needed to reduce noise.
    g = ecdf(f(np.array(u)))
    p = np.diff(g(z)) / np.diff(z)
    zm = (z[1:] + z[:-1]) / 2.0
    plt.clf()
    # working out the breakpoints.
    b = [0] + list(np.argwhere(~(np.fabs(np.diff(p)) < 0.001)).transpose()[0]) + [len(zm)]
    for i, j in zip(b[:-1], b[1:]):
        plt.plot(zm[i + 1:j], p[i + 1:j], 'k-', label='__nolegend__')
    plt.plot([], [], 'k-', label=r'$\rho(x)$')
    plt.plot(zm, p, 'k:', label=r'__nolegend__')
    lines = plt.gca().lines[-1]  # For storing a compressed version of the data.
    plt.plot(z, norm.pdf(z), 'k--', label=r'$\phi(x)$')
    plt.xlabel(r'$x$')
    plt.yticks([])
    plt.xticks([-3, 0, 3])
    # plt.tick_params(length=20)
    plt.ylim(0, 1.1 * max(p))
    plt.xlim(*plt.xticks()[0][[0, -1]])
    plt.legend(frameon=False, loc='upper left', borderaxespad=0)
    # finding the zero regions
    i = np.logical_and(np.logical_and(zm > -2, zm < 0), p == 0)
    i = np.argwhere(i).transpose()[0]
    i = [i[0]] + list(i[1:][np.diff(i) > 10])
    xz = zm[i]
    xz = np.concatenate([xz, -xz])
    y = max(p)
    plt.gca().annotate(r'$\rho(x) = 0$', xy=(xz[0], 0), xytext=(0, 0.2 * y), arrowprops=dict(arrowstyle="-|>", color='k', linewidth=0.5), ha='center')
    for j in range(1, len(xz)):
        plt.gca().annotate(r'\phantom{$\tilde{\phi}(x) = 0$}', xy=(xz[j], 0), xytext=(0, 0.2 * y), arrowprops=dict(arrowstyle="-|>", color='k', linewidth=0.5), ha='center')
    if savefig:
        plt.savefig('piecewise_linear_gaussian_approximation_pdf.pdf', format='pdf', bbox_inches='tight', transparent=True)
        if not plot_from_json:
            with open('piecewise_linear_gaussian_approximation_pdf.json', "w") as output_file:
                output_file.write(json.dumps({'z': lines.get_xdata().tolist(), 'rho':lines.get_ydata().tolist()}, indent=4))



def produce_geometric_brownian_motion_paths(dt, approx=None, precision=None):
    """
    Perform path simulations of a geometric Brownian motion.
    :param dt: Float. (Fraction of time).
    :param approx: List.
    :param precision: Int.
    :return: List. [x_fine_exact, x_coarse_exact, x_fine_approx, x_coarse_approx]
    """
    assert isinstance(dt, float) and np.isfinite(dt) and dt > 0 and (1.0 / dt).is_integer()
    assert approx is not None
    # The parameters.

    x_0 = 1.0
    mu = 0.05
    sigma = 0.2
    T = 1.0

    dt = dt * T
    t_fine = dt
    t_coarse = 2 * dt
    sqrt_t_fine = t_fine ** 0.5
    w_coarse_exact = 0.0
    w_coarse_approx = 0.0

    x_fine_exact = x_0
    x_coarse_exact = x_0
    x_fine_approx = x_0
    x_coarse_approx = x_0
    n_fine = int(1.0 / dt)

    update_coarse = False

    x_0, mu, sigma, T, dt, t_fine, t_coarse, sqrt_t_fine, w_coarse_exact, w_coarse_approx = [mpf(i) for i in [x_0, mu, sigma, T, dt, t_fine, t_coarse, sqrt_t_fine, w_coarse_exact, w_coarse_approx]]
    fabs = mp.fabs

    path_update = None
    if method == 'euler_maruyama':
        path_update = lambda x, w, t: x + mu * x * t + sigma * x * w
    elif method == 'milstein':
        path_update = lambda x, w, t: x + mu * x * t + sigma * x * w + 0.5 * sigma * sigma * (w * w - t)
    assert path_update is not None

    for n in range(n_fine):
        u = np.random.uniform()
        z_exact = norm.ppf(u)
        z_approx = approx(u)
        z_approx = z_approx if isinstance(z_approx, float) else z_approx[0]
        w_fine_exact = sqrt_t_fine * z_exact
        w_fine_approx = sqrt_t_fine * z_approx
        w_coarse_exact += w_fine_exact
        w_coarse_approx += w_fine_approx

        x_fine_exact = path_update(x_fine_exact, w_fine_exact, t_fine)
        x_fine_approx = path_update(x_fine_approx, w_fine_approx, t_fine)
        if update_coarse:
            x_coarse_exact = path_update(x_coarse_exact, w_coarse_exact, t_coarse)
            x_coarse_approx = path_update(x_coarse_approx, w_coarse_approx, t_coarse)
            w_coarse_exact *= 0.0
            w_coarse_approx *= 0.0
        update_coarse = not update_coarse  # We toggle to achieve pairwise summation.
    assert not update_coarse  # This should have been the last thing we did.

    return [x_fine_exact, x_coarse_exact, x_fine_approx, x_coarse_approx, x_fine_approx_kahan, x_coarse_approx_kahan]



deltas = [2.0 ** -i for i in range(1, 7)]
inverse_norm = norm.ppf
piecewise_constant = construct_piecewise_constant_approximation(inverse_norm, n_intervals=1024)
piecewise_linear = construct_symmetric_piecewise_polynomial_approximation(inverse_norm, n_intervals=16, polynomial_order=1)
piecewise_cubic = construct_symmetric_piecewise_polynomial_approximation(inverse_norm, n_intervals=16, polynomial_order=3)
approximations = {'constant': piecewise_constant, 'linear': piecewise_linear, 'cubic': piecewise_cubic, 'rademacher': rademacher_approximation}

results = {method: {term: {} for term in ['original'] + list(approximations.keys())} for method in methods}  # Store the values of delta and the associated data.
time_per_level = 2.0
paths_min = 64
for method in results:
    for approx_name, approx in approximations.items():
        for dt in deltas:
            _, elapsed_time_per_path = time_function(produce_geometric_brownian_motion_paths)(dt, method, approx)
            paths_required = int(time_per_level / elapsed_time_per_path)
            if paths_required < paths_min:
                print("More time required for {} and {} with dt={}".format(method, approx_name, dt))
                break

            originals, corrections = [[None for i in range(paths_required)] for j in range(2)]
            for path in range(paths_required):
                x_fine_exact, x_coarse_exact, x_fine_approx, x_coarse_approx = produce_geometric_brownian_motion_paths(dt, method, approx)
                originals[path] = x_fine_exact - x_coarse_exact
                corrections[path] = min((x_fine_exact - x_coarse_exact) - (x_fine_approx - x_coarse_approx), (x_fine_exact - x_fine_approx) - (x_coarse_exact - x_coarse_approx), sum([x_fine_exact, -x_coarse_exact, -x_fine_approx, x_coarse_approx]))  # might need revising for near machine precision.
            originals, corrections = [[j ** 2 for j in i] for i in [originals, corrections]]
            for name, values in [['original', originals], [approx_name, corrections]]:
                mean = np.mean(values)
                std = np.std(values) / (len(values) ** 0.5)
                [mean, std] = [float(i) for i in [mean, std]]
                results[method][name][dt] = [mean, std]



if __name__ == '__main__':
    plot_params = dict(savefig=True, plot_from_json=True)
