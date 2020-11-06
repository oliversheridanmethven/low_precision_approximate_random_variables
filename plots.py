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
    u = np.linspace(0, 1, 10000000)  # Lots of points needed to reduce noise.
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
                output_file.write(json.dumps({'z': lines.get_xdata().tolist(), 'rho': lines.get_ydata().tolist()}, indent=4))


def produce_geometric_brownian_motion_paths(dt, approx=None, precision=None):
    """
    Perform path simulations of a geometric Brownian motion.
    :param dt: Float. (Fraction of time).
    :param approx: Function.
    :param precision: Int.
    :return: List. [x_fine_exact, x_coarse_exact, x_fine_approx, x_coarse_approx]
    """
    assert isinstance(dt, float) and np.isfinite(dt) and dt > 0 and (1.0 / dt).is_integer()
    assert isinstance(precision, int) and precision > 0
    assert approx is not None
    mp.prec = precision

    # The parameters.
    x_0 = 1.0
    mu = 0.05
    sigma = 0.2
    mu_approx = mpf(mu)
    sigma_approx = mpf(sigma)
    T = 1.0

    dt = dt * T
    t_fine = dt
    t_fine_approx = mpf(dt)
    t_coarse = 2 * t_fine
    t_coarse_approx = 2 * t_fine_approx
    sqrt_t_fine = t_fine ** 0.5
    sqrt_t_fine_approx = t_fine_approx ** 0.5
    w_coarse_exact = 0.0
    w_coarse_approx = mpf(0.0)

    x_fine_exact = x_0
    x_coarse_exact = x_0
    x_fine_approx = mpf(x_0)
    x_coarse_approx = mpf(x_0)
    x_fine_approx_kahan = mpf(x_0)
    x_coarse_approx_kahan = mpf(x_0)

    n_fine = int(1.0 / dt)

    update_coarse = False

    def path_update_exact(x, w, t):
        return x + mu * x * t + sigma * x * w

    def path_update_approx(x, w, t):
        return x + mu_approx * x * t + sigma_approx * x * w

    def path_update_approx_kahan(x, w, t, c):
        dx = mu_approx * x * t + sigma_approx * x * w
        dx_c = dx - c
        a = x + dx_c
        c = (a - x) - dx_c
        return a, c

    fine_compensation = mpf(0)
    coarse_compensation = mpf(0)

    for n in range(n_fine):
        u = np.random.uniform()
        z_exact = norm.ppf(u)
        z_approx = approx(u)
        z_approx = z_approx if isinstance(z_approx, float) else z_approx[0]
        z_approx = mpf(z_approx)
        w_fine_exact = sqrt_t_fine * z_exact
        w_fine_approx = sqrt_t_fine_approx * z_approx
        w_coarse_exact += w_fine_exact
        w_coarse_approx += w_fine_approx

        x_fine_exact = path_update_exact(x_fine_exact, w_fine_exact, t_fine)
        x_fine_approx = path_update_approx(x_fine_approx, w_fine_approx, t_fine_approx)
        x_fine_approx_kahan, fine_compensation = path_update_approx_kahan(x_fine_approx_kahan, w_fine_approx, t_fine_approx, fine_compensation)
        if update_coarse:
            x_coarse_exact = path_update_exact(x_coarse_exact, w_coarse_exact, t_coarse)
            x_coarse_approx = path_update_approx(x_coarse_approx, w_coarse_approx, t_coarse_approx)
            x_coarse_approx_kahan, coarse_compensation = path_update_approx_kahan(x_coarse_approx_kahan, w_coarse_approx, t_coarse_approx, coarse_compensation)
            w_coarse_exact *= 0.0
            w_coarse_approx *= 0.0
        update_coarse = not update_coarse  # We toggle to achieve pairwise summation.
    if n_fine > 1:
        assert not update_coarse  # This should have been the last thing we did.

    return [x_fine_exact, x_coarse_exact, x_fine_approx, x_coarse_approx, x_fine_approx_kahan, x_coarse_approx_kahan]


def plot_error_model_2_way_variances(savefig=False, plot_from_json=True):
    approximation_terms = ['approximation', 'kahan']
    if plot_from_json:
        with open('two_way_variances.json', "r") as file:
            results = json.load(file)
        results = {k: {x: {float(a): b for a, b in y.items()} for x, y in v.items()} for k, v in results.items()}
    else:
        deltas = [2.0 ** -i for i in range(0, 6)]
        precisions = [7, 10, 16, 23]
        approximation = construct_symmetric_piecewise_polynomial_approximation(norm.ppf, n_intervals=16, polynomial_order=3)
        results = {precision: {term: {} for term in approximation_terms} for precision in precisions}
        time_per_level = 2.0
        paths_min = 64
        for precision in results:
            for dt in deltas:
                _, elapsed_time_per_path = time_function(produce_geometric_brownian_motion_paths)(dt, approximation, precision)
                paths_required = int(time_per_level / elapsed_time_per_path)
                if paths_required < paths_min:
                    print("More time required for {} with dt={}".format(precision, dt))
                    break

                errors, errors_kahan = [[None for i in range(paths_required)] for j in range(2)]
                for path in range(paths_required):
                    x_fine_exact, x_coarse_exact, x_fine_approx, x_coarse_approx, x_fine_approx_kahan, x_coarse_approx_kahan = produce_geometric_brownian_motion_paths(dt, approximation, precision)
                    errors[path] = x_fine_exact - x_fine_approx
                    errors_kahan[path] = x_fine_exact - x_fine_approx_kahan
                errors, errors_kahan = [[j ** 2 for j in i] for i in [errors, errors_kahan]]
                for name, values in zip(approximation_terms, [errors, errors_kahan]):
                    mean = np.mean(values)
                    std = np.std(values) / (len(values) ** 0.5)
                    [mean, std] = [float(i) for i in [mean, std]]
                    results[precision][name][dt] = [mean, std]

    precisions = sorted(results.keys())
    precision_markers = {p: m for p, m in zip(precisions, (i for i in ['d', 's', 'o', 'v']))}
    deltas = list(list(list(results.items())[0][1].items())[0][1].keys())
    for term in approximation_terms:
        plt.clf()
        for precision in results:
            marker = precision_markers[precision]
            x, y = zip(*results[precision][term].items())
            y, y_std = list(zip(*y))
            y_error = 1 * np.array(y_std)
            plt.errorbar(x, y, y_error, None, 'k{}:'.format(marker))
        plt.xscale('log', basex=2)
        plt.yscale('log', basey=2)
        plt.xlabel(r'Time increment $\delta$')
        plt.ylabel('Variance')
        plt.ylim(2 ** -35, 2 ** -5)
        plt.yticks([2 ** i for i in range(-35, -5 + 1, 5)])
        plt.xticks(deltas)
        if savefig:
            plt.savefig('two_way_variance_{}.pdf'.format(term), format='pdf', bbox_inches='tight', transparent=True)
            if not plot_from_json:
                with open('two_way_variances.json', "w") as file:
                    file.write(json.dumps(results, indent=4))


def plot_error_model_4_way_variances(savefig=False, plot_from_json=True):
    terms = ['originals', 'corrections_32', 'corrections_16', 'corrections_16_kahan', 'approx_estimator_16', 'approx_estimator_16_kahan']
    min_dts = {term: 2.0 ** -l for term, l in zip(terms, [20, 20, 8, 12, 12, 15])}
    markers = {term: m for term, m in zip(terms, ['o', 'v', 's', 'd', 'x', '^'])}
    if plot_from_json:
        with open('four_way_variance.json', "r") as file:
            results = json.load(file)
        results = {k: {float(x): y for x, y in v.items()} for k, v in results.items()}
    else:
        df = pd.read_csv('kahan_data.csv', header=None)
        df.columns = 'dt,x_fine_exact_64,x_coarse_exact_64,x_fine_approx_32,x_coarse_approx_32,x_fine_approx_16,x_coarse_approx_16,x_fine_approx_16_kahan,x_coarse_approx_16_kahan'.split(',')
        dt = df['dt']
        originals = df['x_fine_exact_64'] - df['x_coarse_exact_64']
        corrections_32 = (df['x_fine_exact_64'] - df['x_coarse_exact_64']) - (df['x_fine_approx_32'] - df['x_coarse_approx_32'])
        corrections_16 = (df['x_fine_exact_64'] - df['x_coarse_exact_64']) - (df['x_fine_approx_16'] - df['x_coarse_approx_16'])
        corrections_16_kahan = (df['x_fine_exact_64'] - df['x_coarse_exact_64']) - (df['x_fine_approx_16_kahan'] - df['x_coarse_approx_16_kahan'])
        approx_estimator_16 = df['x_fine_approx_16'] - df['x_coarse_approx_16']
        approx_estimator_16_kahan = df['x_fine_approx_16_kahan'] - df['x_coarse_approx_16_kahan']
        dfs = [pd.concat([dt, df], axis=1) for df in [originals, corrections_32, corrections_16, corrections_16_kahan, approx_estimator_16, approx_estimator_16_kahan]]
        for df in dfs: df.columns = ['dt', 'correction']
        results = {term: df for term, df in zip(terms, dfs)}
        for term, df in results.items():
            variances = df.groupby('dt')['correction'].apply(np.var)
            variances = variances[variances.index >= min_dts[term]]
            results[term] = variances.to_dict()

    plt.clf()
    for term, variance in results.items():
        plt.plot(*zip(*variance.items()), 'k{}:'.format(markers[term]))
    plt.xscale('log', basex=2)
    plt.yscale('log', basey=2)
    plt.ylim(2 ** -35, 2 ** -10)
    plt.yticks([2 ** i for i in range(-35, -10 + 1, 5)])
    plt.xlim(2 ** -20, 2 ** 0)
    plt.ylabel('Variance')
    plt.xlabel(r'Fine time increment $\delta^{\mathrm{f}}$')

    if savefig:
        plt.savefig('four_way_variance.pdf', format='pdf', bbox_inches='tight', transparent=True)
        if not plot_from_json:
            with open('four_way_variance.json', "w") as file:
                file.write(json.dumps(results, indent=4))


def plot_error_model_4_way_savings(savefig=False, plot_from_json=True):
    terms = ['originals', 'corrections_32', 'corrections_16', 'corrections_16_kahan', 'approx_estimator_16', 'approx_estimator_16_kahan']
    min_dts = {term: 2.0 ** -l for term, l in zip(terms, [20, 20, 8, 12, 12, 15])}
    markers = {term: m for term, m in zip(terms, ['o', 'v', 's', 'd', 'x', '^'])}
    if plot_from_json:
        with open('four_way_savings.json', "r") as file:
            savings = json.load(file)
        savings = {k: {x: {float(a): b for a, b in y.items()} for x, y in v.items()} for k, v in savings.items()}
    else:
        df = pd.read_csv('kahan_data.csv', header=None)
        df.columns = 'dt,x_fine_exact_64,x_coarse_exact_64,x_fine_approx_32,x_coarse_approx_32,x_fine_approx_16,x_coarse_approx_16,x_fine_approx_16_kahan,x_coarse_approx_16_kahan'.split(',')
        dt = df['dt']
        originals = df['x_fine_exact_64'] - df['x_coarse_exact_64']
        corrections_32 = (df['x_fine_exact_64'] - df['x_coarse_exact_64']) - (df['x_fine_approx_32'] - df['x_coarse_approx_32'])
        corrections_16 = (df['x_fine_exact_64'] - df['x_coarse_exact_64']) - (df['x_fine_approx_16'] - df['x_coarse_approx_16'])
        corrections_16_kahan = (df['x_fine_exact_64'] - df['x_coarse_exact_64']) - (df['x_fine_approx_16_kahan'] - df['x_coarse_approx_16_kahan'])
        approx_estimator_16 = df['x_fine_approx_16'] - df['x_coarse_approx_16']
        approx_estimator_16_kahan = df['x_fine_approx_16_kahan'] - df['x_coarse_approx_16_kahan']
        dfs = [pd.concat([dt, df], axis=1) for df in [originals, corrections_32, corrections_16, corrections_16_kahan, approx_estimator_16, approx_estimator_16_kahan]]
        for df in dfs: df.columns = ['dt', 'correction']
        results = {term: df for term, df in zip(terms, dfs)}
        for term, df in results.items():
            variances = df.groupby('dt')['correction'].apply(np.var)
            variances = variances[variances.index >= min_dts[term]]
            results[term] = variances.to_dict()

        savings = {term: {k: {} for k in ['speedup', 'efficiency']} for term in terms if 'correction' in term}
        time_savings = {'32': 7.0, '16': 14.0, '16_kahan': 10.0}
        for term in savings:
            levels = sorted(results[term].keys())
            for level in levels:
                possible_saving = next((v for k, v in time_savings.items() if term.endswith(k)))
                original_variance = results['originals'][level]
                correction_variance = results[term][level]
                variance_reduction = correction_variance / original_variance
                savings[term]['speedup'][level], savings[term]['efficiency'][level] = speed_up_and_efficiency(variance_reduction, possible_saving)

    plt.clf()
    for term in savings:
        x, y = zip(*savings[term]['speedup'].items())
        plt.plot(x, y, 'k{}:'.format(markers[term]))
    plt.xscale('log', basex=2)
    plt.ylim(1, 12)
    plt.yticks([1] + list(range(2, 12 + 1, 2)))
    plt.ylabel('Speed up')
    plt.xlim(2 ** -20, 2 ** 0)
    plt.xlabel(r'Fine time increment $\delta^{\mathrm{f}}$')

    if savefig:
        plt.savefig('four_way_savings.pdf', format='pdf', bbox_inches='tight', transparent=True)
        if not plot_from_json:
            with open('four_way_savings.json', "w") as file:
                file.write(json.dumps(savings, indent=4))


def speed_up_and_efficiency(V, c):
    c = 1.0 / c
    C = 1.0 + c
    e = (1.0 + np.sqrt(V * C / c)) ** 2
    s = c * e
    m = np.sqrt(s / c)
    M = np.sqrt(s * V / C)
    return (1.0 / s, 100.0 / e)


if __name__ == '__main__':
    plot_params = dict(savefig=True, plot_from_json=True)
    plot_piecewise_linear_gaussian_approximation(**plot_params)
    plot_piecewise_linear_gaussian_approximation_pdf(**plot_params)
    plot_error_model_2_way_variances(**plot_params)
    plot_error_model_4_way_variances(**plot_params)
    plot_error_model_4_way_savings(**plot_params)
