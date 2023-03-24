# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:05:18 2023

@author: Cedric
"""
import timeit
from typing import Callable, Union

import numpy as np
from scipy.optimize import minimize


def f(x: float) -> float:
    """
    Function given by excercise 1

    Parameters
    ----------
    x : float
        input value 

    Returns
    -------
    float
        retuns the float value of the defined function

    """
    return 300 + 50 * x - 4 * x ** 2 + 15 * x ** 3 - 20 * x ** 4 + 0.5 * x ** 5 + x ** 6


def df(x: float) -> float:
    """
    First derivative of function given by excercise 1

    Parameters
    ----------
    x : float
        input value 

    Returns
    -------
    float
        retuns the float value of the defined function

    """
    return 50 - 8 * x + 45 * x ** 2 - 80 * x ** 3 + 2.5 * x ** 4 + 6 * x ** 5


def ddf(x: float) -> float:
    """
    Second derivative of function given by excercise 1

    Parameters
    ----------
    x : float
        input value 

    Returns
    -------
    float
        retuns the float value of the defined function

    """
    return -8 + 90 * x - 240 * x ** 2 + 10 * x ** 3 + 30 * x ** 4


def newton_rhapson(f: Callable[[float], float], df: Callable[[float], float], x0, tol, max_iter: int = 1000,
                   num_iter: int = 1
                   ):
    """
    Newthon rhapson method impemented in recursive form

    Parameters
    ----------
    f : Function
    df : Derivative of function
    tol: Tolerance to be reached between f(x_o) and zero
    max_iter : Maximum iteration cycles
    num_iter : Current iteration method

    Returns
    -------
    float
        retuns (x_0,num_iter) x_0 value and number of iterations used value or an error message

    """
    if abs(f(x0)) < tol:
        return x0, num_iter

    if max_iter > 0:
        x = x0 - f(x0) / df(x0)
        return newton_rhapson(f, df, x, tol, max_iter - 1, num_iter + 1)
    else:
        print("More iterations needed")
        print(f"Current vaule x_0 = {x0}")
        print(f"Number of iterations run {num_iter}")
        print(f"Absolute error f(x_0) {abs(f(x0))}")
        print(f"Tolerance set tol={tol}")
        return False


def newton_rhapson_time(repeats: int = 3, iterations: int = 10000, x_0=0):
    SETUP_CODE = '''
from __main__ import newton_rhapson, df, ddf
'''

    TEST_CODE = f'''
newton_rhapson(df, ddf, {x_0}, 1e-3)'''

    # timeit.repeat statement
    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=repeats,
                          number=iterations)

    print(f"Code was executed {iterations} times")
    print(f"with mean execution time of {np.mean(times) / iterations} seconds")
    return np.mean(times) / iterations


def nummerical_1d_diff(f, x, h: float = 1e-3):
    """
    Numerical 1 Dimensional central value differentiation around given x value f(x)
    df(x) = f(x+h)-f(x-h)/2h
    Parameters
    ----------
    f : function
    x : x value
    h : diff size

    Returns
    -------
    float value with numerical differentiation value at point x
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def gradient_descent_with_df(df, x_0, max_iter: int = 1000, tol: float = 1e-3, gamma: float = 1e-3):
    next_x = x_0
    for _i in range(max_iter):
        current_x = next_x
        next_x = current_x - gamma * df(current_x)
        step = next_x - current_x
        if abs(step) <= tol:
            break
    return next_x, _i + 1


def gradient_descent(f, x_0, max_iter: int = 1000, tol: float = 1e-3, gamma: float = 1e-3):
    next_x = x_0
    for _i in range(max_iter):
        current_x = next_x
        next_x = current_x - gamma * nummerical_1d_diff(f, current_x, 1e-3)
        step = next_x - current_x
        if abs(step) <= tol:
            break
    return next_x, _i + 1


def gradient_descent_with_df_time(repeats: int = 3, iterations: int = 10000, x_0=0):
    SETUP_CODE = '''
from __main__ import gradient_descent_with_df, df
'''

    TEST_CODE = f'''
gradient_descent_with_df(df, {x_0}, 1000, gamma=1e-4)'''

    # timeit.repeat statement
    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=repeats,
                          number=iterations)

    print(f"Code was executed {iterations} times")
    print(f"with mean execution time of {np.mean(times) / iterations} seconds")
    return np.mean(times) / iterations


def gradient_descent_time(repeats: int = 3, iterations: int = 10000, x_0=0):
    """
    Measure the execution time of the gradient_descent function.

    This function uses the timeit module to run the gradient_descent function
    multiple times and calculates the mean execution time.

    :param repeats: The number of times the whole experiment is repeated, defaults to 3.
    :type repeats: int, optional
    :param iterations: The number of iterations for each repeat, defaults to 10000.
    :type iterations: int, optional
    :param x_0: The initial value of x for the gradient_descent function, defaults to 0.
    :type x_0: int or float, optional
    :return: The mean execution time (in seconds) of the gradient_descent function.
    :rtype: float

    .. code-block:: python

        # Example usage:
        mean_time = gradient_descent_time(repeats=3, iterations=10000, x_0=0)
        print(f"Mean execution time: {mean_time} seconds")
    """
    SETUP_CODE = '''
from __main__ import gradient_descent, f
'''

    TEST_CODE = f'''
gradient_descent(f, {x_0}, 1000, gamma=1e-4)'''

    # timeit.repeat statement
    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=repeats,
                          number=iterations)

    print(f"Code was executed {iterations} times")
    print(f"with mean execution time of {np.mean(times) / iterations} seconds")
    return np.mean(times) / iterations


def minimize_time(repeats: int = 3, iterations: int = 1000, x_0=0):
    SETUP_CODE = '''
from __main__ import minimize, f
'''

    TEST_CODE = f'''
minimize(f, {x_0}, tol=1e-3)'''

    # timeit.repeat statement
    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=repeats,
                          number=iterations)

    print(f"Code was executed {iterations} times")
    print(f"with mean execution time of {np.mean(times) / iterations} seconds")
    return np.mean(times) / iterations


def ex_1():
    print(" -Ex 1.a) run from x_0 = 0")
    x_0, it = newton_rhapson(df, ddf, 0, 1e-3)
    print(f"x_0 = {x_0:.2f}, function value f(x_0)={f(x_0):.3f}, number of iterations needed = {it}")
    # t_newton_1 = newton_rhapson_time(x_0=0)

    print("\n -Ex 1.b) run from x_0 = -10")
    x_0, it = newton_rhapson(df, ddf, -10, 1e-3)
    print(f"x_0 = {x_0:.3f}, function value f(x_0)={f(x_0):.3f}, number of iterations needed = {it}")
    t_newton_2 = newton_rhapson_time(x_0=-10)
    print(f"x_0 = {x_0} seems to be the global minimum")

    print("\n -Ex 1.c) run from x_0 = 0")
    x_0, it = gradient_descent_with_df(df, 0, 100)
    print(f"x_0 = {x_0:.3f}, function value f(x_0)={f(x_0):.3f}, number of iterations needed = {it}")
    print("It does not converge after 100 iterations")

    print("\n -Ex 1.d) run from x_0 = 0")
    x_0, it = gradient_descent_with_df(df, 0, 1000, gamma=1e-4 * 5)
    print(f"x_0 = {x_0:.3f}, function value f(x_0)={f(x_0):.3f}, number of iterations needed = {it}")
    print(f"It does converge after {it} iterations and gamma = {1e-4 * 5}")
    t_grad_1 = gradient_descent_with_df_time(x_0=0)

    print("\n -Ex 1.e) run from x_0 = 0")
    x_0, it = gradient_descent(f, 0, 10000, gamma=1e-4 * 5)
    print(f"x_0 = {x_0:.3f}, function value f(x_0)={f(x_0):.3f}, number of iterations needed = {it}")
    print(f"It does converge after {it} iterations and gamma = {1e-4 * 5}")
    t_grad_2 = gradient_descent_time(x_0=0)

    print("\n -Ex 1.f) run scipy minimize from x_0 = 0")
    res = minimize(f, 0, tol=1e-3)
    x_0 = res.x[0]
    it = res.nit
    print(res)
    print(f"x_0 = {x_0:.3f}, function value f(x_0)={f(x_0):.3f}, number of iterations needed = {it}")
    t_minimize = minimize_time(x_0=0)
    print("\n -Comparison between the different function call times:\n")
    print(f"Newton-Rhapson: {t_newton_2:.6f}")
    print(f"Gradient descent analytical: {t_grad_1:.6f}")
    print(f"Gradient descent approx: {t_grad_2:.6f}")
    print(f"Scipy minimize: {t_minimize:.6f}")


def pdf(x: Union[float, np.ndarray], f: float, mu_s: float, sigma_s: float) -> Union[float, np.ndarray]:
    """
    Calculate the probability density function (PDF) of a mixture model that combines a Gaussian distribution
    and a uniform distribution.

    :param x: The input value(s) for which to compute the PDF. Can be a single float or a NumPy array.
    :type x: float or numpy.ndarray
    :param f: The mixing coefficient, representing the weight of the Gaussian distribution in the mixture model.
    :type f: float
    :param mu_s: The mean of the Gaussian distribution.
    :type mu_s: float
    :param sigma_s: The standard deviation of the Gaussian distribution.
    :type sigma_s: float
    :return: The computed probability density value(s) for the given input value(s).
    :rtype: float or numpy.ndarray

    .. code-block:: python

        # Example usage:
        x = np.array([1.0, 2.0, 3.0])
        f = 0.6
        mu_s = 2.0
        sigma_s = 1.0

        p = pdf(x, f, mu_s, sigma_s)
        print(p)
    """
    g = 1 / (sigma_s * np.sqrt(2 * np.pi)) * np.exp(-(x - mu_s) ** 2 / (2 * sigma_s ** 2))
    u = 1 / 10
    p = f * g + (1 - f) * u
    return p


def likelihood_gauss(params, data: Union[float, np.ndarray]) -> float:
    """
    Calculate the likelihood of a given distribution for a set of data points.

    :param distribution: The probability density function (for continuous distributions) or
                         probability mass function (for discrete distributions) of the distribution.
    :type distribution: Callable[[Union[float, numpy.ndarray]], float]
    :param data: A list of data points for which to calculate the likelihood.
    :type data: List[float]
    :return: The computed likelihood of the distribution for the given data points.
    :rtype: float

    .. code-block:: python

        # Example usage with a Gaussian distribution:
        import scipy.stats

        def gaussian_pdf(x: float) -> float:
            mu = 2.0
            sigma = 1.0
            return scipy.stats.norm.pdf(x, mu, sigma)

        data_points = [1.0, 2.5, 3.0]
        likelihood_value = likelihood(gaussian_pdf, data_points)
        print(likelihood_value)
    """
    mu_s, sigma_s, f = params
    return np.prod(pdf(data, f, mu_s, sigma_s))


def neg_log_likelihood_gauss(params, data: Union[float, np.ndarray]) -> float:
    """
    Calculate the negative log likelihood of a given distribution for a set of data points.

    :param distribution: The probability density function (for continuous distributions) or
                         probability mass function (for discrete distributions) of the distribution.
    :type distribution: Callable[[Union[float, numpy.ndarray]], float]
    :param data: A list of data points for which to calculate the likelihood.
    :type data: List[float]
    :return: The computed likelihood of the distribution for the given data points.
    :rtype: float

    .. code-block:: python

        # Example usage with a Gaussian distribution:
        import scipy.stats

        def gaussian_pdf(x: float) -> float:
            return scipy.stats.norm.pdf(x, mu, sigma)


        data_points = [1.0, 2.5, 3.0]
        mu = 2.0
        sigma = 1.0
        params = [mu, sigma]
        likelihood_value = likelihood(gaussian_pdf, params, data_points)
        print(likelihood_value)
    """
    mu_s, sigma_s, f = params
    if (f >= 0) & (f <= 1):
        return -np.sum(np.log(pdf(data, f, mu_s, sigma_s)))
    else:
        print("f is out of bound")
        return np.infty


def ex_2(data):
    print(" -Ex 2.a) run pdf with example values:")
    print("x = 4.0, mu = 5, sigma = 0.75, f = 0.5")
    print(f"res = {pdf(4.0, 0.5, 5, 0.75):.3f}\n")

    print(" -Ex 2.b) run liklihood functioon with example values:")
    print("x = [4.0, 5.0, 6.0, 7.0], mu = 5, sigma = 0.75, f = 0.5")
    params = [5.0, 0.75, 0.5]
    x = np.asarray([4.0, 5.0, 6.0, 7.0])
    print(f"res = {likelihood_gauss(params, x)}")
    print("run liklyhood with data, mu = 5, sigma = 0.75, f = 0.5")
    print(f"res = {likelihood_gauss(params, data)}\n")

    print(" -Ex 2.c) run neg logliklihood functioon with example values:")
    print("x = data, mu = 5, sigma = 0.75, f = 0.5")
    print(f"res = {neg_log_likelihood_gauss(data=data, params=[5, 0.75, 0.5]):.3f}\n")
    print("Minimize the negative log liklihood function f bounded (0,1)")
    bnds = ((None, None), (None, None), (0, 1))
    res_liklihood = minimize(likelihood_gauss, x0=(5.0, 0.75, 0.4), bounds=bnds, args=(data))
    res_neg_log = minimize(neg_log_likelihood_gauss, x0=(5.0, 0.75, 0.5), bounds=bnds, args=(data))
    print(res_liklihood)
    print("As we can see if we just optimize the liklihood we get no real result ")
    print("since the np.prod tends against zero (not numerical stable with so ")
    print("many small values)\n")
    print("Using the negative log liklihood to minimize:")
    print(res_neg_log)
    print(f"Here we get a usefull result with:")
    print(f"mu = {res_neg_log.x[0]:.3f}")
    print(f"sigma = {res_neg_log.x[1]:.3f}")
    print(f"f = {res_neg_log.x[2]:.3f}")


if __name__ == "__main__":
    ex_1()
    data = np.loadtxt("signal_and_background.txt")
    ex_2(data)
