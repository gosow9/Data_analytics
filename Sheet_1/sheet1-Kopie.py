# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:04:52 2023

@author: Cedric
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def unit_circle(x):
    return np.sqrt(1-x**2)


def poly_func(x, n=1):
    return x**n


def q_distribution(a, x):
    return a*x**(a-1)


def fitt_slope(x, b):
    return -0.5*x + b


def fitt_slope_poly(x, a, b):
    return a*x + b


def pdf_func(x, m, b):
    return b/(np.pi*(x-m)**2+b**2)


def inverse_CDF(y, m, b):
    return b*np.tan((y-0.5)*np.pi)+m


def print_stats(excercise, N_sim, estimate, exact):
    print(f"#########---{excercise}---################")
    print(f"# Random values: {len(N_sim)}")
    print(f"# Estimated Value: {estimate}")
    print(f"# Exact Value: {exact}")
    print(f"# Absolute Error: {abs(estimate-exact)}")
    print("###################################")


def mc_pi_estimate(stats=False, sample_size=10000):
    N_sampl = np.random.uniform(0, 1, sample_size)
    y = unit_circle(N_sampl)
    y_avg = np.mean(y)
    pi_esti = 4*y_avg
    if stats:
        print_stats("1.a)", N_sampl, pi_esti, np.pi)
    return pi_esti


def poly_estimate_uni(stats=False, sample_size=10000):
    N_sampl = np.random.uniform(0, 1, sample_size)
    y = poly_func(N_sampl, 3)
    y_avg = np.mean(y)
    if stats:
        print_stats("2.a)", N_sampl, y_avg, 1/4)
    return y_avg


def poly_estimate_power(stats=False, sample_size=10000):
    # power function ax^(a-1) -> (k-1)x^k mit k=2.5 -> a = 3.5
    N_sampl = np.random.power(3.5, sample_size)
    y = poly_func(N_sampl, 3)/q_distribution(3.5, N_sampl)
    y_avg = np.mean(y)
    if stats:
        print_stats("2.a)", N_sampl, y_avg, 1/4)
    return y_avg


def ex_1():
    mc_pi_estimate(True)
    print("Excercise 1.b)")
    res_all_pi = [mc_pi_estimate() for x in range(1000)]

    print(str(np.mean(res_all_pi))+" +- " + str(np.std(res_all_pi)))
    print(np.mean(res_all_pi)-np.pi)
    res = []
    for i in range(1, 7):
        res.append([mc_pi_estimate(sample_size=10**i) for x in range(100)])
    est_pi = np.mean(res, axis=1)
    est_std = np.std(res, axis=1)
    print(est_std)
    print(est_pi)
    x_axis = np.asarray([10.0**i for i in np.linspace(1.0, 6.0, 6)])
    y_axis = abs(est_pi - np.pi)

    xdata = np.log(x_axis)
    ydata = np.log(y_axis)
    popt, pcov = curve_fit(fitt_slope, xdata, ydata)
    plt.figure("LogLog plot")
    plt.title("Excercise 1 LogLog plot")

    plt.errorbar(x_axis, y_axis, yerr=est_std, capsize=3, linestyle="None", fmt="ob",
                 label="Monte Carlo estimaed pi values")
    y = np.exp(fitt_slope(xdata, *popt))
    plt.loglog(x_axis, y, label="Fitted function with slope -0.5")
    plt.legend()
    plt.xlabel("N_sample")
    plt.ylabel(r'Absolute error $\vert{\pi-\pi_{est}}\vert$')
    plt.savefig("excercise_1_a.png")


def ex_2():
    true_value = 1/4
    N_sample = [2, 5, 10, 50, 100, 200, 400, 500,
                700, 1000, 2000, 3000, 4000, 5000, 10000]

    poly_estimate_uni(stats=False, sample_size=10000)
    res_uni = []
    for i in N_sample:
        res_uni.append([poly_estimate_uni(sample_size=i) for x in range(100)])

    res_pow = []
    for i in N_sample:
        res_pow.append([poly_estimate_power(sample_size=i)
                       for x in range(100)])
    est_pow = np.mean(res_pow, axis=1)
    est_poly = np.mean(res_uni, axis=1)

    x_axis = N_sample
    y_axis_uni = abs(est_poly - 0.25)
    y_axis_pow = abs(est_pow - 0.25)

    plt.figure("Excercise 2 a and b comparison")
    plt.title("Excercise 2 a and b comparison")

    xdata = np.log(x_axis)
    ydata = np.log(y_axis_uni)
    popt, pcov = curve_fit(fitt_slope_poly, xdata, ydata)
    y_uni = np.exp(fitt_slope_poly(xdata, *popt))
    plt.loglog(x_axis, y_uni, color="red",
               label=f"Fitted slope {popt[0]:.2f} uniform")

    xdata = np.log(x_axis)
    ydata = np.log(y_axis_pow)
    popt, pcov = curve_fit(fitt_slope_poly, xdata, ydata)
    y_pow = np.exp(fitt_slope_poly(xdata, *popt))

    plt.loglog(x_axis, y_pow, color="blue",
               label=f"Fitted slope {popt[0]:.2f} Powerlaw")

    plt.loglog(x_axis, y_axis_uni, ".", color="red",
               label="Error for uniform drawn samples")
    plt.loglog(x_axis, y_axis_pow, ".", color="blue",
               label="Error for Powerlaw drawn samples")

    plt.xlabel("N_sample")
    plt.ylabel(
        r'Absolute error $\vert I_{est}-I\vert$')
    plt.legend()

    plt.savefig("excercise_2_a_b.png")
    print("2.b) It seems that powerlaw generally performs better,")
    print("Sometimes we get random results where the uniform ")
    print("one outperforms the powerlaw one")


def generate_random(size):
    res = []
    while len(res) <= size:
        x = np.random.uniform(-50, 50, 1)
        y = np.random.uniform(0, 0.15, 1)
        if y <= pdf_func(x, 3, 7):
            res.append(x)
    return np.asarray(res)


def ex_3():
    x = generate_random(10000)
    plt.figure("Excercise 3.a) Rejection method")
    plt.title("Excercise 3.a) Rejection method")
    plt.hist(x, bins=100, density=True,
             label="Normalized histogram of generated random variables x")

    x_axis = np.linspace(-50, 50, 10000)
    y = pdf_func(x_axis, 3, 7)

    plt.plot(x_axis, y, label="pdf plottet with m=3 abd b=7")
    plt.xlabel("x values")
    plt.ylabel("probability p(x)")
    plt.savefig("excercise_3_a.png")

    # Inverse cdf method
    u = np.random.uniform(0, 1, 10000)

    x_2 = inverse_CDF(u, 3, 7)
    x_2 = x_2[(x_2 >= -50) & (x_2 <= 50)]
    plt.figure("Excercise 3.b) Inverse cdf method")
    plt.title("Excercise 3.b) Inverse cdf method")
    plt.hist(x_2, bins=100, density=True,
             label="Normalized histogram of generated random variables x")
    plt.plot(x_axis, y, label="pdf plottet with m=3 abd b=7")
    plt.xlabel("x values")
    plt.ylabel("probability p(x)")
    plt.savefig("excercise_3_b.png")


def ex_4(data):
    print("Excercise 4.a)")
    mean = np.mean(data)
    std = np.std(data)
    print(f"Mean: {mean:.2f} and std: {std:.2f} ")


if __name__ == '__main__':
    # ex_1()
    # ex_2()
    # ex_3()
    data = np.loadtxt('gauss_data.txt')
    ex_4(data)
