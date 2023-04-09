import numpy as np
import matplotlib.pyplot as plt
import timeit
from typing import Callable, Union

from scipy.optimize import minimize
import numdifftools as ndt

# Used constants
AVG_DECAY_LENGTH_PI_PLUS = 4188.0
SPEED_OF_LIGHT = 299792458
GENERATE_PLOTS = True

def mixed_exp_pdf(y2_inverse: float, params) -> Union[float, np.ndarray]:
    """
    Calculate the probability density function (PDF) for a mixture of two exponential distributions.
    
    Parameters
    ----------
    y2_inverse : float
        The inverse of the second exponential distribution parameter (1 / lambda_2).
    
    params : tuple
        A tuple containing the inverse of the first exponential distribution parameter (1 / lambda_1) and the data points.
    
    Returns
    -------
    Union[float, np.ndarray]
        The PDF values for the given data points based on the mixture of two exponential distributions.
    """
    y1_inverse, data = params
    y1 = 1/y1_inverse
    y2 = 1/y2_inverse
    p1 = y1 * np.exp(-y1 * data)
    p2 = y2 * np.exp(-y2 * data)
    return 0.84 * p1 + 0.16 * p2


def neg_log_likelihood_mixed_exp(y2_inverse: float, 
                                         params: tuple) -> float:
    """
    Calculate the negative log-likelihood for a Gaussian distribution using the mixed exponential PDF.
    
    Parameters
    ----------
    y2_inverse : float
        The inverse of the second exponential distribution parameter (1 / lambda_2) for the mixed exponential PDF.
    
    params : tuple
        A tuple containing the inverse of the first exponential distribution parameter (1 / lambda_1) and the data points.
    
    Returns
    -------
    float
        The negative log-likelihood value for the Gaussian distribution.
    """
    return -np.sum(np.log(mixed_exp_pdf(y2_inverse, params)))


def plot_fit_mixed_exp(pdf, data, y2, error, y1=AVG_DECAY_LENGTH_PI_PLUS):
    x_axis = data[::100]
    plt.figure("Histogram mixed exponential pdf", figsize=(6.4, 4))
    plt.plot(x_axis, pdf(y2, (y1, x_axis)),"--", color="red",
             linewidth=0.5,
             label=(r"$0.84\cdot\lambda_{\pi^{+}} e^{-\lambda_{\pi^{+}} \cdot x}"+
             r"+0.16\cdot\lambda_{K^{+}} e^{-\lambda_{K^{+}} \cdot x}$"))
    plt.fill_between(x_axis, pdf(y2-error, (y1, x_axis)),
                     pdf(y2+error, (y1, x_axis)),
                     color='red',
                     alpha=0.3,
                     label=r"Error band $\pm$"+str(round(error, 4)))
    plt.hist(data, bins=200, density=True,
             label="Normalized histrogram of measurement")
    plt.xlabel(r'Decay length $[m]$')
    plt.ylabel(r'Probability density $p(x)$')
    plt.legend()
    plt.savefig("img/Histogram_fit_mixed_exp.png",dpi=1200)
    plt.savefig("img/Histogram_fit_mixed_exp.pgf")

    
    
    
    
    
    
if __name__ == "__main__":
    data = np.loadtxt("dec_lengths.txt")
    data = np.sort(data)
    params_mixed_exp = [AVG_DECAY_LENGTH_PI_PLUS, data]
    bounds_mixed_exp_fit = (0, None)
    res_neg_log = minimize(neg_log_likelihood_mixed_exp,
                           method='Powell',
                           x0=(0.1),
                           bounds=(bounds_mixed_exp_fit,),
                           args=params_mixed_exp)
    fit_avg_decay_length_k_plus = res_neg_log.x[0]
    hessian_nll_mixed_exp = ndt.Hessian(neg_log_likelihood_mixed_exp)
    hessian_matrix = hessian_nll_mixed_exp(fit_avg_decay_length_k_plus,
                                           params_mixed_exp)
    inv_hessian_matrix = np.linalg.inv(hessian_matrix)
    uncertainties_exp_fit = np.sqrt(np.diag(inv_hessian_matrix))[0]
    
    
    print(fit_avg_decay_length_k_plus, "+-", uncertainties_exp_fit)
    if GENERATE_PLOTS:
        plot_fit_mixed_exp(mixed_exp_pdf, data, fit_avg_decay_length_k_plus,
                       uncertainties_exp_fit, AVG_DECAY_LENGTH_PI_PLUS)

    