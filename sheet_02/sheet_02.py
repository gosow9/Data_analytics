import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt 


def pdf(x, f, mu_s, sigma_s):
    g = 1/(sigma_s * np.sqrt(2*np.pi)) * np.exp(-(x - mu_s)**2 / (2 * sigma_s**2))
    u = 1/10
    p = f*g +(1-f)*u
    return p
    
def likelihood(data, mu_s, sigma_s, f):
    # Evaluate the Gaussian and uniform PDFs at each data point
    g = 1/(sigma_s * np.sqrt(2*np.pi)) * np.exp(-(data - mu_s)**2 / (2 * sigma_s**2))
    u = 1/10 # Since U(x) is a uniform distribution in [0, 10], its PDF is just 1/10
    
    # Calculate the total PDF for each data point
    p = f * g + (1 - f) * u
    
    # Calculate the likelihood by taking the product of the PDFs
    L = np.prod(p)
    
    return L
    
def neg_log_likelihood(params, data):
    mu_s, sigma_s, f = params
    
    # Make sure f is bounded between 0 and 1
    if f < 0 or f > 1:
        return np.inf
    
    # Evaluate the Gaussian and uniform PDFs at each data point
    g = 1/(sigma_s * np.sqrt(2*np.pi)) * np.exp(-(data - mu_s)**2 / (2 * sigma_s**2))
    u = 1/10 # Since U(x) is a uniform distribution in [0, 10], its PDF is just 1/10
    
    # Calculate the total PDF for each data point
    p = f * g + (1 - f) * u
    
    # Calculate the negative log-likelihood
    neg_ll = -np.sum(np.log(p))
    
    return neg_ll
    
    

def ex_2(data):
    print(data)
    plt.hist(data, bins=20)
    plt.show()
    print(pdf(x=4, f=0.5, mu_s=5, sigma_s=0.75))
    print(likelihood(data, mu_s=5, sigma_s=0.75, f=0.5))
    
    # Set the initial guess for the parameter values
    params0 = [5, 0.75, 0.5]

# Use the Nelder-Mead algorithm to minimize the negative log-likelihood
    result = minimize(neg_log_likelihood, params0, args=(data,), method='Nelder-Mead')

# Print the best fit parameter values and the corresponding negative log-likelihood
    print('Best fit parameter values: mu_s = {:.3f}, sigma_s = {:.3f}, f = {:.3f}'.format(*result.x))
    print('Negative log-likelihood at best fit: {:.3f}'.format(result.fun))


if __name__ == "__main__":
    data = np.loadtxt("signal_and_background.txt")
    ex_2(data)
