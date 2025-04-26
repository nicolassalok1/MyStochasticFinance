import numpy as np
import chapter3.visualization as vs3
from scipy.stats import poisson
import pandas as pd

def poisson_distribution():
    # a set of different λ values
    lamdas = [1, 4, 10, 40]

    # generate a sample of input x (positive integers only) that work as random variates
    x = np.arange(0, 15)

    # Get probabilities of input x for different λ values
    prob_x = {
        lamda: (
            x,
            # pmf returns the probability of input x from poisson PMF
            poisson(lamda).pmf(x),
        )
        for lamda in lamdas
    }
    # plotting x vs probabilities for different λ values
    vs3.poisson_plot(prob_x)


from scipy.stats import norm


def gaussian_distribution():
    # a set of different (μ,σ) pairs
    params_mu_sigma = [(85, 90), (-6, 70), (43, 40), (-10, 19)]

    # generate a sample of input x (both positive & negative) that work as random variates
    x = np.linspace(start=-500, stop=500, num=100)

    # Get probabilities of input x for different (μ,σ) pairs
    prob_x = {
        "(%d,%d)"
        % (mu_sigma[0], mu_sigma[1]): (
            x,
            # pdf returns the probability of input x from normal PDF
            norm.pdf(x, loc=mu_sigma[0], scale=mu_sigma[1]),
        )
        for mu_sigma in params_mu_sigma
    }
    # plotting x vs probabilities for different (μ,σ) pairs
    vs3.gaussian_plot(prob_x, key="(μ,σ)")


from scipy.stats import expon


def exponential_distribution():
    # a set of different λ values (should be positive)
    lamdas = [0.5, 1.3, 0.9, 2]

    # generate a sample of input x (only positive) that work as random variates
    x = np.linspace(start=0, stop=5, num=100)

    # Get probabilities of input x for different λ values
    prob_x = {
        lamda: (
            x,
            # pdf returns the probability of input x from poisson PDF
            expon.pdf(x, loc=0, scale=1 / lamda),
        )
        for lamda in lamdas
    }
    # plotting x vs probabilities for different λ values
    vs3.exponential_plot(prob_x)


from scipy.stats import uniform


def uniform_distribution():
    # generate a sample of input x  that work as random variates
    x = np.linspace(start=-500, stop=500, num=100)

    # Get probabilities of input x for a given range a & b (denoted by loc & scale)
    probs = uniform.pdf(x, loc=100, scale=200)

    # plotting x vs probabilities
    vs3.uniform_plot(x, probs)

import numpy as np
import chapter3.visualization as vs3
import pandas as pd

def characteristic_funcs():
    # Generate freuencies. Ideally frequency can lie between -∝ to +∝. 
    # We generate samples from -100 to 100.
    ω_arr = np.linspace(start=-100, stop=100, num=100)

    def _ϕ_gaussian(ω, μ, σ2): # Characteristic function for Gaussian density
        return np.exp((μ * ω * 1j) - (0.5 * ω * ω * σ2))

    def _ϕ_uniform(ω, b, a): # Characteristic function for Uniform density
        return (np.exp(1j * ω * b)-np.exp(1j * ω * a))/(1j * ω * (b - a))

    def _ϕ_exponential(ω, λ): # Characteristic function for Exponential density
        return 1.0/(1.0 - ((ω * 1j)/λ))

    def _ϕ_poisson(ω, λ): # Characteristic function for Poisson density
        return np.exp(λ * (np.exp(1j * ω)-1))

    def _generate_ϕ_ω_values(ϕ_ω, **θ):
        """
        Generate ϕ(ω) values for all given frequencies and use the  
        real part only for plotting
        """
        return pd.DataFrame([{'ω':ω, 'ϕ(ω)':ϕ_ω(ω, **θ).real} for ω in ω_arr])

    # Generate ϕ(ω) values for four different densities with fixed parameters
    μ = 200
    σ2 = 10
    b = 990
    a = 90
    λ = 10.0
    poisson_λ = 100

    cf_gaussian = ('Gaussian ϕ(ω; μ, σ2)', _generate_ϕ_ω_values(ϕ_ω=_ϕ_gaussian, μ=μ, σ2=σ2))
    cf_uniform = ('Uniform ϕ(ω; b, a)', _generate_ϕ_ω_values(ϕ_ω=_ϕ_uniform, b=b, a=a))
    cf_exponential = ('Exponential ϕ(ω; λ)', _generate_ϕ_ω_values(ϕ_ω=_ϕ_exponential, λ=λ))
    cf_poisson = ('Poisson ϕ(ω; λ)', _generate_ϕ_ω_values(ϕ_ω=_ϕ_poisson, λ=poisson_λ))
    
    vs3.plot_cf([cf_gaussian, cf_uniform, cf_exponential, cf_poisson])

    
def characteristic_func_uniform_and_exponential():
    
    ω_arr = np.linspace(start=-100, stop=100, num=100)
    params_b_a = [(85, 90), (-6, -1), (43, 40), (-10, 19)]
    λ_arr = [0.5, 1.3, 8, 15]

    def _ϕ_exponential(ω, λ): # Characteristic function for Exponential density
        return 1.0/(1.0 - ((ω * 1j)/λ))

    def _ϕ_uniform(ω, b, a): # Characteristic function for Uniform density
        return (np.exp(1j * ω * b)-np.exp(1j * ω * a))/(1j * ω * (b - a))
    
    ϕ_ω_values = {'Exponential': pd.DataFrame([{'ω':ω, 'ϕ(ω)':_ϕ_exponential(ω, λ).real, 'λ': λ} for λ in λ_arr for ω in ω_arr]), 
                  'Uniform': pd.DataFrame([{'ω':ω, 'ϕ(ω)':_ϕ_uniform(ω, b, a).real, '(b,a)': (b, a)} for b, a in params_b_a for ω in ω_arr])}
    vs3.plot_uniform_and_exponential_cf(ϕ_ω_values)

