from scipy.stats import uniform
from numpy import exp


def inverse_transform_method_rvs(F_inverse, n_rv=1000):
    """
    Function to genrate random variables from any inverseky tranformed density

    F_invesre:  Inverse function of probability distribution function.
    """
    # Generate random variables from Uniform distribution.
    probs = uniform(0, 1).rvs(size=n_rv)

    return [F_inverse(p) for p in probs]


def generate_poisson_rv(λ, n_rv):
    # inspired by [9]Devroy
    u = uniform.rvs(0, 1, n_rv)

    def _get_single_x(u_i):
        x = 0
        p = exp(-λ)
        s = p
        while u_i > s:
            x = x + 1
            p = (p * λ)/x
            s = s + p
        return x

    return [_get_single_x(u_i) for u_i in u]
