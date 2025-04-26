from scipy.stats import expon, norm, beta, uniform, cosine
import numpy as np
from chapter4.random_number_gen_accept_reject import AcceptanceRejectionMethod, HatFunctionEstimator


class GaussianVariateGeneratorWithExponential(AcceptanceRejectionMethod):

    """
    Class to sample from Gaussian density using a Exponential proposal desnity
    """

    def __init__(self, μ, σ, hat_func_optimizer: HatFunctionEstimator):
        self._μ = μ
        self._σ = σ
        super().__init__(hat_func_optimizer)

    def target_pdf_f(self, x):
        return norm.pdf(x, loc=self._μ, scale=self._σ)

    def _proposal_pdf_g(self, x, θ: tuple):
        return expon.pdf(x, scale=1 / θ[0])

    def _sample_from_proposal_g_with_θ_optimal(self, n_rv):
        return expon.rvs(scale=1 / self._hat_func_estimator.θ_optimal_for_g[0], size=n_rv)


class BetaVariateGeneratorWithGaussian(AcceptanceRejectionMethod):
    """
    Class to sample from Beta density using a Gaussian proposal desnity
    """

    def __init__(self, a, b, hat_func_optimizer: HatFunctionEstimator):
        self._a = a
        self._b = b
        super().__init__(hat_func_optimizer)

    def target_pdf_f(self, x):
        return beta(self._a, self._b).pdf(x)

    def _proposal_pdf_g(self, x, θ: tuple):
        return norm.pdf(x, loc=θ[0], scale=θ[1])

    def _sample_from_proposal_g_with_θ_optimal(self, n_rv):
        return norm.rvs(loc=self._hat_func_estimator.θ_optimal_for_g[0],
                        scale=self._hat_func_estimator.θ_optimal_for_g[1],
                        size=n_rv)


class CosineVariateGeneratorWithUniform(AcceptanceRejectionMethod):

    """
    Class to sample from Gaussian density using a Exponential proposal desnity
    """

    def __init__(self, hat_func_optimizer: HatFunctionEstimator):
        super().__init__(hat_func_optimizer)

    def target_pdf_f(self, x):
        return cosine.pdf(x)

    def _proposal_pdf_g(self, x, θ: tuple):
        return uniform.pdf(x, loc=θ[0], scale=θ[1])

    def _sample_from_proposal_g_with_θ_optimal(self, n_rv):
        return uniform.rvs(loc=self._hat_func_estimator.θ_optimal_for_g[0],
                           scale=self._hat_func_estimator.θ_optimal_for_g[1],
                           size=n_rv)
