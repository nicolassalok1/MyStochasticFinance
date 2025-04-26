from scipy.stats import norm
from chapter4.monte_carlo_simulation import TargetSamplingDensity, ImportanceSampling


class StandardNormalTargetSamplingDensity(TargetSamplingDensity):

    '''
    StanddardNormalTargetSamplingDensity
    '''

    def pdf(self, x):
        return norm.pdf(x)

    def sample(self, n_vars, n_sample_paths=1):
        return norm().rvs(size=(n_sample_paths, n_vars))


class GaussianImportanceSampling(ImportanceSampling):

    '''
    GaussianImportanceSampling
    '''

    def _proposal_g_x(self, x, θ: tuple):
        return norm.pdf(x, loc=θ[0], scale=θ[1])

    def _sample_from_proposal_density_g(self, θ: tuple):
        return norm.rvs(size=(self._n_sample_paths, self._n_vars), loc=θ[0], scale=θ[1])
