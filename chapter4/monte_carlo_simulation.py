
from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
from numpy import average, sqrt, sum, power, vectorize
from scipy.optimize import minimize, shgo


def _compute_standard_error(H_x, mean, axis, n):
    H_x_2 = power(H_x, 2)
    return (sum(
        H_x_2, axis=axis) - (n * (mean ** 2))) / (n*(n-1))


class TargetSamplingDensity(ABC):

    '''
    Abstract template for implementing probabilty density to sample from.
    Monte carlo methods leverage this density to simulate random behaviour.
    '''

    @abstractmethod
    def pdf(self, x): ...

    @abstractmethod
    def sample(self, n_vars, n_sample_paths=1): ...


class VarReduction(ABC):
    '''
    Template class for various reduction schemes. Underlying scheme should
    extend this class override sample_H_x function.
    '''

    def __init__(self):
        self._axis = 0
        self._n = 0
        self._h_x_fun: callable = None
        self._n_vars = 0
        self._n_sample_paths = 1

    @abstractmethod
    def sample_H_x(self, x=None): ...
    '''
    Function to provide samples to measure H(x) from the 
    provided samples of x 
    '''


class NoVarReduction(VarReduction):
    '''
    Class for vanila sampling with no variance reduction schemes.
    It is used when there is no need to adjust H(x). 
    '''

    def sample_H_x(self, x=None):
        H_x = vectorize(lambda y: self._h_x_fun(y), otypes=[float])
        return H_x(x)


class AntitheticSampling(VarReduction):
    '''
    Class for Antithetic sampling as variance reduction scheme.
    It uses average of negatively correlated variables to adjust H(x)
    '''

    def sample_H_x(self, x=None):
        print('Antithetic Sampling')
        H_x = vectorize(lambda y: (self._h_x_fun(
            y) + self._h_x_fun(-y))/2, otypes=[float])
        return H_x(x)


class ImportanceSampling(VarReduction, ABC):

    '''
    Class for Importance sampling as variance reduction scheme.
    It samples from a alternative proposal distributionn g(x,θ) 
    and adjusts H(x) with the likelihood ratio defined as f(x)/g(x,θ)
    '''

    def __init__(self):
        self._target_sampling_density = None
        super().__init__()

    @abstractmethod
    def _proposal_g_x(self, x, θ: tuple): ...

    @abstractmethod
    def _sample_from_proposal_density_g(self, θ: tuple): ...

    def sample_H_x(self, x=None):
        print('Importance Sampling')

        def _likelihood_ratio(x, θ):
            return self._target_sampling_density.pdf(x)/self._proposal_g_x(x, θ)

        def _H_x_with_θ(θ):
            x = self._sample_from_proposal_density_g(θ)
            H_x = vectorize(lambda y: self._h_x_fun(y)
                            * _likelihood_ratio(y, θ), otypes=[float])
            return H_x(x)

        def _compute_total_variance(θ: tuple):
            H_x = _H_x_with_θ(θ)
            mean = average(H_x, axis=self._axis)

            # Sum of variances of all sample paths
            return sum(_compute_standard_error(H_x=H_x, mean=mean, axis=self._axis, n=self._n))

        # Finds the optimal θ in g(x,θ) for which the total variance of H(x) would
        # be minimum
        # optimal_θ = minimize(_compute_total_variance, bounds=[
        #                     (0.001, None), ((0.001, None))], x0=[0.01, 0.01]).x
        optimal_θ = shgo(_compute_total_variance, bounds=[
            (-2, 2), (0.1, 2)]).x

        print('Optimal θ' + str(optimal_θ))

        return _H_x_with_θ(optimal_θ)


class MonteCarloSimulation:
    '''
    Class to perform Monte Carlo simulation over a function h_x_fun with a given 
    sampling density for x and a optional variance reduction scheme
    '''
    @dataclass
    class MCEstimate:
        '''
         Dataclass returned as ouput of the simulation
        '''
        samples: List = None  # values of H(x)
        mean: float = 0.0
        standard_error: float = 0.0

    def __init__(self, h_x_fun: callable,
                 target_sampling_density: TargetSamplingDensity,
                 n_vars,
                 n_sample_paths=1,
                 var_reduction: VarReduction = NoVarReduction()):

        if not callable(h_x_fun):
            raise TypeError(
                "h_x_fun should be callable: function or class with __call__()")
        self._h_x_fun = h_x_fun
        self._n_vars = n_vars
        self._n_sample_paths = n_sample_paths
        self._axis, self._n = (0, self._n_sample_paths) if self._n_sample_paths > 1 else (
            1, self._n_vars)

        self._target_sampling_desnity = target_sampling_density

        self._var_reduction = var_reduction
        if hasattr(self._var_reduction, '_target_sampling_density'):
            setattr(self._var_reduction, '_target_sampling_density',
                    target_sampling_density)
        self._var_reduction._target_sampling_density = target_sampling_density
        self._var_reduction._n_sample_paths = n_sample_paths
        self._var_reduction._n_vars = n_vars
        self._var_reduction._h_x_fun = h_x_fun
        self._var_reduction._axis = self._axis
        self._var_reduction._n = self._n

    def new_estimate(self) -> MCEstimate:
        '''
        Function to return a new estimated result of H(x). Calling it 
        each time may return different result because of the 
        random simulation.
        '''
        x = self._target_sampling_desnity.sample(
            n_vars=self._n_vars, n_sample_paths=self._n_sample_paths)
        if not (len(x) == self._n_sample_paths and len(x[0]) == self._n_vars):
            raise ValueError('Random variable should be in shape (' +
                             str(self._n_sample_paths) + ',' + str(self._n_vars) + ')')
        estimate = MonteCarloSimulation.MCEstimate()
        estimate.samples = self._var_reduction.sample_H_x(x)
        estimate.mean = average(estimate.samples, axis=self._axis)
        estimate.standard_error = sqrt(_compute_standard_error(H_x=estimate.samples,
                                                               mean=estimate.mean, axis=self._axis, n=self._n))
        return estimate
