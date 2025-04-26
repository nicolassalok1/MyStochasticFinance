import numpy as np
from chapter5.base_forecasting import (
    ForecastingProcess, TargetSamplingDensity
)
from scipy.stats import norm, poisson

from abc import abstractmethod, ABC


class ParametricJumpProcessSamplingDensity(TargetSamplingDensity, ABC):

    '''
    Base class for Sampliing density of jump process
    '''

    def __init__(self, r, σ, λ):
        self._r = r
        self._σ = σ
        self._λ = λ

    def pdf(self, x):
        return None

    def sample(self, n_vars, n_sample_paths=1):
        dWs = norm.rvs(size=(n_sample_paths, n_vars))
        dJ_p = poisson.rvs(mu=self._λ, size=(n_sample_paths, n_vars))
        dJ = self._sample_jumps(n_sample_paths, n_vars)

        return np.array([[{'dW': w, 'dJ_p': j_p, 'dJ': j
                           } for w, j, j_p in zip(dw, dj, dj_p)] for dw, dj, dj_p in zip(dWs, dJ, dJ_p)])

    @abstractmethod
    def _sample_jumps(self, n_sample_paths, n_jumps): ...
    '''
        Function to sample jumps from the specific jump size distributions
    '''


class ParametricJumpProcess(ForecastingProcess, ABC):

    '''
    Base class for jump stochastic process. Function _compute_jump_drift
    should be overridden by the child class (Merton or Kou models)
    '''

    def __init__(self, r, σ, λ, sampling_density: TargetSamplingDensity = None, initial_state=1.0, n_sample_paths=5):
        self._r = r
        self._σ = σ
        self._λ = λ
        self._state_t = initial_state
        self._μ = self._compute_jump_drift()

        super().__init__(n_sample_paths, initial_state=initial_state,
                         sampling_density=sampling_density)

    @abstractmethod
    def _compute_jump_drift(self): ...

    def _update_current_state(self, z):
        self._reset_new_sample_path_state()

        dW = z['dW']
        dJ = z['dJ']
        dJ_p = z['dJ_p']

        self._state_t = self._state_t * \
            np.exp(self._μ + (self._σ * dW) + (dJ*dJ_p))
        return self._state_t
