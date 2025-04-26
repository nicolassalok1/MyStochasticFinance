from chapter7.jump_process import ParametricJumpProcess, ParametricJumpProcessSamplingDensity
import numpy as np
from chapter5.base_forecasting import (
    TimeUnitTransformer, ForecastingProcess
)
from typing import List
from scipy.stats import norm
from chapter6.diffusion_model import DiffusionProcessParameters, LoglikelihoodOptimizer, DefaultSHGOOptimizer
from chapter2.stock_price_dataset_adapters import StockPriceDatasetAdapter
from chapter7.density_recovery_methods import DensityRecoveryBasedAssetPriceModel, NelderMeadLLOptimizer, COSMethodBasedDensityRecovery, \
    RecoveredDistributionGenerator


class MertonProcessSamplingDensity(ParametricJumpProcessSamplingDensity):

    '''
    Sampliing density for generating RVs for Merton model
    '''

    def __init__(self, r, σ, λ, μ_j, σ_j):
        self._μ_j = μ_j
        self._σ_j = σ_j

        super().__init__(r=r, σ=σ, λ=λ)

    def _sample_jumps(self, n_sample_paths, n_jumps):
        return norm.rvs(size=(n_sample_paths, n_jumps),
                        loc=self._μ_j, scale=self._σ_j)


class MertonProcess(ParametricJumpProcess):
    '''
    Stochastic process for Merton model
    '''

    def __init__(self, r, σ, λ, μ_j, σ_j, initial_state=1.0, n_sample_paths=5):
        self._μ_j = μ_j
        self._σ_j = σ_j

        super().__init__(r=r, σ=σ, λ=λ, n_sample_paths=n_sample_paths, initial_state=initial_state,
                         sampling_density=MertonProcessSamplingDensity(r=r, σ=σ, λ=λ, μ_j=μ_j, σ_j=σ_j))

    def _compute_jump_drift(self):
        return self._r - (0.5*(self._σ**2)) - (self._λ *
                                               (np.exp(self._μ_j + 0.5*(self._σ_j**2)-1)))


class MertonProcessParameters(DiffusionProcessParameters):
    λ: float = None
    μ_j: float = None
    σ_j: float = None


merton_process_ll_optimizer = NelderMeadLLOptimizer(
    x0=[0.004, 0.007, 0.02, 1.004, 0.004],
    θ0_bounds=[(0.0001, 0.1), (0.001, 0.1), (0.00001, 0.9),
               (0.0001, 2), (0.0001, 2)])

merton_process_ll_optimizer_1 = DefaultSHGOOptimizer(
    θ0_bounds=[(0.0001, 5), (0.001, 5), (0.001, 5.0),
               (0.0001, 5), (0.001, 5.0)])


class MertonProcessAssetPriceModel(DensityRecoveryBasedAssetPriceModel):

    def __init__(self, time_unit_transformer: TimeUnitTransformer,
                 settings={'N_freq': 2048, 'n_workers': 10},
                 asset_price_dataset_adapter: StockPriceDatasetAdapter = None,
                 ll_optimizer: LoglikelihoodOptimizer = merton_process_ll_optimizer,
                 n_sample_paths=100,
                 training_required=True,
                 ):
        print('MertonProcessAssetPriceModel.training_required' +
              str(training_required))
        super().__init__(time_unit_transformer=time_unit_transformer,
                         asset_price_dataset_adapter=asset_price_dataset_adapter,
                         ll_optimizer=ll_optimizer,
                         n_sample_paths=n_sample_paths,
                         training_required=training_required,
                         settings=settings)

    def _create_empty_param_instance(self):
        return MertonProcessParameters()

    def _create_density_recovery_method(self, cf_φ_ω: callable, N_Freq):
        print('N_Freq - ' + str(N_Freq))
        return COSMethodBasedDensityRecovery(N_Freq, cf_φ_ω)

    def _tuple_to_param_order(self, θ):
        print(θ)
        self._parameters['r'], self._parameters['σ'], self._parameters[
            'λ'], self._parameters['μ_j'], self._parameters['σ_j'] = θ

    def _characteristic_function_ϕ_ω(self, ω, θ, t=None):
        r, σ, λ, μ_j, σ_j = θ
        '''
        print('μ_j--' + str(μ_j))
        print('σ_j--' + str(σ_j))
        print('μ_j+0.5*(σ_j**2))-1--' + str(μ_j+0.5*(σ_j**2)-1)) '''
        merton_μ = r - 0.5*(σ**2) - (λ*(np.exp(μ_j+0.5*(σ_j**2))-1))
        base_term = np.exp(ω * merton_μ * 1j * t - (0.5*(σ**2)*(ω**2)*t))
        merton_model_term = np.exp(
            λ * t * (np.exp((ω * μ_j * 1j) - (0.5*(σ_j**2)*(ω**2))) - 1))

        return np.exp(1j * ω * self._x0) * base_term * merton_model_term

    def _κ1(self, θ, t):
        r, σ, λ, μ_j, σ_j = θ
        merton_μ = r - 0.5*(σ**2) - (λ*(np.exp(μ_j+0.5*(σ_j**2))-1))
        return t * (merton_μ + (λ * μ_j))

    def _κ2(self, θ, t):
        _, σ, λ, μ_j, σ_j = θ
        return t * ((σ ** 2) + (λ * ((μ_j ** 2) + (σ_j ** 2))))

    def _κ4(self, θ, t):
        _, _, λ, μ_j, σ_j = θ
        return λ * t * ((μ_j ** 4) + (6 * (μ_j ** 2) * (σ_j ** 2)) + (3 * (σ_j ** 4) * λ))

    def _create_forecasting_process(self,
                                    parameters, n_sample_paths) -> ForecastingProcess:
        return MertonProcess(
            r=parameters['r'],
            σ=parameters['σ'],
            λ=parameters['λ'],
            μ_j=parameters['μ_j'],
            σ_j=parameters['σ_j'],
            initial_state=parameters['s0'],
            n_sample_paths=n_sample_paths)

    def _get_rv_generator_for_viz(self, s_t_1, t):
        θ = (self._parameters['r'], self._parameters['σ'],
             self._parameters['λ'], self._parameters['μ_j'],
             self._parameters['σ_j'])
        return RecoveredDistributionGenerator(density_recovery_method=COSMethodBasedDensityRecovery(
            N_freq=2000, ϕ_ω=self._characteristic_function_φ_ω),
            θ=θ,
            s_i=s_t_1,
            t=t,
            cumulants={'κ1': self._κ1(θ, t),
                       'κ2': self._κ2(θ, t),
                       'κ4': self._κ4(θ, t),
                       })
