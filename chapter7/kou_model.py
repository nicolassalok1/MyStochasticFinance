from chapter7.jump_process import ParametricJumpProcess, ParametricJumpProcessSamplingDensity
import numpy as np
from chapter5.base_forecasting import (
    TimeUnitTransformer, ForecastingProcess
)
from scipy.stats import laplace

from chapter6.diffusion_model import DiffusionProcessParameters, LoglikelihoodOptimizer
from chapter2.stock_price_dataset_adapters import StockPriceDatasetAdapter
from chapter7.density_recovery_methods import DensityRecoveryBasedAssetPriceModel, L_BFGS_BLLOptimizer, COSMethodBasedDensityRecovery, \
    RecoveredDistributionGenerator

from chapter4.random_number_gen_accept_reject import HatFunctionEstimator, DefaultSupremumEstimator

from chapter4.random_number_gen_accept_reject import AcceptanceRejectionMethod


class AsymmetricDoubleExponentialGenerator(AcceptanceRejectionMethod):

    def __init__(self, p, α_1, α_2, hat_func_optimizer: HatFunctionEstimator):
        self._p = p
        self._α_1 = α_1
        self._α_2 = α_2
        super().__init__(hat_func_optimizer)

    def _proposal_pdf_g(self, x, θ: tuple):
        return laplace.pdf(x, loc=θ[0], scale=1/θ[1])

    def _sample_from_proposal_g_with_θ_optimal(self, n_rv):
        θ = self._hat_func_estimator.θ_optimal_for_g
        return laplace.rvs(loc=θ[0], scale=1/θ[1], size=n_rv)

    def target_pdf_f(self, x):

        return np.where(np.array(x) >= 0,
                        np.dot(self._p*self._α_1,
                               np.exp(-np.dot(self._α_1, x))),
                        np.dot((1-self._p)*self._α_2,
                               np.exp(np.dot(self._α_2, x))))


class KouProcessSamplingDensity(ParametricJumpProcessSamplingDensity):

    '''
    Sampliing density for generating RVs for Kou model
    '''

    def __init__(self, r, σ, λ, p, α_1, α_2):
        self._p = p
        self._α_1 = α_1
        self._α_2 = α_2

        super().__init__(r=r, σ=σ, λ=λ)

    def _sample_jumps(self, n_sample_paths, n_jumps):
        jumps, _ = AsymmetricDoubleExponentialGenerator(p=self._p, α_1=self._α_1, α_2=self._α_2,
                                                        hat_func_optimizer=DefaultSupremumEstimator(
                                                            x0=[
                                                                0.00001],
                                                            x0_bounds=[
                                                                (0.00001, None)],
                                                            θ0=[0.00001,
                                                                0.00001],
                                                            θ0_bounds=[(0.00001, None), (0.00001, None)])).sample(n_jumps*n_sample_paths)
        return np.array(jumps).reshape(n_sample_paths, n_jumps)


class KouProcess(ParametricJumpProcess):
    '''
    Stochastic process for Kou model
    '''

    def __init__(self, r, σ, λ, p, α_1, α_2, initial_state=1.0, n_sample_paths=5):
        self._p = p
        self._α_1 = α_1
        self._α_2 = α_2

        super().__init__(r=r, σ=σ, λ=λ, n_sample_paths=n_sample_paths, initial_state=initial_state,
                         sampling_density=KouProcessSamplingDensity(r=r, σ=σ, λ=λ, p=p, α_1=α_1, α_2=α_2))

    def _compute_jump_drift(self):
        de_term_1 = self._p*self._α_1*(1/(self._α_1-1))
        de_term_2 = (1-self._p)*self._α_2*(1/(self._α_2+1))
        return self._r - (0.5*(self._σ**2)) + (self._λ *
                                               (1-de_term_1-de_term_2))


class KouProcessParameters(DiffusionProcessParameters):
    λ: float = None
    p: float = None
    α_1: float = None
    α_2: float = None

    def __getstate__(self):
        """Used for serializing instances"""

        # start with a copy so we don't accidentally modify the object state
        # or cause other conflicts
        return self.__dict__.copy()

    def __setstate__(self, state):
        """Used for deserializing"""
        # restore the state which was picklable
        self.__dict__.update(state)
        print('Helo.........')

        # restore unpicklable entries


kou_process_ll_optimizer = L_BFGS_BLLOptimizer(
    x0=[0.004, 0.007, 0.02, 0.003, 1.004, 0.004],
    θ0_bounds=[(None, None), (0.00001, None), (0.00001, None),
               (0.00001, 1.0), (1.00001, None), (0.00001, None)])


class KouProcessAssetPriceModel(DensityRecoveryBasedAssetPriceModel):

    def __init__(self, time_unit_transformer: TimeUnitTransformer,
                 settings={'N_freq': 200, 'n_workers': 10},
                 asset_price_dataset_adapter: StockPriceDatasetAdapter = None,
                 ll_optimizer: LoglikelihoodOptimizer = kou_process_ll_optimizer,
                 n_sample_paths=100,
                 training_required=True,
                 ):
        print('KouProcessAssetPriceModel.training_required' +
              str(training_required))
        super().__init__(time_unit_transformer=time_unit_transformer,
                         asset_price_dataset_adapter=asset_price_dataset_adapter,
                         ll_optimizer=ll_optimizer,
                         n_sample_paths=n_sample_paths,
                         training_required=training_required,
                         settings=settings)

    def _create_empty_param_instance(self):
        return KouProcessParameters()

    def _create_density_recovery_method(self, cf_φ_ω: callable, N_Freq):
        return COSMethodBasedDensityRecovery(N_Freq, cf_φ_ω)

    def _tuple_to_param_order(self, θ):
        print(θ)
        self._parameters['r'], self._parameters['σ'], self._parameters[
            'λ'], self._parameters['p'], self._parameters['α_1'], self._parameters['α_2'] = θ

    def _characteristic_function_ϕ_ω(self, ω, θ, t=None):
        r, σ, λ, p, α_1, α_2 = θ
        de_term_1 = p*α_1*(1/(α_1-1))
        de_term_2 = (1-p)*α_2*(1/(α_2+1))

        kou_μ = r - 0.5*(σ**2) + (λ * (1-de_term_1-de_term_2))
        base_term = np.exp(ω * kou_μ * 1j * t - (0.5*(σ**2)*(ω**2)*t))

        de_term_1_j = (p*α_1)/(α_1-1j*ω)
        de_term_2_j = ((1-p)*α_2)/(α_2+1j*ω)

        kou_model_term = np.exp(
            λ * t * (de_term_1_j + de_term_2_j - 1))

        return np.exp(1j * ω * self._x0) * base_term * kou_model_term

    def _κ1(self, θ, t):
        r, σ, λ, p, α_1, α_2 = θ
        de_term_1 = p*α_1*(1/(α_1-1))
        de_term_2 = (1-p)*α_2*(1/(α_2+1))

        kou_μ = r - 0.5*(σ**2) + (λ * (1-de_term_1-de_term_2))
        return t * (kou_μ + (λ * (p/α_1 - (1-p)/α_2)))

    def _κ2(self, θ, t):
        _, σ, λ, p, α_1, α_2 = θ
        return t * ((σ ** 2) + (2 * λ * (p/(α_1**2)+(1-p)/(α_2**2))))

    def _κ4(self, θ, t):
        _, _, λ, p, α_1, α_2 = θ
        return 24 * λ * t * (p/(α_1**4)+(1-p)/(α_2**4))

    def _create_forecasting_process(self,
                                    parameters, n_sample_paths) -> ForecastingProcess:
        return KouProcess(
            r=parameters['r'],
            σ=parameters['σ'],
            λ=parameters['λ'],
            p=parameters['p'],
            α_1=parameters['α_1'],
            α_2=parameters['α_2'],
            initial_state=parameters['s0'],
            n_sample_paths=n_sample_paths)

    def _get_rv_generator_for_viz(self, s_t_1, t):
        θ = (self._parameters['r'], self._parameters['σ'],
             self._parameters['λ'], self._parameters['p'],
             self._parameters['α_1'], self._parameters['α_2'])
        return RecoveredDistributionGenerator(density_recovery_method=COSMethodBasedDensityRecovery(
            N_freq=2000, ϕ_ω=self._characteristic_function_φ_ω),
            θ=θ,
            s_i=s_t_1,
            t=t,
            cumulants={'κ1': self._κ1(θ, t),
                       'κ2': self._κ2(θ, t),
                       'κ4': self._κ4(θ, t),
                       })
