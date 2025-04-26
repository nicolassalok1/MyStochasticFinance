import numpy as np
from abc import abstractmethod, ABC
from chapter5.base_forecasting import TimeUnitTransformer
from scipy.optimize import minimize, basinhopping
from typing import List
from scipy.fft import fft, ifft

from chapter6.diffusion_model import BaseAssetPriceModel, LoglikelihoodOptimizer
from chapter2.stock_price_dataset_adapters import StockPriceDatasetAdapter
from loky import get_reusable_executor
from typing import List, TypedDict
from math import isclose
from chapter4.monte_carlo_simulation import VarReduction, NoVarReduction

π = np.pi


class Cumulants(TypedDict):
    κ1: float = None
    κ2: float = None
    κ4: float = None


class DensityRecoveryMethod(ABC):

    '''
     Base class for recovering density from charateristic function
    '''

    def __init__(self, N_freq, ϕ_ω: callable):
        self._N_freq = N_freq
        self._ϕ_ω = ϕ_ω

    def _get_integration_range(self, x_i, cumulants: Cumulants):
        intg_range = 8.0
        if (cumulants['κ2'] > 0 and cumulants['κ4'] > 0):
            intg_range = 8 * \
                np.sqrt(cumulants['κ2'] + np.sqrt(cumulants['κ4']))
            if isclose(intg_range, 0.00):
                intg_range = 8.0

        a = x_i + cumulants['κ1'] - intg_range
        b = x_i + cumulants['κ1'] + intg_range
        return b, a

    @abstractmethod
    def recover(self, x_i, t, cumulants: Cumulants, θ): ...
    '''
        Method returns density commputed at point x_i
    '''


class FFTBasedDensityRecovery(DensityRecoveryMethod):

    '''
      Class to recover density by FFT. This class should be used with
      caution as it involves hefty amount of computation.

    '''

    def __init__(self, N_freq, ϕ_ω: callable):
        self._k = np.array([i for i in range(N_freq)])
        self._j = np.array([i for i in range(N_freq)])
        super().__init__(N_freq, ϕ_ω)

    def _complex_terms(self, factors):
        return [np.ones(self._N_freq)*(-1) ** float(factor)
                for factor in factors]

    def recover(self, x_i, t, cumulants: Cumulants, θ):
        b, a = self._get_integration_range(x_i, cumulants)
        d = b - a
        u = (self._j - self._N_freq/2)/d
        ω = 2*π*u
        factors_1 = ((a/d) + (self._k/self._N_freq)) * self._N_freq
        C_k = np.divide(self._complex_terms(factors_1), d)

        factors_2 = -(2*a/d)*self._j
        ϕ_j = np.dot(self._complex_terms(factors_2), self._ϕ_ω(ω, θ, t))

        return np.abs(np.sum(np.dot(C_k, fft(ϕ_j))).real)


class COSMethodBasedDensityRecovery(DensityRecoveryMethod):

    '''
      Class to recover density by Cosine method as suggestd by Oosterlee

    '''

    def __init__(self, N_freq, ϕ_ω: callable):
        self._k = np.array([i for i in range(N_freq)])
        super().__init__(N_freq, ϕ_ω)

    def recover(self, x_i, t, cumulants: Cumulants, θ):
        b, a = self._get_integration_range(x_i, cumulants)
        d = b - a
        u = (self._k*π)/d
        f_k = (2.0/d) * (self._ϕ_ω(u, θ, t) * np.exp(-1j * a * u)).real
        f_k[0] = f_k[0] * 0.5
        return np.abs(np.sum(np.dot(f_k, np.cos(self._k*π*(x_i-a)/d))))


class L_BFGS_BLLOptimizer(LoglikelihoodOptimizer):

    def __init__(self, x0: List, θ0_bounds: List[tuple]):
        self._θ0_bounds = θ0_bounds
        self._x0 = x0

    def optimize(self, log_likelihood_func: callable):
        return minimize(fun=log_likelihood_func, x0=self._x0,
                        bounds=self._θ0_bounds, method='L-BFGS-B').x


class NelderMeadLLOptimizer(LoglikelihoodOptimizer):

    def __init__(self, x0: List, θ0_bounds: List[tuple]):
        self._θ0_bounds = θ0_bounds
        self._x0 = x0

    def optimize(self, log_likelihood_func: callable):
        return minimize(fun=log_likelihood_func, x0=self._x0,
                        bounds=self._θ0_bounds, method='Nelder-Mead').x


class DensityRecoveryBasedAssetPriceModel(BaseAssetPriceModel, ABC):

    '''
       Base class for any asset price model that used density recovery methods to
       estimate probability of any price.
    '''

    def __init__(self, time_unit_transformer: TimeUnitTransformer,
                 asset_price_dataset_adapter: StockPriceDatasetAdapter = None,
                 ll_optimizer: LoglikelihoodOptimizer = None,
                 n_sample_paths=100,
                 training_required=True,
                 settings={'N_freq': 500, 'n_workers': 10}
                 ):
        self._s_t_filtered = None
        self._settings = settings
        self._s_t_parts = None
        self._x0 = None

        self._recovery_method = self._create_density_recovery_method(
            self._characteristic_function_φ_ω, self._settings['N_freq'])

        super().__init__(time_unit_transformer=time_unit_transformer,
                         asset_price_dataset_adapter=asset_price_dataset_adapter,
                         ll_optimizer=ll_optimizer,
                         n_sample_paths=n_sample_paths,
                         training_required=training_required)

    @ abstractmethod
    def _create_density_recovery_method(self, cf_φ_ω: callable, N_Freq): ...

    def _preprocess(self):
        self._x0 = np.log(float(self._s_t.head(1)['stock price']))
        self._s_t_filtered = self._s_t.drop(
            [self._s_t.index[0]], inplace=False)
        self._s_t_filtered['log_s_t'] = np.log(
            self._s_t_filtered['stock price'])

        # Capture time index
        self._s_t_filtered['t'] = self._s_t_filtered.apply(
            lambda row: row.name, axis=1)
        # Splitting the dataset into 10 parts
        self._s_t_parts = np.array_split(
            self._s_t_filtered[['log_s_t', 't']], 10)

        self._s_t = self._s_t.drop([self._s_t.index[0]], inplace=False)

    def forecast(self, T,
                 var_reduction: VarReduction = NoVarReduction(),
                 prob_dist_viz_required=False,
                 prob_dist_viz_settings: dict = {'n_workers': 5, 'ts': [5, 13, 17, 20, 22, 28]}):
        forecast_result = super().forecast(
            T, var_reduction, prob_dist_viz_required, prob_dist_viz_settings)
        forecast_result.log_scale_display = True
        return forecast_result

    @ abstractmethod
    def _characteristic_function_ϕ_ω(self, ω, θ, t): ...

    @ abstractmethod
    def _κ1(self, θ, t): ...

    @ abstractmethod
    def _κ2(self, θ, t): ...

    @ abstractmethod
    def _κ4(self, θ, t): ...

    def _pdf(self, θ: tuple):

        def _create_pdf_computation_jobs(row):
            loky_executor = get_reusable_executor(
                max_workers=self._settings.get('n_workers'))
            t = row['t']
            return loky_executor.submit(
                self._recovery_method.recover, row['log_s_t'],
                t,
                {'κ1': self._κ1(θ, t),
                 'κ2': self._κ2(θ, t),
                 'κ4': self._κ4(θ, t),
                 },
                θ
            )
        all_jobs = [part.apply(_create_pdf_computation_jobs, axis=1)
                    for part in self._s_t_parts]
        return np.concatenate(
            [job.apply(lambda row: row.result()).to_numpy() for job in all_jobs])


class RecoveredDistributionGenerator():

    """
    Class to generate samples to display densities recovered using FFT or Cosine method
    """

    def __init__(self, density_recovery_method: DensityRecoveryMethod, s_i, t, cumulants: Cumulants, θ):
        self._θ = θ
        self._cumulants = cumulants
        self._t = t
        self._x_i = np.log(s_i)
        self._recovery_method_viz = density_recovery_method

    def sample(self, n_rv):
        return np.linspace(start=self._x_i - 1.0, stop=self._x_i + 1.0, num=n_rv), None

    def target_pdf_f(self, x):
        return [self._recovery_method_viz.recover(x_i=x_i,
                                                  t=self._t,
                                                  cumulants={'κ1': self._cumulants['κ1'],
                                                             'κ2': self._cumulants['κ2'],
                                                             'κ4': self._cumulants['κ4'],
                                                             },
                                                  θ=self._θ
                                                  ) for x_i in x]
