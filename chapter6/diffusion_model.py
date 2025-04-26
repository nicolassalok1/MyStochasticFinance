import numpy as np
import inspect
import pandas as pd
from abc import abstractmethod, ABC

from scipy.optimize import shgo
from loky import get_reusable_executor
from scipy.stats import expon
from typing import List, TypedDict
from dateutil.relativedelta import relativedelta
from dateparser import parse
from datetime import datetime
from chapter2.stock_price_dataset_adapters import Frequency
from chapter4.monte_carlo_simulation import VarReduction, NoVarReduction
from chapter4.random_number_gen_accept_reject import AcceptanceRejectionMethod, HatFunctionEstimator, SupremumEstimatorTemplate
from chapter2.stock_price_dataset_adapters import StockPriceDatasetAdapter
from chapter5.base_forecasting import (
    BrownianMotionProcess, TimeUnitTransformer, ForecastingProcess)


class CommonSupremumEstimator(SupremumEstimatorTemplate):

    '''
      Default implemenation of SupremumEstimatorTemplate.
      It leverage scipy.optimize module to find maxima & minima of
      f/g ratio.
    '''

    def __init__(self,
                 x0_bounds: List[tuple],
                 # Sequence of (min, max) pairs of parameter  of θ
                 θ0_bounds: List[tuple]
                 ):
        self._x0_bounds = x0_bounds
        self._θ0_bounds = θ0_bounds
        super().__init__()

    def _maximize_wrt_x(self, ratio_f: callable) -> tuple:
        res = shgo(ratio_f,  bounds=self._x0_bounds)
        return -res.fun, res.x

    def _minimize_wrt_θ(self, ratio_f: callable) -> tuple:
        res = shgo(ratio_f,  bounds=self._θ0_bounds)
        return res.fun, res.x


class MarkovLogNormalVariateGenerator(AcceptanceRejectionMethod):

    """
    Class to sample from Markov Log Normal density using a Exponential proposal desnity
    """

    def __init__(self, r, σ, x_t_1, hat_func_optimizer: HatFunctionEstimator):
        self._r = r
        self._σ = σ
        self._x_t_1 = x_t_1
        super().__init__(hat_func_optimizer)

    def target_pdf_f(self, x):
        ll_factor_1 = 1.0/(np.multiply(self._σ, x) * np.sqrt(2*np.pi))
        ll_factor_2 = np.exp(
            -0.5*np.power((np.log(np.divide(x, self._x_t_1)) - (self._r-0.5*self._σ*self._σ))/self._σ, 2))

        return ll_factor_1 * ll_factor_2

    def _proposal_pdf_g(self, x, θ: tuple):
        return expon.pdf(x, scale=1 / θ[0])

    def _sample_from_proposal_g_with_θ_optimal(self, n_rv):
        return expon.rvs(scale=1 / self._hat_func_estimator.θ_optimal_for_g[0], size=n_rv)

    @property
    def n_rv(self): return 5000


class GeometricBrownianMotionProcess(BrownianMotionProcess):
    '''
    GeometricBrownianMotionProcess
    '''

    def __init__(self, r, σ,
                 initial_state=1.0,
                 n_sample_paths=5):

        self._r = r
        self._σ = σ
        super().__init__(μ=self._r - (self._σ ** 2 / 2.0),
                         σ=self._σ,
                         initial_state=initial_state,
                         n_sample_paths=n_sample_paths)

    def _update_current_state(self, z):
        self._reset_new_sample_path_state()
        self._state_t = self._state_t * np.exp(self._μ + (self._σ * z))
        return self._state_t


class IndexedTimeTransformer(TimeUnitTransformer):

    def __init__(self, time_freq: Frequency):
        self._time_freq = time_freq

    def inverse_transform(self, path):
        l = len(path)
        match self._time_freq:
            case Frequency.MONTHLY:
                return [
                    str(self._t0 + relativedelta(months=m)).split(' ')[0] for m in range(1, l+1)]
            case Frequency.WEEKLY:
                return [
                    str(self._t0 + relativedelta(weeks=w)).split(' ')[0] for w in range(1, l+1)]
            case Frequency.DAILY:
                return [
                    str(self._t0 + relativedelta(days=d)).split(' ')[0] for d in range(1, l+1)]


class LoglikelihoodOptimizer(ABC):

    @abstractmethod
    def optimize(log_likelihood_func: callable): ...


class DefaultSHGOOptimizer(LoglikelihoodOptimizer):

    def __init__(self, θ0_bounds: List[tuple]):
        self._θ0_bounds = θ0_bounds

    def optimize(self, log_likelihood_func: callable):
        result = shgo(log_likelihood_func, bounds=self._θ0_bounds)
        return result.x


diffusion_process_ll_optimizer = DefaultSHGOOptimizer(
    θ0_bounds=[(-5, 5), (0.001, 5)])


class BaseParameters(TypedDict):
    s0: float = None
    t0: datetime = None


class DiffusionProcessParameters(BaseParameters):
    r: float = None
    σ: float = None


class BaseAssetPriceModel(ABC):

    def __init__(self, time_unit_transformer: TimeUnitTransformer,
                 param_type: str = None,
                 asset_price_dataset_adapter: StockPriceDatasetAdapter = None,
                 ll_optimizer: LoglikelihoodOptimizer = diffusion_process_ll_optimizer,
                 n_sample_paths=100,
                 training_required=True):
        self._time_u_t = time_unit_transformer
        self._ll_optimizer = ll_optimizer
        self._n_sample_paths = n_sample_paths
        self._asset_price_dataset_adapter = asset_price_dataset_adapter
        self._training_required = training_required
        self._forecasting_process = None
        self._parameters = self._create_empty_param_instance()
        self._s_t_1 = None
        self._last_s = None

        print('BaseAssetPriceModel.training_required' + str(training_required))
        if training_required:
            if asset_price_dataset_adapter is None:
                raise ValueError(
                    'StockPriceDatasetAdapter is required for model training.')
            if ll_optimizer is None:
                raise ValueError(
                    'An optimizer is required for model training.')
            self._s_t = asset_price_dataset_adapter.training_set
            print(self._parameters)
            self._fit()

    def _adjust_records(self):
        self._s_t_1 = self._s_t.shift(1)
        self._last_s = self._s_t.tail(1)

        self._time_u_t._t0 = parse(
            self._last_s['time'].iat[0], date_formats=['YYYY-MM-DD'])
        self._s_t_1.iloc[0, 1] = self._s_t.iloc[0, 1]

        self._parameters['s0'] = float(self._last_s['stock price'])
        self._parameters['t0'] = self._time_u_t._t0

    def _fit(self):
        self._adjust_records()
        self._preprocess()
        θ = self._ll_optimizer.optimize(self._negative_loglikelihood)
        self._tuple_to_param_order(θ)
        self._forecasting_process = self._create_forecasting_process(
            self._parameters, self._n_sample_paths)

    def _negative_loglikelihood(self, θ: tuple):
        return -np.sum(np.log(self._pdf(θ)))

    def forecast(self, T,
                 var_reduction: VarReduction = NoVarReduction(),
                 prob_dist_viz_required=False,
                 prob_dist_viz_settings: dict = {'n_workers': 5, 'ts': [5, 13, 17, 20, 22, 28]}):

        forecast_result = self._forecasting_process.forecast(
            T=T, var_reduction=var_reduction, time_unit_transformer=self._time_u_t)

        self._compute_prob_distribution_of_mean_path(
            forecast_result, prob_dist_viz_required, prob_dist_viz_settings)
        return forecast_result

    def _compute_prob_distribution_of_mean_path(self, forecast_result, prob_dist_viz_required, prob_dist_viz_settings):
        if prob_dist_viz_required:
            if '_get_rv_generator_for_viz' not in dict(inspect.getmembers(type(self), inspect.isfunction)):
                raise NotImplementedError(
                    'Subclass should define function _get_rv_generator_for_viz')

            forecast_result.time_indices_of_probability_distributions_of_path = prob_dist_viz_settings[
                'ts']

            def _samples_gen_task(s_t_1, t):
                '''
                   Inner function to be used through a multiprocessor
                   for performing a sinle task of generating RVs
                '''
                rv_generator = getattr(
                    self, '_get_rv_generator_for_viz')(s_t_1=s_t_1, t=t)

                # generated samples & their probabilities
                samples, _ = rv_generator.sample(n_rv=500)
                probs = rv_generator.target_pdf_f(samples)

                return pd.DataFrame({"Sample": samples, "Density": probs})

            def _compute_distributions_from_path(prob_dist_viz_settings, forecast_result, _samples_gen_task):
                '''
                   Inner function to compute the distributions and
                   append to the forecast result
                '''

                # Task execuor from loky
                viz_task_executor = get_reusable_executor(
                    max_workers=prob_dist_viz_settings['n_workers'])

                mean_path = forecast_result.mean_path.values  # forecasted mean path

                # Sumiiting task for all of the required t's
                forecast_result.probability_distributions_from_path = [viz_task_executor.submit(_samples_gen_task,
                                                                                                mean_path[task_i-1][0],
                                                                                                # for S_t_1
                                                                                                task_i-1) for task_i in
                                                                       prob_dist_viz_settings['ts']]
                viz_task_executor.shutdown(wait=False)

            _compute_distributions_from_path(
                prob_dist_viz_settings, forecast_result, _samples_gen_task)

    @property
    def parameters_(self):
        return self._parameters

    @classmethod
    def load(cls, parameters: BaseParameters,
             time_unit_transformer: TimeUnitTransformer,
             n_sample_paths=100,
             ):
        instance = cls(time_unit_transformer=time_unit_transformer,
                       n_sample_paths=n_sample_paths,
                       ll_optimizer=None,
                       training_required=False)
        instance._time_u_t = time_unit_transformer
        instance._time_u_t._t0 = parameters['t0']
        instance._x0 = np.log(parameters['s0'])

        instance._parameters = parameters
        instance._forecasting_process = instance._create_forecasting_process(
            parameters, n_sample_paths=n_sample_paths)

        return instance

    @abstractmethod
    def _create_empty_param_instance(self): ...

    @abstractmethod
    def _preprocess(self): ...

    @abstractmethod
    def _tuple_to_param_order(self, t: tuple): ...

    @abstractmethod
    def _pdf(self, θ: tuple): ...

    @abstractmethod
    def _create_forecasting_process(self,
                                    parameters, n_sample_paths) -> ForecastingProcess: ...


class DiffusionProcessAssetPriceModel(BaseAssetPriceModel):

    def __init__(self, time_unit_transformer: TimeUnitTransformer,
                 asset_price_dataset_adapter: StockPriceDatasetAdapter = None,
                 ll_optimizer: LoglikelihoodOptimizer = diffusion_process_ll_optimizer,
                 n_sample_paths=100,
                 training_required=True,
                 ):
        self._log_ratio = None
        super().__init__(time_unit_transformer, 'DiffusionProcessParameters', asset_price_dataset_adapter,
                         ll_optimizer, n_sample_paths, training_required)

    def _tuple_to_param_order(self, θ):
        self._parameters['r'], self._parameters['σ'] = θ

    def _create_empty_param_instance(self):
        return DiffusionProcessParameters()

    def _preprocess(self):
        self._log_ratio = np.log(
            self._s_t['stock price'] / self._s_t_1['stock price'])
        self._s_t = self._s_t.drop([self._s_t.index[0]], inplace=False)
        self._log_ratio = self._log_ratio.drop(
            [self._log_ratio.index[0]], inplace=False)

    def _pdf(self, θ: tuple):
        r, σ = θ
        ll_factor_1 = (1/(σ * self._s_t['stock price']
                       * np.sqrt(2*np.pi)))
        ll_factor_2 = np.exp(
            -0.5*np.power((self._log_ratio -
                           (r-0.5*(σ**2)))/σ, 2))

        return ll_factor_1 * ll_factor_2

    def _create_forecasting_process(self,
                                    parameters, n_sample_paths) -> ForecastingProcess:
        return GeometricBrownianMotionProcess(
            r=parameters['r'],
            σ=parameters['σ'],
            initial_state=parameters['s0'],
            n_sample_paths=n_sample_paths)

    def _get_rv_generator_for_viz(self, s_t_1, t):
        return MarkovLogNormalVariateGenerator(r=self._parameters['r'],
                                               σ=self._parameters['σ'],
                                               x_t_1=s_t_1,
                                               hat_func_optimizer=CommonSupremumEstimator(x0_bounds=[(0.001, np.inf)],
                                                                                          θ0_bounds=[(0.0001, np.inf)]))
