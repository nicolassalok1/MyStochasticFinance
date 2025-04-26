import numpy as np
from chapter5.base_forecasting import (
    TimeUnitTransformer, ForecastingProcess, TargetSamplingDensity, AssetPriceBackTesting
)

from chapter6.diffusion_model import BaseAssetPriceModel, BaseParameters
from chapter2.stock_price_dataset_adapters import StockPriceDatasetAdapter
from chapter4.random_number_gen_accept_reject import AcceptanceRejectionMethod, HatFunctionEstimator
from chapter6.diffusion_model import CommonSupremumEstimator


from scipy.stats import norm, expon
from skopt import gp_minimize
from loky import get_reusable_executor
from concurrent.futures import wait


def gaussian_kernel(x):
    return np.exp(-(x*x/2)) / np.sqrt(2*np.pi)


class CompositeJumpSamplingDensity(TargetSamplingDensity):

    '''
    Class for sampling a combination of standard normal
    variate and a normal variate with variance σ_2_J
    '''

    def __init__(self, x_arr, h, σ_2_J):
        self._x_arr = x_arr
        self._h = h
        self._σ_2_J = σ_2_J
        self._n = len(self._x_arr)

    def sample(self, n_vars, n_sample_paths=1):
        dJs = norm.rvs(size=(n_sample_paths, n_vars),
                       scale=np.sqrt(self._σ_2_J))
        dWs = norm.rvs(size=(n_sample_paths, n_vars))

        '''
          Return the variates as array of dictionaries as there 
          are two different random variables
        '''
        return np.array([[{'dW': w, 'dJ': j} for w, j in zip(dw, dj)]
                         for dw, dj in zip(dWs, dJs)])

    def pdf(self, x):
        return np.sum(gaussian_kernel((self._x_arr-x)/self._h))/(self._n*self._h)


def kth_raw_moment_of_increments(x_arr, x, h, increments_k):
    '''
     kth raw moment is computed as,
     (M_k ) ̂(x)=(∑_(t=1)^n▒K((X_t-x)/h)  (X_(t+1)-X_t )^k)/(∑_(t=1)^n▒K((X_t-x)/h) ),

     K is Gaussian Kernel
    '''
    kernel_sum = np.sum(
        gaussian_kernel(
            (x_arr-x)/h))
    # increments_k is computed as (X_(t+1)-X_t )^k
    weighted_increments = np.sum(
        gaussian_kernel(
            (x_arr-x)/h) * increments_k)

    return weighted_increments / kernel_sum


class GaussianKernelJumpProcess(ForecastingProcess):

    '''
    Class for simulating paths of Gaussian Kernel based
    jump stochastic process. 
    '''

    def __init__(self,
                 x_arr,
                 σ_2_J: float,
                 increments,
                 initial_state,
                 h,
                 n_sample_paths=20):
        # Variance of the jumps
        self._σ_2_J = σ_2_J

        # Tarining set X_t
        self._x_arr = x_arr

        # Kernel bandwidth h
        self._h = h

        # Precompute all required powers of increments (X_(t+1)-X_t )^k
        self._increments = increments
        self._increments_2 = increments ** 2
        self._increments_4 = increments ** 4

        self._state_t = initial_state

        super().__init__(n_sample_paths, initial_state=initial_state,
                         sampling_density=CompositeJumpSamplingDensity(x_arr=x_arr, h=h, σ_2_J=σ_2_J))

    def _update_current_state(self, z):
        self._reset_new_sample_path_state()

        dW = z['dW']
        dJ = z['dJ']

        λ_x = self._λ_x(self._state_t)
        σ_2_x = self._σ_2_x(self._state_t, λ_x)

        '''
         σ_2_x can be negative so, sign should be handled
         seperately to avoid error
        '''
        self._state_t = self._state_t + self._M_1_x(self._state_t) + (
            np.sqrt(abs(σ_2_x)) * np.sign(σ_2_x) * dW) + (λ_x * dJ)

        # Exponential conversion to bring back from log-scale
        return np.exp(self._state_t)

    def _λ_x(self, x):
        return self._M_4_x(x)/(3 * (self._σ_2_J ** 2))

    def _σ_2_x(self, x, λ_x):
        M_2 = self._M_2_x(x)
        return M_2 - (λ_x * self._σ_2_J)

    def _M_1_x(self, x):
        return kth_raw_moment_of_increments(x_arr=self._x_arr, x=x, h=self._h, increments_k=self._increments)

    def _M_2_x(self, x):
        return kth_raw_moment_of_increments(x_arr=self._x_arr, x=x, h=self._h, increments_k=self._increments_2)

    def _M_4_x(self, x):
        return kth_raw_moment_of_increments(x_arr=self._x_arr, x=x, h=self._h, increments_k=self._increments_4)


class GaussianKernelDensityVariateGenerator(AcceptanceRejectionMethod):

    """
    Class to sample from Markov Log Normal density using a Exponential proposal desnity
    """

    def __init__(self, x_arr, bandwidth, hat_func_optimizer: HatFunctionEstimator):
        self._x_arr = x_arr
        self._bandwidth = bandwidth
        super().__init__(hat_func_optimizer)

    def target_pdf_f(self, x):
        return np.sum(gaussian_kernel((self._x_arr-x)/self._bandwidth)) / \
            (len(self._x_arr)*self._bandwidth)

    def _proposal_pdf_g(self, x, θ: tuple):
        return expon.pdf(x, scale=1 / θ[0])

    def _sample_from_proposal_g_with_θ_optimal(self, n_rv):
        return expon.rvs(scale=1 / self._hat_func_estimator.θ_optimal_for_g[0], size=n_rv)


class GaussianKernelJumpProcessParameters(BaseParameters):
    x0: float = None
    x_arr = None
    σ_2_J = None
    increments = None
    h: float = None


class GaussianKernelJumpAssetPriceModel(BaseAssetPriceModel):

    '''
    This is a a model class for Gaussian Kernel jump process with
    estimations for h & 〖σ_Y〗^2 (Y).
    '''

    def __init__(self, time_unit_transformer: TimeUnitTransformer,
                 asset_price_dataset_adapter: StockPriceDatasetAdapter = None,
                 bandwidth='silverman',
                 bandwidth_bounds: tuple = None,
                 n_sample_paths=5,
                 training_required=True,
                 ):
        self._n = 0
        self._bandwidth = bandwidth
        self._bandwidth_bounds = bandwidth_bounds
        super().__init__(time_unit_transformer, 'GaussianKernelJumpProcessParameters', asset_price_dataset_adapter,
                         None, n_sample_paths, training_required)

    def _preprocess(self):
        self._parameters['x_arr'] = np.log(self._s_t['stock price'])
        _x_t_1 = np.log(self._s_t_1['stock price'])
        self._parameters['increments'] = self._parameters['x_arr'] - _x_t_1
        self._parameters['x_arr'] = self._parameters['x_arr'].drop(
            [self._parameters['x_arr'].index[0]], inplace=False)
        self._parameters['increments'] = self._parameters['increments'].drop(
            [self._parameters['increments'].index[0]], inplace=False)
        self._parameters['x0'] = np.log(self._parameters['s0'])
        self._n = len(self._parameters['x_arr'])

    def _fit(self):
        self._adjust_records()
        self._preprocess()

        self._estimate_bandwidth()
        self._estimate_σ_2_J()

        self._forecasting_process = self._create_forecasting_process(
            self._parameters, self._n_sample_paths)

    def _create_forecasting_process(self,
                                    parameters, n_sample_paths) -> ForecastingProcess:
        return GaussianKernelJumpProcess(
            h=parameters['h'],
            x_arr=parameters['x_arr'],
            increments=parameters['increments'],
            σ_2_J=parameters['σ_2_J'],
            initial_state=parameters['x0'],
            n_sample_paths=n_sample_paths)

    def _bayesian_search(self):
        '''
          Function to search for optimal h within a given
          bounds using Bayesian optimization.
        '''
        def _average_rmse_score(h):
            '''
              Target function to be optimized using Bayesian optimization.
              It is used inside acquisition function and returns a average 
              RMSE score by running the forecasting simulation multiple 
              times on test dataset 
            '''
            test_set = self._asset_price_dataset_adapter.validation_set
            T = len(test_set)
            self._parameters['h'] = h
            self._estimate_σ_2_J()
            temp_forecasting_process = self._create_forecasting_process(
                self._parameters, self._n_sample_paths)

            def _get_rmse_score():
                return AssetPriceBackTesting(
                    s_true=test_set, s_forecast=temp_forecasting_process.forecast(T=T).mean_path).rmse_score

            # Mutiprocessing based backtesting to compute average RMSE score on test dataset
            back_testing_executor = get_reusable_executor(max_workers=8)

            rmse_scores = [back_testing_executor.submit(
                _get_rmse_score) for _ in range(10)]
            back_testing_executor.shutdown(wait=False)
            wait(rmse_scores)
            return np.average([rmse_score.result() for rmse_score in rmse_scores])

        if self._bandwidth_bounds is None:
            raise ValueError(
                'Bandwidth bounds should be supplied if bandwidth is set as auto')

        # Run Bayesian Optimization on given bounds
        return gp_minimize(_average_rmse_score,
                           self._bandwidth_bounds, n_calls=10).x[0]

    def _estimate_bandwidth(self):
        match self._bandwidth:
            # Silverman's approach
            case 'silverman': self._parameters['h'] = 1.06 * self._parameters['x_arr'].std() * (self._n ** (-0.2))

            # Interquartile range approach
            case 'iqr':
                Q3 = self._parameters['x_arr'].quantile(0.75)
                Q1 = self._parameters['x_arr'].quantile(0.25)
                self._parameters['h'] = 1.06 * min(self._parameters['x_arr'].std(),
                                                   (Q3-Q1)/1.34) * (self._n ** (-0.2))

            case 'auto': self._parameters['h'] = self._bayesian_search()
            case _: self._parameters['h'] = self._bandwidth

    def _estimate_σ_2_J(self):
        self._parameters['σ_2_J'] = np.sum(
            self._M_6()/(5 * self._M_4()))/self._n

    def _M_6(self):
        return np.array([kth_raw_moment_of_increments(
            x_arr=self._parameters['x_arr'], x=x_i,
            h=self._parameters['h'], increments_k=self._parameters['increments'] ** 6) for x_i in self._parameters['x_arr']])

    def _M_4(self):
        return np.array([kth_raw_moment_of_increments(
            x_arr=self._parameters['x_arr'], x=x_i,
            h=self._parameters['h'], increments_k=self._parameters['increments'] ** 4) for x_i in self._parameters['x_arr']])

    def _create_forecasting_process(self,
                                    parameters, n_sample_paths) -> ForecastingProcess:
        return GaussianKernelJumpProcess(
            h=parameters['h'],
            x_arr=parameters['x_arr'],
            increments=parameters['increments'],
            σ_2_J=parameters['σ_2_J'],
            initial_state=parameters['x0'],
            n_sample_paths=n_sample_paths)

    def _pdf(self, θ: tuple):
        raise NotImplementedError()

    def _tuple_to_param_order(self, θ):
        raise NotImplementedError()

    def _get_rv_generator_for_viz(self, s_t_1):
        return GaussianKernelDensityVariateGenerator(x_arr=self._parameters['x_arr'],
                                                     bandwidth=self._parameters['h'],
                                                     hat_func_optimizer=CommonSupremumEstimator(x0_bounds=[(0.001, np.inf)],
                                                                                                θ0_bounds=[(0.0001, np.inf)]))
