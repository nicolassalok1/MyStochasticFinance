from scipy.stats import poisson
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from abc import abstractmethod, ABC

from chapter4.gaussian_mc_simulation import StandardNormalTargetSamplingDensity, TargetSamplingDensity
from chapter4.monte_carlo_simulation import MonteCarloSimulation, VarReduction, NoVarReduction
import numpy as np
from matplotlib.ticker import FormatStrFormatter


class TimeUnitTransformer(ABC):

    def __init__(self):
        self._t0 = None

    @abstractmethod
    def inverse_transform(self, path): ...


class ForecastResult:
    '''
    ForecastResult
    '''

    def __init__(self, mcs: MonteCarloSimulation.MCEstimate,
                 time_unit_transformer: TimeUnitTransformer = None):
        self._time_unit_transformer = time_unit_transformer
        self._sample_paths: pd.DataFrame = self._extract_sample_paths(mcs)
        self._mean_path: pd.DataFrame = self._extract_mean_path(mcs)
        self._uncertainty_bounds: tuple = self._extract_uncertainty_bounds(
            mcs)
        self._prob_dist_from_path = None
        self._prob_dist_from_sample_paths = None
        self._times_prob_dist_of_path = None
        self._log_scale_display = False

    def _extract_sample_paths(self, mcs: MonteCarloSimulation.MCEstimate) -> pd.DataFrame:
        ts = self._transform_time(path=mcs.samples[0])
        return pd.DataFrame(mcs.samples, columns=ts).transpose()

    def _transform_time(self, path):
        ts = None
        if self._time_unit_transformer is None:
            ts = [t for t in range(1, len(path)+1)]
        else:
            ts = self._time_unit_transformer.inverse_transform(path)
        return ts

    def _extract_mean_path(self, mcs: MonteCarloSimulation.MCEstimate) -> pd.DataFrame:
        ts = self._transform_time(path=mcs.mean)
        return pd.DataFrame([mcs.mean], columns=ts).transpose()

    def _extract_uncertainty_bounds(self, mcs: MonteCarloSimulation.MCEstimate) -> tuple:
        ts = self._transform_time(path=mcs.mean)
        lb = mcs.mean - mcs.standard_error
        ub = mcs.mean + mcs.standard_error
        return (pd.DataFrame([lb], columns=ts).transpose(), pd.DataFrame([ub], columns=ts).transpose())

    @property
    def log_scale_display(self):
        return self._log_scale_display

    @log_scale_display.setter
    def log_scale_display(self, val):
        self._log_scale_display = val

    @ property
    def sample_paths(self):
        return self._sample_paths

    @ property
    def mean_path(self):
        return self._mean_path

    @ property
    def uncertainty_bounds(self):
        return self._uncertainty_bounds

    @ property
    def probability_distributions_from_path(self):
        return self._prob_dist_from_path

    @ probability_distributions_from_path.setter
    def probability_distributions_from_path(self, prob_dist_from_path):
        self._prob_dist_from_path = prob_dist_from_path

    @ property
    def time_indices_of_probability_distributions_of_path(self):
        return self._times_prob_dist_of_path

    @ time_indices_of_probability_distributions_of_path.setter
    def time_indices_of_probability_distributions_of_path(self, time_indices):
        self._times_prob_dist_of_path = time_indices


class ForecastResultDisplay:

    def __init__(self, result: ForecastResult, xlabel='t', ylabel='X(t)'):
        self._result: ForecastResult = result
        self._xlabel = xlabel
        self._ylabel = ylabel
        plt.style.use("seaborn-v0_8")

    def plot_sample_paths(self, ax=None):
        ax1 = plt.gca() if ax is None else ax
        self._result.sample_paths.plot(ax=ax1,
                                       xlabel=self._xlabel, ylabel=self._ylabel)
        ax1.legend([])
        if ax is None:
            plt.show()

    def plot_mean_path(self, ax=None):
        ax1 = plt.gca() if ax is None else ax
        self._result.mean_path.plot(ax=ax1,
                                    xlabel=self._xlabel, ylabel=self._ylabel)
        if ax is None:
            plt.show()

    def plot_uncertainity_bounds(self, ax=None):
        ax1 = plt.gca() if ax is None else ax
        lb, ub = self._result.uncertainty_bounds

        lb.plot(ax=ax1,
                xlabel=self._xlabel, ylabel=self._ylabel, color='green')
        self._result.mean_path.plot(ax=ax1,
                                    xlabel=self._xlabel, ylabel=self._ylabel, color='blue')
        ub.plot(ax=ax1,
                xlabel=self._xlabel, ylabel=self._ylabel, color='red')

        ax1.fill_between(ax1.lines[0].get_xdata(), ax1.lines[0].get_ydata(
        ), ax1.lines[2].get_ydata(), color='grey', alpha=.5)

        ax1.legend(['Uncertainity_LB', 'Mean', 'Uncertainity_UB'])
        if ax is None:
            plt.show()

    def plot_probability_distributions_from_path(self):
        prob_dists = [prob_d.result() for prob_d in
                      self._result.probability_distributions_from_path]
        self._plot_probability_distributtions_from_path(prob_dists)

    def _plot_probability_distributtions_from_path(self, prob_dists):
        prob_dists = prob_dists[:15]
        n = len(prob_dists)
        nrows = int(n/3) + 2
        labels = np.empty(shape=(nrows, 3), dtype=object)
        c = 0
        ts = self._result.time_indices_of_probability_distributions_of_path
        if self._result.log_scale_display:
            lable_pref = 'X'
        else:
            lable_pref = 'S'

        for i in range(nrows-2):
            labels[i][0] = lable_pref + str(ts[c])
            labels[i][1] = lable_pref + str(ts[c+1])
            labels[i][2] = lable_pref + str(ts[c+2])
            c = c + 3

        labels[nrows-2] = ['path_densities']*3
        labels[nrows-1] = ['path_densities']*3

        fig, ax = plt.subplot_mosaic(
            mosaic=labels, layout='constrained')
        n = len(prob_dists)
        for i, t in enumerate(ts):
            if self._result.log_scale_display:
                legend = 'X' + str(t)
            else:
                legend = 'S' + str(t)

            ax[legend].xaxis.set_major_formatter(
                FormatStrFormatter('%.2f'))
            sns.lineplot(
                ax=ax[legend], data=prob_dists[i], x='Sample', y='Density')
            ax[legend].set_xticks(ax[legend].get_xticks())
            ax[legend].set_xticklabels(
                ax[legend].get_xticklabels(), rotation=90, ha='right')
            ax[legend].legend([legend])

        ax_path_ds_spec = ax['path_densities'].get_subplotspec()

        ax['path_densities'].remove()
        ax['path_densities'] = fig.add_subplot(
            ax_path_ds_spec, projection='3d')
        ax['path_densities'].xaxis.set_major_formatter(
            FormatStrFormatter('%.2f'))

        ys = None
        if self._result.log_scale_display:
            ys = np.log(self._result.mean_path.values[:, 0])
        else:
            ys = self._result.mean_path.values[:, 0]
        ax['path_densities'].set(ylabel=self._ylabel, zlabel='Density')
        ax['path_densities'].plot3D(
            [i for i in range(len(ys))], ys, np.zeros(len(ys)), 'blue')
        ax['path_densities'].set_zlim3d(
            [0, np.max([pdsn['Density'].max() for pdsn in prob_dists])])
        for i, t in enumerate(ts):
            prob_dists_sorted = prob_dists[i].sort_values(
                'Sample', ascending=True).reset_index(drop=True)
            ax['path_densities'].plot(
                [t for _ in range(len(prob_dists[i]))], prob_dists_sorted['Sample'], prob_dists_sorted['Density'])

        fig.tight_layout()
        plt.show()


class AssetPriceBackTesting:

    def __init__(self, s_true: pd.DataFrame, s_forecast: pd.DataFrame, col='stock price'):
        self._s_true = s_true
        self._s_forecast = s_forecast
        self._col = col
        self._rmse_score = self._compute_rmse()
        plt.style.use("seaborn-v0_8")

    def _compute_rmse(self):
        return np.sqrt(np.average((self._s_true[self._col]
                                   - self._s_forecast.iloc[:, 0].values)**2))

    @ property
    def rmse_score(self):
        return self._rmse_score

    def plot_comparison(self, ax=None):
        ax1 = plt.gca() if ax is None else ax
        self._s_true.plot(ax=ax1)
        self._s_forecast.plot(ax=ax1, style='r--')
        ax1.legend(['True Asset path', 'Forecasted Asset path'])
        if ax is None:
            plt.show()


class ForecastingProcess(ABC):

    def __init__(self, n_sample_paths, initial_state, sampling_density: TargetSamplingDensity):
        self._n_sample_paths = n_sample_paths
        self._t = 0
        self._T = 0
        self._state_t = initial_state
        self._initial_state = initial_state
        self._sampling_density = sampling_density

    def forecast(self, T, var_reduction: VarReduction = NoVarReduction(),
                 time_unit_transformer: TimeUnitTransformer = None) -> ForecastResult:
        return ForecastResult(
            mcs=self._forecast_internal(T, var_reduction),
            time_unit_transformer=time_unit_transformer)

    def _forecast_internal(self, T, var_reduction):
        self._T = T
        self._t = 0
        mcs = MonteCarloSimulation(h_x_fun=self._update_current_state,
                                   n_vars=T, n_sample_paths=self._n_sample_paths,
                                   var_reduction=var_reduction,
                                   target_sampling_density=self._sampling_density)
        e = mcs.new_estimate()
        self._state_t = self._initial_state
        return e

    @ abstractmethod
    def _update_current_state(self, z): ...

    def _reset_new_sample_path_state(self):
        if self._t >= self._T:
            self._state_t = self._initial_state
            self._t = 0
        self._t = self._t + 1


class BrownianMotionProcess(ForecastingProcess):
    '''
    BrownianMotionProcess
    '''

    def __init__(self, μ, σ,
                 initial_state=0,
                 n_sample_paths=5):
        super().__init__(initial_state=initial_state,
                         n_sample_paths=n_sample_paths, sampling_density=StandardNormalTargetSamplingDensity())
        self._μ = μ
        self._σ = σ

    def _update_current_state(self, z):
        self._reset_new_sample_path_state()
        self._state_t = self._state_t + self._μ + (self._σ * z)
        return self._state_t


class PoissonTargetSamplingDensity(TargetSamplingDensity):

    '''
    PoissonTargetSamplingDensity
    '''

    def __init__(self, λ):
        self._λ = λ

    def pdf(self, x):
        return poisson(self._λ).pmf(x)

    def sample(self, n_vars, n_sample_paths=1):
        return poisson.rvs(self._λ, size=(n_sample_paths, n_vars))


class PoissonProcess(ForecastingProcess):
    '''
    PoissonProcess
    '''

    def __init__(self, λ,
                 initial_state=0,
                 n_sample_paths=5):
        super().__init__(initial_state=initial_state,
                         n_sample_paths=n_sample_paths, sampling_density=PoissonTargetSamplingDensity(λ=λ))
        self._λ = λ

    def _update_current_state(self, z):
        self._reset_new_sample_path_state()
        self._state_t = self._state_t + z
        return self._state_t
