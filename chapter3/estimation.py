from scipy.stats import norm
from scipy.stats import expon
import numpy as np
from chapter3 import visualization as vs3

from chapter2.stock_price_dataset_adapters import YahooFinancialsAdapter

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd


from abc import ABC, abstractmethod
from typing import TypedDict, List, Dict


class LogLikelihoodFunctionAnalysis(ABC):

    """
    Base class for Loglikelihood function for any continuous probability distribution. This 
    should be extended, and _compute_likelihood function should be overridden to have 
    any density-specific behaviour. 

    This class provides a study of the log-likelihood function with appropriate visualization. 

    """
    __instance_key = object()

    class Dataset(TypedDict):
        """
           This specially typed dictionary works as a key-valued dataset. 'source' is the 
           name of the data source, and 'x' is the data array.  
       """

        source: str
        x: List

    def __init__(self, instance_key,
                 θ_sets: Dict[str, List],
                 datasets: List[Dataset]):
        assert (
            instance_key == LogLikelihoodFunctionAnalysis.__instance_key
        ), "LogLikelihoodFunctionAnalysis cannot be instantiated explicitely from outside. Always use instantiate function"

        self._θ_sets = θ_sets
        self._datasets = datasets
        self._total_loglikelihood = self._compute_total_loglikelihood()
        self._max_loglikelihood_details = self._get_max_loglikelihoods()

    @abstractmethod
    def _compute_likelihood(self, x, **θ):
        """
        The sub-class should override this function. It should return the likelihood
        of x. You may use the readily available likelihood function or implement any custom 
        one.
        """
        ...

    @classmethod
    def for_parameters_and_datasets(
        cls, θ_sets: Dict[str, List], datasets: List[Dataset]
    ):
        """
        Factory function to create new instance of LogLikelihoodFunctionAnalysis
        """
        return cls(LogLikelihoodFunctionAnalysis.__instance_key, θ_sets, datasets)

    @property
    def max_loglikelihood_details(self):
        return self._max_loglikelihood_details

    import numpy as np

    def _prepare_combibnations_for_θ(self) -> Dict[str, List]:
        """
        Prepare combinations of parameters from the list of supplied simulated values.
        For example, in a two-parameter setting, if the supplied values are [3,5,10] and 
        [-6,8,190] respectively, then the few combinations are (3,-6), (3,190),(5,8), 
        (5,-6) and so on. 

        The function returns combinations as a dictionary of values, keeping 
        the positional indices intact.
        """
        θ_grid = None
        θ_name_grid_index = {}
        for i, (θ_name, θ_val) in enumerate(self._θ_sets.items()):
            if i == 0:
                θ_grid = np.meshgrid(θ_val)
            else:
                θ_grid = np.meshgrid(θ_grid, θ_val)

            θ_name_grid_index[θ_name] = i

        return {
            θ_name: θ_grid[θ_index].flatten()
            for θ_name, θ_index in θ_name_grid_index.items()
        }

    def _compute_total_loglikelihood(self):
        """
        Computes total log-likelihood for each of the data sources in self._datasets.
        It uses combinations of parameters as returned by _prepare_combibnations_for_θ 
        and feeds each of those into the likelihood function to create parameter likelihood 
        tuples.
        """
        total_llh = {}

        def _get_single_name_value_for_θ(index, θ_combs):
            return {
                θ_combs_k: θ_combs_v[index] for θ_combs_k, θ_combs_v in θ_combs.items()
            }

        θ_combs = self._prepare_combibnations_for_θ()
        num_θ_values = len(list(θ_combs.values())[0])

        # Create dictionaries of tuples of format (θ, likelihood) for each dataset
        for ds in self._datasets:
            llh = [
                (
                    _get_single_name_value_for_θ(i, θ_combs),
                    self.get_loglikelihood_for_observations(
                        ds["x"], **_get_single_name_value_for_θ(i, θ_combs)
                    ),
                )
                for i in range(num_θ_values)
            ]

            total_llh[ds["source"]] = llh

        return total_llh

    def _get_max_loglikelihoods(self):
        """
         It iterates over all log-likelihoods and returns maximum for each data sources. 
        """

        return {
            k: max(v, key=lambda t: t[1]) for k, v in self._total_loglikelihood.items()
        }

    def get_loglikelihood_for_observations(self, x, **θ):
        """
        Gets total log likelihood for a given observation. 
        This function can be used for parameter optimization by external 
        components.
        """
        return np.sum(np.log(self._compute_likelihood(x, **θ)))

    def plot(self, θ_names: List[str] = None):
        plt.style.use("seaborn")

        def _annotate_max_likllihood_point(ax, source):
            max_loglikelihood_point = self._max_loglikelihood_details[source]
            liklihood_val = max_loglikelihood_point[1]
            if len(self._θ_sets) == 1:
                θ_name = list(max_loglikelihood_point[0].keys())[0]
                θ_val = list(max_loglikelihood_point[0].values())[0]
                ax.text(
                    θ_val,
                    liklihood_val,
                    θ_name + " = " + str(round(θ_val, 3)),
                )
            else:
                θ_val_1 = max_loglikelihood_point[0][θ_names[0]]
                θ_val_2 = max_loglikelihood_point[0][θ_names[1]]
                ax.text(
                    θ_val_1,
                    θ_val_2,
                    liklihood_val,
                    "X",
                    color="green",
                )
                ax.text(
                    θ_val_1 + 10,
                    θ_val_2 + 10,
                    liklihood_val + 1,
                    "Optimal ("
                    + θ_names[0]
                    + ","
                    + θ_names[1]
                    + ") = ("
                    + str(round(θ_val_1))
                    + ","
                    + str(round(θ_val_2))
                    + ")",
                )

        if len(self._θ_sets) == 1:
            θ_name = list(self._θ_sets.keys())[0]
            records_df = pd.DataFrame()
            for source, liklihood_details in self._total_loglikelihood.items():
                record = {}
                record["Source"] = source

                t_arr = np.array(liklihood_details)
                θ_name_val_df = pd.DataFrame.from_records(t_arr[:, 0])

                record[θ_name] = θ_name_val_df[θ_name]
                record["Log Likelihood"] = t_arr[:, 1]

                records_df = pd.concat(
                    [records_df, pd.DataFrame(record)], ignore_index=True
                )
                _annotate_max_likllihood_point(plt.gca(), source)

            sns.lineplot(
                data=records_df,
                x=θ_name,
                y="Log Likelihood",
                hue="Source",
                style="Source",
                lw=3,
            )
        else:
            fig = plt.figure(figsize=(10, 7))
            n = len(self._total_loglikelihood)
            row = int(n / 2)

            def _plot_for_single_source(source, liklihood_details, i):
                ax = fig.add_subplot(
                    row, 2, i, projection="3d", computed_zorder=False)
                t_arr = np.array(liklihood_details)
                θ_name_val_df = pd.DataFrame.from_records(t_arr[:, 0])
                ax.plot_trisurf(
                    θ_name_val_df[θ_names[0]],
                    θ_name_val_df[θ_names[1]],
                    list(t_arr[:, 1]),
                    cmap=plt.cm.gnuplot2,
                    edgecolor="black",
                    linewidth=0.2,
                    zorder=1,
                )
                ax.set_xlabel(θ_names[0])
                ax.set_ylabel(θ_names[1])
                ax.set_zlabel("Log likelihood")
                ax.set_title(source)

                _annotate_max_likllihood_point(ax, source)

            i = 1
            for source, liklihood_details in self._total_loglikelihood.items():
                _plot_for_single_source(source, liklihood_details, i)
                i = i + 1

            fig.tight_layout()
        plt.show()


class ExponentialLogLikelihoodFunctionAnalysis(LogLikelihoodFunctionAnalysis):
    """
      Class for studying the likelihood function of Exponential distribution with 
      parameter λ.

    """

    def _compute_likelihood(self, x, λ):
        return expon.pdf(x, loc=0, scale=1 / λ)


class GaussianLogLikelihoodFunctionAnalysis(LogLikelihoodFunctionAnalysis):
    """
      Class for studying the likelihood function of Gaussian distribution with 
      parameters μ & σ2.
    """

    def _compute_likelihood(self, x, μ, σ2):
        return norm.pdf(x, loc=μ, scale=np.sqrt(σ2))


def iterative_gaussian_gaussian_bayesian_estimation_with_prior(x,
                                                               prior_α=None,
                                                               prior_β_2=None,
                                                               σ2=None):
    '''
        Bayesian belief update algorithm for Gaussian-Gaussian settings.
    '''
    posterior_α = 0.0
    posterior_β_2 = 0.0

    temp_prior_α = prior_α
    temp_prior_β_2 = prior_β_2

    # Iteratively compute posteriors in closed form for the Gaussian-Gaussian
    # settings assuming x is a streaming dataset.
    for x_i in x:
        posterior_β_2 = (temp_prior_β_2 * σ2) / (
            temp_prior_β_2 + σ2
        )

        posterior_α = posterior_β_2 * (
            (x_i / σ2) + (temp_prior_α / temp_prior_β_2)
        )

        # Update priors with computed posteriors for next iteration
        temp_prior_β_2 = posterior_β_2
        temp_prior_α = posterior_α

    # Draw samples from prior & posteror distrubutions
    prior_μ_rvs = norm.rvs(loc=prior_α, scale=np.sqrt(prior_β_2), size=1000)
    posterior_μ_rvs = norm.rvs(
        loc=posterior_α, scale=np.sqrt(posterior_β_2), size=1000
    )

    # Compute all sample likelihoods from prior & posterior distrubutions
    # for visualization
    prob_x = {
        "Prior μ": (
            prior_μ_rvs,
            norm.pdf(prior_μ_rvs, loc=prior_α, scale=np.sqrt(prior_β_2)),
        ),
        "Posterior μ": (
            posterior_μ_rvs,
            norm.pdf(
                posterior_μ_rvs,
                loc=posterior_α,
                scale=np.sqrt(posterior_β_2),
            ),
        ),
    }

    return prob_x, prior_α, posterior_α
