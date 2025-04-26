from chapter3.estimation import (
    GaussianLogLikelihoodFunctionAnalysis,
    ExponentialLogLikelihoodFunctionAnalysis,
    iterative_gaussian_gaussian_bayesian_estimation_with_prior
)

from chapter2.stock_price_dataset_adapters import YahooFinancialsAdapter
import chapter2.visualization as vs2
import chapter3.visualization as vs3

import numpy as np

def test_visualize_stock_price_for_bayesian():
    records = {'Apple Inc':
    YahooFinancialsAdapter(
        ticker="AAPL",
        training_set_date_range=("2021-01-01", "2022-06-30"),
    ).training_set}
    
    vs2.plot_security_prices(records, 'stock price')


def test_exponential_likelihood_func_analysis():
    datasets = [
        {
            "source": "Dataset 1",
            "x": np.linspace(start=200, stop=300, num=1000),
        },
        {
            "source": "Dataset 2",
            "x": np.linspace(start=2, stop=8, num=1000),
        },
    ]
    θ_sets = {"λ": np.linspace(start=0, stop=3, num=500)}

    ExponentialLogLikelihoodFunctionAnalysis.for_parameters_and_datasets(
        θ_sets=θ_sets, datasets=datasets
    ).plot()


def test_gaussian_likelihood_func_analysis():
    datasets = [
        {
            "source": "Apple Inc",
            "x": YahooFinancialsAdapter(
                ticker="AAPL",
                training_set_date_range=("2021-02-01", "2021-04-30"),
            ).training_set["stock price"],
        },
        {
            "source": "ADP",
            "x": YahooFinancialsAdapter(
                ticker="ADP",
                training_set_date_range=("2021-02-01", "2021-04-30"),
            ).training_set["stock price"],
        },
    ]

    θ_sets = {
        "μ": np.linspace(start=100.0, stop=200, num=10),
        "σ2": np.linspace(start=100.0, stop=400, num=10),
    }

    GaussianLogLikelihoodFunctionAnalysis.for_parameters_and_datasets(
        θ_sets=θ_sets, datasets=datasets
    ).plot(θ_names=["μ", "σ2"])

def test_iterative_bayesian_estimation():
    # Observations for Bayesian parameter estimation
    yf_adapter = YahooFinancialsAdapter(
        ticker="AAPL",
        training_set_date_range=("2021-01-01", "2022-06-30"),
    )
    x = yf_adapter.training_set['stock price']

    σ2 = 100 # Asssume σ2 is constant and known to us, so no prior distribution is set on this

    # Compute posterior distrubutions for a collection of four prior parameter settings (for μ)
    prob_x_arr = [
        iterative_gaussian_gaussian_bayesian_estimation_with_prior(x=x,prior_α=110, prior_β_2=5, σ2=σ2),
        iterative_gaussian_gaussian_bayesian_estimation_with_prior(x=x,prior_α=146, prior_β_2=2, σ2=σ2),
        iterative_gaussian_gaussian_bayesian_estimation_with_prior(x=x,prior_α=150, prior_β_2=20, σ2=σ2),
        iterative_gaussian_gaussian_bayesian_estimation_with_prior(x=x,prior_α=160, prior_β_2=0.5, σ2=σ2),
    ]
    vs3.bayesian_estimation_plot(prob_x_arr)
