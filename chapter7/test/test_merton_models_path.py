
from chapter5.base_forecasting import ForecastResultDisplay, AssetPriceBackTesting
import chapter7.visualization as vz7
from chapter7.merton_model import MertonProcess, MertonProcessAssetPriceModel
from chapter7.density_recovery_methods import COSMethodBasedDensityRecovery, DensityRecoveryMethod, FFTBasedDensityRecovery
from chapter2.stock_price_dataset_adapters import YahooFinancialsAdapter, Frequency
from chapter6.diffusion_model import IndexedTimeTransformer
from dateparser import parse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import pandas as pd


def test_merton_process_paths():
    mp = MertonProcess(
        r=0.001, σ=0.03,
        λ=0.009,
        μ_j=0.02, σ_j=0.03, n_sample_paths=10, initial_state=1000)

    result = mp.forecast(T=250)
    result_display = ForecastResultDisplay(result, ylabel='S(t)')
    vz7.plot_merton_process_paths_for_single_set_params(result_display)


def test_merton_process_params_familiarization():

    r_1 = -0.003
    σ_1 = 0.04
    λ_1 = 0.0001
    μ_j_1 = 0.02
    σ_j_1 = 0.005
    mp_1 = MertonProcess(r=r_1, σ=σ_1, λ=λ_1, μ_j=μ_j_1,
                         σ_j=σ_j_1, n_sample_paths=5)
    result_display_1 = ForecastResultDisplay(
        result=mp_1.forecast(T=500), ylabel='S(t)')

    r_2 = 0.01
    σ_2 = 0.01
    λ_2 = 0.01
    μ_j_2 = 1.1
    σ_j_2 = 1.05
    mp_2 = MertonProcess(r=r_2, σ=σ_2, λ=λ_2, μ_j=μ_j_2,
                         σ_j=σ_j_2, n_sample_paths=5)
    result_display_2 = ForecastResultDisplay(
        result=mp_2.forecast(T=500), ylabel='S(t)')

    r_3 = 0.05
    σ_3 = 0.2
    λ_3 = 1.0
    μ_j_3 = 0
    σ_j_3 = 0.5
    mp_3 = MertonProcess(r=r_3, σ=σ_3, λ=λ_3, μ_j=μ_j_3,
                         σ_j=σ_j_3, n_sample_paths=5)
    result_display_3 = ForecastResultDisplay(
        result=mp_3.forecast(T=100), ylabel='S(t)')

    r_4 = -0.004
    σ_4 = 0.1
    λ_4 = 0.0066
    μ_j_4 = 0.5
    σ_j_4 = 0.5
    mp_4 = MertonProcess(r=r_4, σ=σ_4, λ=λ_4, μ_j=μ_j_4,
                         σ_j=σ_j_4, n_sample_paths=5)
    result_display_4 = ForecastResultDisplay(
        result=mp_4.forecast(T=500), ylabel='S(t)')
    vz7.plot_merton_process_parameter_famailiarization(
        (result_display_1, r_1, σ_1, λ_1, μ_j_1,
         σ_j_1), (result_display_2, r_2, σ_2, λ_2, μ_j_2,
                  σ_j_2),
        (result_display_3, r_3, σ_3, λ_3, μ_j_3,
         σ_j_3), (result_display_4, r_4, σ_4, λ_4, μ_j_4,
                  σ_j_4))


test_merton_process_params_familiarization()
