
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


def _gaussian_cf(ω, θ, t):
    mu, σ = θ
    return np.exp(1j*mu*ω-0.5*(σ ** 2)*(ω ** 2))


def _κ1(θ):
    mu, _ = θ
    return mu


def _κ2(θ):
    _, σ = θ
    return (σ ** 2)


def _κ4(θ): return 0


def test_gaussian_density_recovery_cos_method():
    θ = (3.05, 1.2)
    b = 3.0
    x = np.linspace(start=b - 1.5, stop=b + 1.5, num=1000)
    _recover(COSMethodBasedDensityRecovery(
        N_freq=2000, ϕ_ω=_gaussian_cf), x, θ, 'COS Method')


def test_gaussian_density_recovery_fft_method():
    θ = (4.05, 11.2)
    b = 3.0
    x = np.linspace(start=b - 10.5, stop=b + 10.5, num=100)
    _recover(FFTBasedDensityRecovery(
        N_freq=3000, ϕ_ω=_gaussian_cf), x, θ, 'FFT Method')


def _recover(recovery_method: DensityRecoveryMethod, x, θ, title: str):

    true_d = pd.DataFrame(
        {'Sample': x, 'Density': norm.pdf(x, loc=θ[0], scale=θ[1]),  'type':
         'true'})

    recovered_d = pd.DataFrame({'Sample': x, 'Density': [recovery_method.recover(x_i=s_i,
                                                                                 t=100,
                                                                                 cumulants={'κ1': _κ1(θ),
                                                                                            'κ2': _κ2(θ),
                                                                                            'κ4': _κ4(θ),
                                                                                            },
                                                                                 θ=θ,
                                                                                 ) for s_i in x], 'type':
                                'recovered'})

    plt.style.use("seaborn-v0_8")
    _, ax = plt.subplots(nrows=1, ncols=2)
    sns.lineplot(ax=ax[0], data=true_d, x='Sample', y='Density', hue='type')
    sns.lineplot(ax=ax[1], data=recovered_d,
                 x='Sample', y='Density', hue='type')
    plt.title(title)
    plt.show()


def test_merton_process_paths():
    mp = MertonProcess(
        r=0.003,
        σ=0.004,
        λ=0.009,
        μ_j=0.0001,
        σ_j=0.093,
        n_sample_paths=20,
        initial_state=1000)

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
    λ_3 = 0.08
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


def test_merton_model_param_estimation():
    ticker = '^GSPC'  # S&P 500 Index
    time_freq = Frequency.DAILY
    yfa = YahooFinancialsAdapter(
        ticker=ticker,
        frequency=time_freq,
        training_set_date_range=("2011-01-01", "2014-01-01"),
        validation_set_date_range=("2014-01-01", "2015-01-01"))

    merton_asset_model = MertonProcessAssetPriceModel(time_unit_transformer=IndexedTimeTransformer(time_freq=time_freq),
                                                      asset_price_dataset_adapter=yfa, n_sample_paths=10)
    print(ticker)
    print_parameters(merton_asset_model.parameters_)


def print_parameters(params):
    for key, value in params.items():
        print(f"{key}: {value}")


def _test_forecasting_with_pdf(model, dsa):
    result = model.forecast(T=252,
                            prob_dist_viz_required=True,
                            prob_dist_viz_settings={'n_workers': 4, 'ts': [20, 100, 230]})
    result_display = ForecastResultDisplay(result, ylabel='X(t)')

    ap_back_testing = AssetPriceBackTesting(
        s_true=dsa.validation_set, s_forecast=result.mean_path)

    print('RMSE: ' + str(ap_back_testing.rmse_score))
    vz7.plot_model_forecasting_results(result_display, ap_back_testing)
    result_display.plot_probability_distributions_from_path()


def test_saved_asset_price_model():
    time_freq = Frequency.DAILY
    dsa = YahooFinancialsAdapter(
        ticker='^GSPC',
        frequency=time_freq,
        training_set_date_range=("2011-01-01", "2014-01-01"),
        validation_set_date_range=("2014-01-01", "2015-01-01"))

    params = {'s0': 1848.3599853515625,
              'r': 0.0003015488157022716,
              'σ': 0.004072847319647617,
              'λ': 1.0000000000000003e-05,
              'μ_j': 0.00010024644519917462,
              'σ_j': 0.09300358840075287,
              't0': parse('2013-12-01', date_formats=['YYYY-mm-dd'])}

    model = MertonProcessAssetPriceModel.load(parameters=params,
                                              time_unit_transformer=IndexedTimeTransformer(
                                                  time_freq=time_freq),
                                              n_sample_paths=5)
    _test_forecasting_with_pdf(model, dsa)


test_merton_process_params_familiarization()
