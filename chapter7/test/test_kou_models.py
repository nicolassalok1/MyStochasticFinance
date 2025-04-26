
from chapter5.base_forecasting import ForecastResultDisplay, AssetPriceBackTesting
import chapter7.visualization as vz7
from chapter7.kou_model import KouProcess, KouProcessAssetPriceModel, AsymmetricDoubleExponentialGenerator
from chapter2.stock_price_dataset_adapters import YahooFinancialsAdapter, Frequency
from chapter6.diffusion_model import IndexedTimeTransformer
from dateparser import parse
from chapter4.random_number_gen_accept_reject import DefaultSupremumEstimator


def test_sampling_of_asym_double_exponential():
    gaussian_supremum_estimator = DefaultSupremumEstimator(
        x0=[0.00001], x0_bounds=[(0.00001, None)], θ0=[0.00001, 0.00001], θ0_bounds=[(0.00001, None), (0.00001, None)])
    _, sample_trace = AsymmetricDoubleExponentialGenerator(p=0.7, α_1=2.4, α_2=0.7,
                                                           hat_func_optimizer=gaussian_supremum_estimator).sample(n_rv=1000)
    sample_trace.plot()


def test_kou_process_paths():
    mp = KouProcess(
        r=0.00032894797531725286, σ=0.009310530997044748,
        λ=1e-05,
        p=0.011733673454519264, α_1=1.1466664732568665, α_2=1.4526511493269534, n_sample_paths=10, initial_state=1848.3599853515625)

    result = mp.forecast(T=250)
    result_display = ForecastResultDisplay(result, ylabel='S(t)')
    vz7.plot_merton_process_paths_for_single_set_params(result_display)


def test_kou_process_params_familiarization():

    r_1 = 0.0003
    σ_1 = 0.009
    λ_1 = 0.01
    p_1 = 0.1
    α_1_1 = 2.5
    α_2_1 = 1.5

    kp_1 = KouProcess(r=r_1,
                      σ=σ_1,
                      λ=λ_1,
                      p=p_1,
                      α_1=α_1_1,
                      α_2=α_2_1,
                      n_sample_paths=5)
    result_display_1 = ForecastResultDisplay(
        result=kp_1.forecast(T=500), ylabel='S(t)')

    r_2 = -0.0003
    σ_2 = 0.009
    λ_2 = 0.02
    p_2 = 0.8
    α_1_2 = 4.5
    α_2_2 = 2.5

    kp_2 = KouProcess(r=r_2, σ=σ_2, λ=λ_2, p=p_2,
                      α_1=α_1_2, α_2=α_2_2, n_sample_paths=5)
    result_display_2 = ForecastResultDisplay(
        result=kp_2.forecast(T=500), ylabel='S(t)')

    vz7.plot_kou_process_parameter_famailiarization(
        (result_display_1, r_1, σ_1, λ_1, p_1, α_1_1,
         α_2_1), (result_display_2, r_2, σ_2, λ_2, p_2,
                  α_1_2,
                  α_2_2))


def test_kou_model_param_estimation():
    ticker = '^GSPC'  # S&P 500 Index
    time_freq = Frequency.DAILY
    yfa = YahooFinancialsAdapter(
        ticker=ticker,
        frequency=time_freq,
        training_set_date_range=("2011-01-01", "2014-01-01"),
        validation_set_date_range=("2014-01-01", "2015-01-01"))

    kou_asset_model = KouProcessAssetPriceModel(time_unit_transformer=IndexedTimeTransformer(time_freq=time_freq),
                                                asset_price_dataset_adapter=yfa, n_sample_paths=10)
    print(ticker)
    print_parameters(kou_asset_model.parameters_)


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
              'r': 0.00032894797531725286,
              'σ': 0.009310530997044748,
              'λ': 1e-05,
              'p': 0.011733673454519264,
              'α_1': 1.1466664732568665,
              'α_2': 1.4526511493269534,
              't0': parse('2013-12-01', date_formats=['YYYY-mm-dd'])}

    model = KouProcessAssetPriceModel.load(parameters=params,
                                           time_unit_transformer=IndexedTimeTransformer(
                                               time_freq=time_freq),
                                           n_sample_paths=5)
    _test_forecasting_with_pdf(model, dsa)


# test_sampling_of_asym_double_exponential()
test_kou_process_params_familiarization()
# test_acceptance_rejection_de_with_gaussian()
# test_kou_model_param_estimation()
# test_kou_process_paths()
# test_merton_model_forecast()
# test_saved_asset_price_model()
# test_gaussian_density_recovery_cos_method()
# test_gaussian_density_recovery_fft_method()
# test_merton_process_paths()
