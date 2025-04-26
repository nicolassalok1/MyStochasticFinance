from chapter6.diffusion_model import (GeometricBrownianMotionProcess,
                                      DiffusionProcessAssetPriceModel,
                                      IndexedTimeTransformer)
from chapter5.base_forecasting import ForecastResultDisplay, AssetPriceBackTesting
import chapter6.visualization as vz6
from chapter2.stock_price_dataset_adapters import YahooFinancialsAdapter, Frequency
from dateparser import parse


def test_gbm_process_paths():
    bmp = GeometricBrownianMotionProcess(
        r=0.001990902456610044, σ=0.0021216124821245566,
        n_sample_paths=2, initial_state=1848.3599853515625)

    result = bmp.forecast(T=100)
    result_display = ForecastResultDisplay(result, ylabel='S(t)')
    result_display.plot_sample_paths()
    result_display.plot_mean_path()
    result_display.plot_uncertainity_bounds()


def test_gbm_process_params_familiarization():

    r_1 = -0.003
    σ_1 = 0.04
    bmp_1 = GeometricBrownianMotionProcess(r=r_1, σ=σ_1, n_sample_paths=5)
    result_display_1 = ForecastResultDisplay(
        result=bmp_1.forecast(T=500), ylabel='S(t)')

    r_2 = 0.001
    σ_2 = 0.01
    bmp_2 = GeometricBrownianMotionProcess(r=r_2, σ=σ_2, n_sample_paths=5)
    result_display_2 = ForecastResultDisplay(
        result=bmp_2.forecast(T=500), ylabel='S(t)')

    r_3 = 0.005
    σ_3 = 0.05
    bmp_3 = GeometricBrownianMotionProcess(r=r_3, σ=σ_3, n_sample_paths=5)
    result_display_3 = ForecastResultDisplay(
        result=bmp_3.forecast(T=500), ylabel='S(t)')

    r_4 = -0.004
    σ_4 = 0.02
    bmp_4 = GeometricBrownianMotionProcess(r=r_4, σ=σ_4, n_sample_paths=5)
    result_display_4 = ForecastResultDisplay(
        result=bmp_4.forecast(T=500), ylabel='S(t)')

    vz6.plot_gbm_parameter_famailiarization(
        (result_display_1, r_1, σ_1), (result_display_2, r_2, σ_2),
        (result_display_3, r_3, σ_3), (result_display_4, r_4, σ_4))


def test_asset_price_model_estimation():
    ticker = '^GSPC'  # S&P 500 Index
    time_freq = Frequency.WEEKLY
    yfa = YahooFinancialsAdapter(
        ticker=ticker,
        frequency=time_freq,
        training_set_date_range=("2010-01-01", "2015-01-01"),
        validation_set_date_range=("2015-01-01", "2019-07-01"))

    dfp_model = DiffusionProcessAssetPriceModel(time_unit_transformer=IndexedTimeTransformer(time_freq=time_freq),
                                                asset_price_dataset_adapter=yfa)
    print(ticker)
    print_parameters(dfp_model.parameters_)

    # _test_forecasting(dfp_model, yfa)


def print_parameters(params):
    for key, value in params.items():
        print(f"{key}: {value}")


def _test_forecasting(model, dsa):
    result = model.forecast(T=len(dsa.validation_set),
                            prob_dist_viz_required=True,
                            prob_dist_viz_settings={'n_workers': 4, 'ts': [50, 85, 150]})
    result_display = ForecastResultDisplay(result, ylabel='S(t)')

    ap_back_testing = AssetPriceBackTesting(
        s_true=dsa.validation_set, s_forecast=result.mean_path)

    print('RMSE: ' + str(ap_back_testing.rmse_score))
    vz6.plot_model_forecasting_results(result_display, ap_back_testing)
    result_display.plot_probability_distributions_from_path()


def test_saved_asset_price_model():
    time_freq = Frequency.WEEKLY
    dsa = YahooFinancialsAdapter(
        ticker='^GSPC',
        frequency=time_freq,
        training_set_date_range=("2010-01-01", "2015-01-01"),
        validation_set_date_range=("2015-01-01", "2019-07-01"))

    params = {'s0': 2058.89990234375,
              'r': 0.002476615753153449, 'σ': 0.02042995488960794,
              't0': parse('2015-01-01', date_formats=['YYYY-mm-dd'])}

    model = DiffusionProcessAssetPriceModel.load(parameters=params,
                                                 time_unit_transformer=IndexedTimeTransformer(
                                                     time_freq=time_freq),
                                                 n_sample_paths=5)
    _test_forecasting(model, dsa)


# test_gbm_process_paths()
# test_gbm_process_params_familiarization()
# test_asset_price_model_estimation()
test_saved_asset_price_model()
