from chapter7.non_parametric_jump_process_model import GaussianKernelJumpProcess, GaussianKernelJumpAssetPriceModel, gaussian_kernel
from chapter5.base_forecasting import ForecastResultDisplay, AssetPriceBackTesting
import chapter7.visualization as vz7
from chapter2.stock_price_dataset_adapters import YahooFinancialsAdapter, Frequency
from chapter6.diffusion_model import IndexedTimeTransformer
import numpy as np
from scipy.stats import norm


def test_gaussian_kernel_density_estimate():
    # Create a mixture of data points from three different distributions
    x_arr = sorted(np.concatenate((norm.rvs(loc=150, scale=100, size=100),
                                   norm.rvs(loc=15, scale=10, size=100),
                                   norm.rvs(loc=1000, scale=200, size=5))))
    n = len(x_arr)

    def f(x, h):
        return np.sum(gaussian_kernel((x_arr-x)/h))/(n*h)

    # Setup a pool of bandwdiths to be tested
    h_1 = 15
    h_2 = 30
    h_3 = 100
    h_4 = 60
    vz7.plot_kde_with_hs((x_arr, [f(x, h_1) for x in x_arr], h_1),
                         (x_arr, [f(x, h_2) for x in x_arr], h_2),
                         (x_arr, [f(x, h_3) for x in x_arr], h_3),
                         (x_arr, [f(x, h_4) for x in x_arr], h_4))


def test_non_parametric_jump_process_full():
    ticker = '^GSPC'  # S&P 500 Index
    time_freq = Frequency.DAILY
    yfa = YahooFinancialsAdapter(
        ticker=ticker,
        frequency=time_freq,
        training_set_date_range=("2010-01-01", "2012-01-01"),
        validation_set_date_range=("2012-01-01", "2013-01-01"))

    # preoare dataset for kernel function
    _s_t = yfa.training_set
    initial_state = float(np.log(_s_t.tail(1)['stock price']))
    _s_t_1 = _s_t.shift(1)
    _x_t = np.log(_s_t['stock price'])
    _x_t_1 = np.log(_s_t_1['stock price'])
    _increments = _x_t - _x_t_1
    _x_t = _x_t.drop([_x_t.index[0]], inplace=False)
    _increments = _increments.drop([_increments.index[0]], inplace=False)

    n_sample_paths = 5
    h = 5.5
    σ_2_J = 0.044
    gjp = GaussianKernelJumpProcess(
        h=h, x_arr=_x_t, increments=_increments, σ_2_J=σ_2_J, initial_state=initial_state, n_sample_paths=n_sample_paths)
    forecast_result = gjp.forecast(T=250)
    ap_back_testing = AssetPriceBackTesting(
        s_true=yfa.validation_set, s_forecast=forecast_result.mean_path)
    print('RMSE: ' + str(ap_back_testing.rmse_score))

    vz7.plot_full_testing_results(ForecastResultDisplay(
        forecast_result, ylabel='S(t)'), ap_back_testing, h=h, σ_2_J=σ_2_J)


def test_non_parametric_jump_process_params_familiarization():

    ticker = '^GSPC'  # S&P 500 Index
    time_freq = Frequency.DAILY
    yfa = YahooFinancialsAdapter(
        ticker=ticker,
        frequency=time_freq,
        training_set_date_range=("2010-01-01", "2011-01-01"),
        validation_set_date_range=("2016-01-01", "2018-07-01"))

    # preoare dataset for kernel function
    _s_t = yfa.training_set
    _s_t_1 = _s_t.shift(1)
    _x_t = np.log(_s_t['stock price'])
    _x_t_1 = np.log(_s_t_1['stock price'])
    _increments = _x_t - _x_t_1
    _x_t = _x_t.drop([_x_t.index[0]], inplace=False)
    _increments = _increments.drop([_increments.index[0]], inplace=False)

    initial_state = np.log(2000.00)

    h_1 = np.log(1000)
    σ_2_J_1 = np.log(3000)
    gjp_1 = GaussianKernelJumpProcess(
        h=h_1, x_arr=_x_t, increments=_increments, σ_2_J=σ_2_J_1, initial_state=initial_state)
    result_display_1 = ForecastResultDisplay(
        result=gjp_1.forecast(T=100), ylabel='S(t)')

    h_2 = np.log(1.5)
    σ_2_J_2 = np.log(3000)
    gjp_2 = GaussianKernelJumpProcess(
        h=h_2, x_arr=_x_t, increments=_increments, σ_2_J=σ_2_J_2, initial_state=initial_state)
    result_display_2 = ForecastResultDisplay(
        result=gjp_2.forecast(T=100), ylabel='S(t)')

    h_3 = np.log(10000)
    σ_2_J_3 = np.log(30)
    gjp_3 = GaussianKernelJumpProcess(
        h=h_3, x_arr=_x_t, increments=_increments, σ_2_J=σ_2_J_3, initial_state=initial_state)
    result_display_3 = ForecastResultDisplay(
        result=gjp_3.forecast(T=100), ylabel='S(t)')

    # h_4 = np.log(10000)
    h_4 = np.log(1.001)
    σ_2_J_4 = np.log(30000000)
    gjp_4 = GaussianKernelJumpProcess(
        h=h_4, x_arr=_x_t, increments=_increments, σ_2_J=σ_2_J_4, initial_state=initial_state)
    result_display_4 = ForecastResultDisplay(
        result=gjp_4.forecast(T=100), ylabel='S(t)')

    vz7.plot_gaussian_kernel_process_parameter_famailiarization(
        (result_display_1, round(h_1), round(σ_2_J_1)
         ), (result_display_2, round(h_2), round(σ_2_J_2)),
        (result_display_3, round(h_3), round(σ_2_J_3)), (result_display_4, round(h_4), round(σ_2_J_4)))


def test_back_testing_with_different_settings():
    ticker = '^GSPC'  # S&P 500 Index
    time_freq = Frequency.DAILY
    yfa = YahooFinancialsAdapter(
        ticker=ticker,
        frequency=time_freq,
        training_set_date_range=("2010-01-01", "2012-01-01"),
        validation_set_date_range=("2012-01-01", "2013-01-01"))

    _s_t = yfa.training_set
    _s_t_1 = _s_t.shift(1)
    _x_t = np.log(_s_t['stock price'])
    _x_t_1 = np.log(_s_t_1['stock price'])
    _increments = _x_t - _x_t_1
    _x_t = _x_t.drop([_x_t.index[0]], inplace=False)
    _increments = _increments.drop([_increments.index[0]], inplace=False)

    initial_state = 7.136960402261725

    n_sample_paths = 5
    h_1 = 9
    σ_2_J_1 = 3
    gjp_1 = GaussianKernelJumpProcess(
        h=h_1, x_arr=_x_t, increments=_increments, σ_2_J=σ_2_J_1, initial_state=initial_state, n_sample_paths=n_sample_paths)
    ap_back_testing_1 = AssetPriceBackTesting(
        s_true=yfa.validation_set, s_forecast=gjp_1.forecast(T=250).mean_path)
    print('RMSE: ' + str(ap_back_testing_1.rmse_score))

    h_2 = 6
    σ_2_J_2 = 1.2
    gjp_2 = GaussianKernelJumpProcess(
        h=h_2, x_arr=_x_t, increments=_increments, σ_2_J=σ_2_J_2, initial_state=initial_state, n_sample_paths=n_sample_paths)
    ap_back_testing_2 = AssetPriceBackTesting(
        s_true=yfa.validation_set, s_forecast=gjp_2.forecast(T=250).mean_path)
    print('RMSE: ' + str(ap_back_testing_2.rmse_score))

    h_3 = 10
    σ_2_J_3 = 2
    gjp_3 = GaussianKernelJumpProcess(
        h=h_3, x_arr=_x_t, increments=_increments, σ_2_J=σ_2_J_3, initial_state=initial_state, n_sample_paths=n_sample_paths)
    ap_back_testing_3 = AssetPriceBackTesting(
        s_true=yfa.validation_set, s_forecast=gjp_3.forecast(T=250).mean_path)
    print('RMSE: ' + str(ap_back_testing_3.rmse_score))

    h_4 = 5.5
    σ_2_J_4 = 0.044

    gjp_4 = GaussianKernelJumpProcess(
        h=h_4, x_arr=_x_t, increments=_increments, σ_2_J=σ_2_J_4, initial_state=initial_state, n_sample_paths=n_sample_paths)
    ap_back_testing_4 = AssetPriceBackTesting(
        s_true=yfa.validation_set, s_forecast=gjp_4.forecast(T=250).mean_path)
    print('RMSE: ' + str(ap_back_testing_4.rmse_score))

    vz7.plot_back_testing_results(
        (ap_back_testing_1, round(h_1, 1), round(σ_2_J_1, 2)
         ), (ap_back_testing_2, round(h_2, 1), round(σ_2_J_2, 2)),
        (ap_back_testing_3, round(h_3, 1), round(σ_2_J_3, 2)), (ap_back_testing_4, round(h_4, 1), round(σ_2_J_4, 2)))


def test_kernel_gaussian_model_with_auto_bandwidth_selection():
    ticker = '^GSPC'  # S&P 500 Index
    time_freq = Frequency.DAILY
    yfa = YahooFinancialsAdapter(
        ticker=ticker,
        frequency=time_freq,
        training_set_date_range=("2011-01-01", "2014-01-01"),
        validation_set_date_range=("2014-01-01", "2015-01-01"))

    kernel_jump_model = GaussianKernelJumpAssetPriceModel(time_unit_transformer=IndexedTimeTransformer(time_freq=time_freq),
                                                          bandwidth='auto',
                                                          bandwidth_bounds=[(
                                                              1.0, 10.0)],
                                                          asset_price_dataset_adapter=yfa)
    print(kernel_jump_model.parameters_)

    result = kernel_jump_model.forecast(T=252, prob_dist_viz_required=False)

    ap_back_testing = AssetPriceBackTesting(
        s_true=yfa.validation_set, s_forecast=result.mean_path)

    print('RMSE: ' + str(ap_back_testing.rmse_score))
    vz7.plot_full_testing_results(ForecastResultDisplay(
        result, ylabel='S(t)'), ap_back_testing, h=kernel_jump_model.parameters_['h'], σ_2_J=kernel_jump_model.parameters_['σ_2_J'])


# test_back_testing_with_different_settings()
# test_non_parametric_jump_process_params_familiarization()
# test_kernel_gaussian_model_with_auto_bandwidth_selection()
# test_non_parametric_jump_process_full()
test_gaussian_kernel_density_estimate()
