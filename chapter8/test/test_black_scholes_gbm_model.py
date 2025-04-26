
from abc import ABC
from chapter2.stock_price_dataset_adapters import Frequency
from chapter8.options_common import plot_options_surface
from chapter8.black_scholes_gbm_model import BlackScholesOptionsRiskNeutralGBMModel, OptionGreeks
from chapter5.base_forecasting import ForecastResultDisplay
from chapter6.diffusion_model import IndexedTimeTransformer, GeometricBrownianMotionProcess
from dateparser import parse
import numpy as np
from scipy.stats import norm as Φ

import matplotlib.pyplot as plt
import seaborn as sns


def test_stock_paths_and_options():
    plt.style.use("seaborn-v0_8")
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    bmp = GeometricBrownianMotionProcess(
        r=0.0018, σ=0.009, n_sample_paths=1, initial_state=110)

    path_1_display = ForecastResultDisplay(bmp.forecast(T=100), ylabel='S(t)')
    path_1_display.plot_sample_paths(ax=ax1)
    ax1.axhline(y=115)
    ax1.text(x=50, y=116, s='K=115')

    path_2_display = ForecastResultDisplay(bmp.forecast(T=100), ylabel='S(t)')
    path_2_display.plot_sample_paths(ax=ax2)
    ax2.axhline(y=115)
    ax2.text(x=50, y=116, s='K=115')
    plt.show()


class BlackScholesOptionsGBMProcessStatistics(ABC):

    @staticmethod
    def d1(s, t, r, σ, K, T):
        return (np.log(s/K) + (r + 0.5*(σ ** 2)) * (T-t))/(σ * np.sqrt(T-t))

    @staticmethod
    def d2(s, t, r, σ, K, T):
        return BlackScholesOptionsGBMProcessStatistics.d1(s, t, r, σ, K, T) - σ * np.sqrt(T - t)

    class Call(ABC):

        @staticmethod
        def option_value(s, t, r, σ, K, T):
            d1_val = BlackScholesOptionsGBMProcessStatistics.d1(
                s, t, r, σ, K, T)
            d2_val = BlackScholesOptionsGBMProcessStatistics.d2(
                s, t, r, σ, K, T)

            return (s * Φ.cdf(d1_val)) - (K * np.exp(-r*(T-t)) * Φ.cdf(d2_val))

        @staticmethod
        def delta(s, t, r, σ, K, T):
            d1_val = BlackScholesOptionsGBMProcessStatistics.d1(
                s, t, r, σ, K, T)
            return Φ.cdf(d1_val)

        @staticmethod
        def gamma(s, t, r, σ, K, T):
            d1_val = BlackScholesOptionsGBMProcessStatistics.d1(
                s, t, r, σ, K, T)
            return Φ.pdf(d1_val)/(σ * s * np.sqrt(T-t))

        @staticmethod
        def theta(s, t, r, σ, K, T):
            d1_val = BlackScholesOptionsGBMProcessStatistics.d1(
                s, t, r, σ, K, T)
            d2_val = BlackScholesOptionsGBMProcessStatistics.d2(
                s, t, r, σ, K, T)

            return -((σ * s * Φ.pdf(d1_val))/(2.0*np.sqrt(T-t))) \
                - (r * K * np.exp(-r*(T-t)) * Φ.cdf(d2_val))

        @staticmethod
        def vega(s, t, r, σ, K, T):
            d1_val = BlackScholesOptionsGBMProcessStatistics.d1(
                s, t, r, σ, K, T)
            return Φ.pdf(d1_val)*(σ * s * np.sqrt(T-t))

        @staticmethod
        def rho(s, t, r, σ, K, T):
            d2_val = BlackScholesOptionsGBMProcessStatistics.d2(
                s, t, r, σ, K, T)
            return K * np.exp(-r*(T-t)) * Φ.cdf(d2_val)

    class Put(ABC):

        @staticmethod
        def option_value(s, t, r, σ, K, T):
            d1_val = BlackScholesOptionsGBMProcessStatistics.d1(
                s, t, r, σ, K, T)
            d2_val = BlackScholesOptionsGBMProcessStatistics.d2(
                s, t, r, σ, K, T)

            return (K * np.exp(-r*(T-t)) * Φ.cdf(-d2_val)) - (s * Φ.cdf(-d1_val))

        @staticmethod
        def delta(s, t, r, σ, K, T):
            d1_val = BlackScholesOptionsGBMProcessStatistics.d1(
                s, t, r, σ, K, T)
            return Φ.cdf(d1_val) - 1

        @staticmethod
        def gamma(s, t, r, σ, K, T):
            d1_val = BlackScholesOptionsGBMProcessStatistics.d1(
                s, t, r, σ, K, T)
            return Φ.pdf(d1_val)/(σ * s * np.sqrt(T-t))

        @staticmethod
        def theta(s, t, r, σ, K, T):
            d1_val = BlackScholesOptionsGBMProcessStatistics.d1(
                s, t, r, σ, K, T)
            d2_val = BlackScholesOptionsGBMProcessStatistics.d2(
                s, t, r, σ, K, T)

            return -((σ * s * Φ.pdf(-d1_val))/(2.0*np.sqrt(T-t))) \
                + (r*K * np.exp(-r*(T-t))*Φ.cdf(-d2_val))

        @staticmethod
        def vega(s, t, r, σ, K, T):
            d1_val = BlackScholesOptionsGBMProcessStatistics.d1(
                s, t, r, σ, K, T)

            return Φ.pdf(d1_val) * (σ * s * np.sqrt(T-t))

        @staticmethod
        def rho(s, t, r, σ, K, T):
            d2_val = BlackScholesOptionsGBMProcessStatistics.d2(
                s, t, r, σ, K, T)
            return -1.0 * K * np.exp(-r*(T-t)) * Φ.cdf(-d2_val)


def test_call_put_simulation():
    n = 50
    T = 100
    ts = [t_i for t_i in range(T)]
    # Sample asset values
    S = np.linspace(1000, 5000, n)
    time_grid, s_grid = np.meshgrid(ts, S)
    r, σ = 0.009757165676274302, 0.03720863108733509

    V_call = [BlackScholesOptionsGBMProcessStatistics.Call.option_value(
        s=s, t=t, r=r, σ=σ, T=T, K=3000) for s, t in zip(s_grid, time_grid)]
    test_plot_options_surface(t=np.array(time_grid), S=np.array(
        s_grid), V=np.array(V_call), label="Call")

    V_put = [BlackScholesOptionsGBMProcessStatistics.Put.option_value(
        s=s, t=t, r=r, σ=σ, T=T, K=8000) for s, t in zip(s_grid, time_grid)]
    test_plot_options_surface(t=np.array(time_grid), S=np.array(
        s_grid), V=np.array(V_put), label="Put", is_call=False)


def test_greeks_simulation():
    n = 50
    T = 100
    ts = [t_i for t_i in range(T)]
    S = np.linspace(1000, 5000, n)
    time_grid, s_grid = np.meshgrid(ts, S)
    r, σ = 0.009757165676274302, 0.03720863108733509

    V_delta_call = [BlackScholesOptionsGBMProcessStatistics.Call.delta(
        s=s, t=t, r=r, σ=σ, T=T, K=3000) for s, t in zip(s_grid, time_grid)]
    test_plot_options_surface(t=np.array(time_grid), S=np.array(
        s_grid), V=np.array(V_delta_call), label="Delta Call")

    V_gamma_put = [BlackScholesOptionsGBMProcessStatistics.Put.gamma(
        s=s, t=t, r=r, σ=σ, T=T, K=4000) for s, t in zip(s_grid, time_grid)]
    test_plot_options_surface(t=np.array(time_grid), S=np.array(
        s_grid), V=np.array(V_gamma_put), label="Gamma Put", is_call=False)

    V_theta_put = [BlackScholesOptionsGBMProcessStatistics.Put.theta(
        s=s, t=t, r=r, σ=σ, T=T, K=4000) for s, t in zip(s_grid, time_grid)]
    test_plot_options_surface(t=np.array(time_grid), S=np.array(
        s_grid), V=np.array(V_theta_put), label="Theta Put", is_call=False)

    V_theta_call = [BlackScholesOptionsGBMProcessStatistics.Call.theta(
        s=s, t=t, r=r, σ=σ, T=T, K=4000) for s, t in zip(s_grid, time_grid)]
    test_plot_options_surface(t=np.array(time_grid), S=np.array(
        s_grid), V=np.array(V_theta_call), label="Theta Call")

    V_vega_put = [BlackScholesOptionsGBMProcessStatistics.Put.vega(
        s=s, t=t, r=r, σ=σ, T=T, K=4000) for s, t in zip(s_grid, time_grid)]
    test_plot_options_surface(t=np.array(time_grid), S=np.array(
        s_grid), V=np.array(V_vega_put), label="Vega Put")

    V_rho_call = [BlackScholesOptionsGBMProcessStatistics.Call.rho(
        s=s, t=t, r=r, σ=σ, T=T, K=4000) for s, t in zip(s_grid, time_grid)]
    test_plot_options_surface(t=np.array(time_grid), S=np.array(
        s_grid), V=np.array(V_rho_call), label="Rho Call")


def test_plot_options_surface(t, S, V, label, is_call=True):
    plt.style.use("seaborn-v0_8")
    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)
    print(S.shape)
    print(V.shape)
    sns.scatterplot(ax=ax2, x=S.flatten(), y=V.flatten())

    if is_call:
        ax1.plot_surface(t, S, V, rstride=5, cstride=5,
                         cmap=plt.cm.gnuplot2, edgecolor="black")

        ax1.set_xlabel('t')
        ax1.set_ylabel('S')
    else:
        ax1.plot_surface(S, t, V, rstride=5, cstride=5,
                         cmap=plt.cm.gnuplot2, edgecolor="black")

        ax1.set_xlabel('S')
        ax1.set_ylabel('t')

    ax1.set_zlabel(label)

    ax2.set_xlabel('S')
    ax2.set_ylabel(label)

    fig.tight_layout()
    plt.show()


def test_BS_call_process():
    params = {'s0': 2043.93994140625,
              'r': 0.009757165676274302, 'σ': 0.03720863108733509,
              't0': parse('2015-12-01', date_formats=['YYYY-mm-dd'])}

    time_freq = Frequency.MONTHLY
    gbm_options_model = BlackScholesOptionsRiskNeutralGBMModel(
        params,
        IndexedTimeTransformer(time_freq=time_freq))

    result, _ = gbm_options_model.estimate_call(
        expiry_time_T=100,
        strike_price_K=3000.00)

    print('Call Value at 0: ' + str(result.value_at_0))

    result_display = ForecastResultDisplay(
        result.all_values, ylabel=result.label)
    result_display.plot_sample_paths()
    result_display.plot_uncertainity_bounds()

    plot_options_surface(result)


def test_BS_put_process():
    params = {'s0': 2043.93994140625,
              'r': 0.009757165676274302, 'σ': 0.03720863108733509,
              't0': parse('2015-12-01', date_formats=['YYYY-mm-dd'])}

    time_freq = Frequency.MONTHLY
    gbm_options_model = BlackScholesOptionsRiskNeutralGBMModel(params, IndexedTimeTransformer(
        time_freq=time_freq))

    result, _ = gbm_options_model.estimate_put(
        expiry_time_T=100, strike_price_K=3000.00)

    print('Put Value at 0: ' + str(result.value_at_0))

    result_display = ForecastResultDisplay(
        result.all_values, ylabel=result.label)
    result_display.plot_sample_paths()
    result_display.plot_uncertainity_bounds()

    plot_options_surface(result)


def test_BS_delta_for_call_process():
    params = {'s0': 2043.93994140625,
              'r': 0.009757165676274302, 'σ': 0.03720863108733509,
              't0': parse('2015-12-01', date_formats=['YYYY-mm-dd'])}

    time_freq = Frequency.MONTHLY
    gbm_options_model = BlackScholesOptionsRiskNeutralGBMModel(params, IndexedTimeTransformer(
        time_freq=time_freq))

    _, greeks_result = gbm_options_model.estimate_call(
        expiry_time_T=100, strike_price_K=3000.00, greeks=[OptionGreeks.Δ])

    delta_result = greeks_result[OptionGreeks.Δ]
    print('Delta Call Value at 0: ' +
          str(delta_result.value_at_0))
    delta_result_display = ForecastResultDisplay(
        delta_result.all_values, ylabel=delta_result.label)
    delta_result_display.plot_sample_paths()
    delta_result_display.plot_uncertainity_bounds()
    plot_options_surface(delta_result)


def test_BS_gamma_for_call_process():
    params = {'s0': 2043.93994140625,
              'r': 0.009757165676274302, 'σ': 0.03720863108733509,
              't0': parse('2015-12-01', date_formats=['YYYY-mm-dd'])}

    time_freq = Frequency.MONTHLY
    gbm_options_model = BlackScholesOptionsRiskNeutralGBMModel(params, IndexedTimeTransformer(
        time_freq=time_freq))

    _, greeks_result = gbm_options_model.estimate_call(
        expiry_time_T=100, strike_price_K=1500.00, greeks=[OptionGreeks.Γ])

    gamma_result = greeks_result[OptionGreeks.Γ]
    print('Gamma Call Value at 0: ' +
          str(gamma_result.value_at_0))
    gamma_result_display = ForecastResultDisplay(
        gamma_result.all_values, ylabel=gamma_result.label)
    gamma_result_display.plot_sample_paths()
    gamma_result_display.plot_uncertainity_bounds()
    plot_options_surface(gamma_result)


def test_BS_delta_for_put_process():
    params = {'s0': 2043.93994140625,
              'r': 0.009757165676274302, 'σ': 0.03720863108733509,
              't0': parse('2015-12-01', date_formats=['YYYY-mm-dd'])}

    time_freq = Frequency.MONTHLY
    gbm_options_model = BlackScholesOptionsRiskNeutralGBMModel(params, IndexedTimeTransformer(
        time_freq=time_freq))

    _, greeks_result = gbm_options_model.estimate_put(
        expiry_time_T=100, strike_price_K=1500.00, greeks=[OptionGreeks.Δ])

    delta_result = greeks_result[OptionGreeks.Δ]
    print('Delta Put Value at 0: ' +
          str(delta_result.value_at_0))
    delta_result_display = ForecastResultDisplay(
        delta_result.all_values, ylabel=delta_result.label)
    delta_result_display.plot_sample_paths()
    delta_result_display.plot_uncertainity_bounds()
    plot_options_surface(delta_result)


def test_BS_gamma_for_put_process():
    params = {'s0': 2043.93994140625,
              'r': 0.009757165676274302, 'σ': 0.03720863108733509,
              't0': parse('2015-12-01', date_formats=['YYYY-mm-dd'])}

    time_freq = Frequency.MONTHLY
    gbm_options_model = BlackScholesOptionsRiskNeutralGBMModel(params, IndexedTimeTransformer(
        time_freq=time_freq))

    _, greeks_result = gbm_options_model.estimate_put(
        expiry_time_T=100, strike_price_K=4000.00, greeks=[OptionGreeks.Γ])

    gamma_result = greeks_result[OptionGreeks.Γ]
    print('Gamma Put Value at 0: ' +
          str(gamma_result.value_at_0))
    gamma_result_display = ForecastResultDisplay(
        gamma_result.all_values, ylabel=gamma_result.label)
    gamma_result_display.plot_sample_paths()
    gamma_result_display.plot_uncertainity_bounds()
    plot_options_surface(gamma_result)


def test_BS_vega_for_call_process():
    params = {'s0': 2043.93994140625,
              'r': 0.009757165676274302, 'σ': 0.03720863108733509,
              't0': parse('2015-12-01', date_formats=['YYYY-mm-dd'])}

    time_freq = Frequency.MONTHLY
    gbm_options_model = BlackScholesOptionsRiskNeutralGBMModel(params, IndexedTimeTransformer(
        time_freq=time_freq))

    _, greeks_result = gbm_options_model.estimate_call(
        expiry_time_T=100, strike_price_K=1500.00, greeks=[OptionGreeks.Κ])

    vega_result = greeks_result[OptionGreeks.Κ]
    print('Vega Call Value at 0: ' +
          str(vega_result.value_at_0))
    vega_result_display = ForecastResultDisplay(
        vega_result.all_values, ylabel=vega_result.label)
    vega_result_display.plot_sample_paths()
    vega_result_display.plot_uncertainity_bounds()
    plot_options_surface(vega_result)


def test_BS_vega_for_put_process():
    params = {'s0': 2043.93994140625,
              'r': 0.009757165676274302, 'σ': 0.03720863108733509,
              't0': parse('2015-12-01', date_formats=['YYYY-mm-dd'])}

    time_freq = Frequency.MONTHLY
    gbm_options_model = BlackScholesOptionsRiskNeutralGBMModel(params, IndexedTimeTransformer(
        time_freq=time_freq))

    _, greeks_result = gbm_options_model.estimate_put(
        expiry_time_T=100, strike_price_K=4000.00, greeks=[OptionGreeks.Κ])

    vega_result = greeks_result[OptionGreeks.Κ]
    print('Vega Put Value at 0: ' +
          str(vega_result.value_at_0))
    vega_result_display = ForecastResultDisplay(
        vega_result.all_values, ylabel=vega_result.label)
    vega_result_display.plot_sample_paths()
    vega_result_display.plot_uncertainity_bounds()
    plot_options_surface(vega_result)


def test_BS_rho_for_call_process():
    params = {'s0': 2043.93994140625,
              'r': 0.009757165676274302, 'σ': 0.03720863108733509,
              't0': parse('2015-12-01', date_formats=['YYYY-mm-dd'])}

    time_freq = Frequency.MONTHLY
    gbm_options_model = BlackScholesOptionsRiskNeutralGBMModel(params, IndexedTimeTransformer(
        time_freq=time_freq))

    _, greeks_result = gbm_options_model.estimate_call(
        expiry_time_T=100, strike_price_K=4000.00, greeks=[OptionGreeks.Ρ])

    rho_result = greeks_result[OptionGreeks.Ρ]
    print('Rho Call Value at 0: ' +
          str(rho_result.value_at_0))
    rho_result_display = ForecastResultDisplay(
        rho_result.all_values, ylabel=rho_result.label)
    rho_result_display.plot_sample_paths()
    rho_result_display.plot_uncertainity_bounds()
    plot_options_surface(rho_result)


def test_BS_rho_for_put_process():
    params = {'s0': 2043.93994140625,
              'r': 0.009757165676274302, 'σ': 0.03720863108733509,
              't0': parse('2015-12-01', date_formats=['YYYY-mm-dd'])}

    time_freq = Frequency.MONTHLY
    gbm_options_model = BlackScholesOptionsRiskNeutralGBMModel(params, IndexedTimeTransformer(
        time_freq=time_freq))

    _, greeks_result = gbm_options_model.estimate_put(
        expiry_time_T=100, strike_price_K=1500.00, greeks=[OptionGreeks.Ρ])

    rho_result = greeks_result[OptionGreeks.Ρ]
    print('Rho Put Value at 0: ' +
          str(rho_result.value_at_0))
    rho_result_display = ForecastResultDisplay(
        rho_result.all_values, ylabel=rho_result.label)
    rho_result_display.plot_sample_paths()
    rho_result_display.plot_uncertainity_bounds()
    plot_options_surface(rho_result)


def test_BS_theta_for_call_process():
    params = {'s0': 2043.93994140625,
              'r': 0.009757165676274302, 'σ': 0.03720863108733509,
              't0': parse('2015-12-01', date_formats=['YYYY-mm-dd'])}

    time_freq = Frequency.MONTHLY
    gbm_options_model = BlackScholesOptionsRiskNeutralGBMModel(params, IndexedTimeTransformer(
        time_freq=time_freq))

    _, greeks_result = gbm_options_model.estimate_call(
        expiry_time_T=100, strike_price_K=3000.00, greeks=[OptionGreeks.Θ])

    theta_result = greeks_result[OptionGreeks.Θ]
    print('Theta Call Value at 0: ' +
          str(theta_result.value_at_0))
    theta_result_display = ForecastResultDisplay(
        theta_result.all_values, ylabel=theta_result.label)
    theta_result_display.plot_sample_paths()
    theta_result_display.plot_uncertainity_bounds()
    plot_options_surface(theta_result)


def test_BS_theta_for_put_process():
    params = {'s0': 2043.93994140625,
              'r': 0.009757165676274302, 'σ': 0.03720863108733509,
              't0': parse('2015-12-01', date_formats=['YYYY-mm-dd'])}

    time_freq = Frequency.MONTHLY
    gbm_options_model = BlackScholesOptionsRiskNeutralGBMModel(params, IndexedTimeTransformer(
        time_freq=time_freq))

    _, greeks_result = gbm_options_model.estimate_put(
        expiry_time_T=100, strike_price_K=1500.00, greeks=[OptionGreeks.Θ])

    theta_result = greeks_result[OptionGreeks.Θ]
    print('Theta Put Value at 0: ' +
          str(theta_result.value_at_0))
    theta_result_display = ForecastResultDisplay(
        theta_result.all_values, ylabel=theta_result.label)
    theta_result_display.plot_sample_paths()
    theta_result_display.plot_uncertainity_bounds()
    plot_options_surface(theta_result)
