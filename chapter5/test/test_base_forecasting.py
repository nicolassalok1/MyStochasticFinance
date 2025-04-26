from chapter5.base_forecasting import BrownianMotionProcess, ForecastResultDisplay, PoissonProcess
import chapter5.visualization as vz5
from chapter5.scaled_symmetric_random_walk import ScaledSymmetricRandomWalkModel


def test_bm_process_paths():
    bmp = BrownianMotionProcess(μ=10, σ=300, n_sample_paths=30)
    result = bmp.forecast(T=500)
    result_display = ForecastResultDisplay(result)
    result_display.plot_sample_paths()
    result_display.plot_mean_path()
    result_display.plot_uncertainity_bounds()


def test_poisson_process_sample_paths():
    λ_1 = 0.009
    n_sample_paths = 10
    pp_1 = PoissonProcess(λ=λ_1, n_sample_paths=n_sample_paths)
    result_display_1 = ForecastResultDisplay(pp_1.forecast(T=500))

    λ_2 = 0.03
    n_sample_paths = 10
    pp_2 = PoissonProcess(λ=λ_2, n_sample_paths=n_sample_paths)
    result_display_2 = ForecastResultDisplay(pp_2.forecast(T=500))

    λ_3 = 0.06
    n_sample_paths = 10
    pp_3 = PoissonProcess(λ=λ_3, n_sample_paths=n_sample_paths)
    result_display_3 = ForecastResultDisplay(pp_3.forecast(T=500))

    λ_4 = 0.09
    n_sample_paths = 10
    pp_4 = PoissonProcess(λ=λ_4, n_sample_paths=n_sample_paths)
    result_display_4 = ForecastResultDisplay(pp_4.forecast(T=500))

    vz5.plot_all_sample_paths_for_pp(
        (result_display_1, λ_1), (result_display_2, λ_2),
        (result_display_3, λ_3), (result_display_4, λ_4))


def test_bm_process_sample_paths():

    μ_1 = 10
    σ_1 = 300
    n_sample_paths = 10
    bmp_1 = BrownianMotionProcess(μ=μ_1, σ=σ_1, n_sample_paths=n_sample_paths)
    result_display_1 = ForecastResultDisplay(bmp_1.forecast(T=500))

    μ_2 = -15
    σ_2 = 200
    bmp_2 = BrownianMotionProcess(μ=μ_2, σ=σ_2, n_sample_paths=n_sample_paths)
    result_display_2 = ForecastResultDisplay(bmp_2.forecast(T=500))

    μ_3 = 6
    σ_3 = 50
    bmp_3 = BrownianMotionProcess(μ=μ_3, σ=σ_3, n_sample_paths=n_sample_paths)
    result_display_3 = ForecastResultDisplay(bmp_3.forecast(T=500))

    μ_4 = -0.03
    σ_4 = 1
    bmp_4 = BrownianMotionProcess(μ=μ_4, σ=σ_4, n_sample_paths=n_sample_paths)
    result_display_4 = ForecastResultDisplay(bmp_4.forecast(T=500))

    vz5.plot_all_sample_paths_for_bm(
        (result_display_1, μ_1, σ_1), (result_display_2, μ_2, σ_2),
        (result_display_3, μ_3, σ_3), (result_display_4, μ_4, σ_4))


def test_bm_process_mean_paths():

    μ_1 = 10
    σ_1 = 300
    bmp_1 = BrownianMotionProcess(μ=μ_1, σ=σ_1, n_sample_paths=2)
    result_display_1 = ForecastResultDisplay(bmp_1.forecast(T=500))

    μ_2 = -15
    σ_2 = 200
    bmp_2 = BrownianMotionProcess(μ=μ_2, σ=σ_2, n_sample_paths=2)
    result_display_2 = ForecastResultDisplay(bmp_2.forecast(T=500))

    μ_3 = 6
    σ_3 = 50
    bmp_3 = BrownianMotionProcess(μ=μ_3, σ=σ_3, n_sample_paths=2)
    result_display_3 = ForecastResultDisplay(bmp_3.forecast(T=500))

    μ_4 = -0.03
    σ_4 = 1
    bmp_4 = BrownianMotionProcess(μ=μ_4, σ=σ_4, n_sample_paths=2)
    result_display_4 = ForecastResultDisplay(bmp_4.forecast(T=500))

    vz5.plot_all_mean_paths_for_bm(
        (result_display_1, μ_1, σ_1), (result_display_2, μ_2, σ_2),
        (result_display_3, μ_3, σ_3), (result_display_4, μ_4, σ_4))


def test_scaled_random_walk():
    m = ScaledSymmetricRandomWalkModel(scale_factor=2, total_time=20)
    m.plot_scaled_walk()


# test_scaled_random_walk()
# test_bm_process_sample_paths()
# test_bm_process_mean_paths()
test_poisson_process_sample_paths()
